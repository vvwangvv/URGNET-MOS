from __future__ import annotations
from transformers import AutoFeatureExtractor, AutoModel, WhisperModel
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from typing import List
from urgent_mos.model.common import scale_grad


class AudioEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        gradient_scale: float = 0.1,
        layer_aggregation: bool = True,
    ):
        super().__init__()
        if model_name == "giangndm/qwen3-30b-omni-audio-encoder":
            self.model = Qwen3OmniMoeAudioEncoder.from_pretrained(model_name)
            self.model._get_feat_extract_output_lengths = _get_feat_extract_output_lengths
        elif model_name == "Atotti/Kimi-Audio-Whisper-Encoder":
            self.model = WhisperModel.from_pretrained(model_name, trust_remote_code=True).encoder
        elif model_name == "Atotti/AFWhisper":
            self.model = Qwen2AudioEncoder.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.model_type = self.model.config.model_type
        if model_name == "Atotti/AFWhisper":
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
        else:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

        if layer_aggregation:
            self.weights = nn.Parameter(torch.zeros(self.model.config.num_hidden_layers + 1))
        else:
            self.weights = None
        self.freeze = freeze
        self.gradient_scale = gradient_scale
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.prev_params = None

    def forward(self, audios: List[float["t"]]):
        kwargs = {"padding": True, "return_attention_mask": True, "sampling_rate": 16000, "return_tensors": "pt"}
        if self.model_type in ["qwen2_audio_encoder", "whisper"]:
            kwargs["padding"] = "max_length"
        inputs = self.feature_extractor([audio.cpu().numpy() for audio in audios], **kwargs).to(self.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=self.dtype)
        input_lengths = inputs.attention_mask.sum(dim=1)

        inputs = self.process_inputs(inputs)
        context = contextlib.nullcontext() if not self.freeze else torch.no_grad()

        if self.weights is not None:
            with context:
                hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states
                try:
                    hidden_states = torch.stack(hidden_states, dim=1)  # [b, l, t, d]
                except:
                    breakpoint()
                hidden_states = scale_grad(hidden_states, self.gradient_scale)
            output = hidden_states * rearrange(F.softmax(self.weights, dim=0), "l -> 1 l 1 1")
            output = output.sum(dim=2)  # [b ,t ,d]
        else:
            with context:
                output = self.model(**inputs).last_hidden_state
        output_lengths = self.model._get_feat_extract_output_lengths(input_lengths)
        output, output_lengths = self.process_output(output, output_lengths)
        return output, output_lengths

    def process_inputs(self, inputs):
        if self.model_type == "qwen3_omni_moe_audio_encoder":
            inputs["feature_lens"] = inputs.pop("attention_mask").sum(dim=1)
            inputs["input_features"] = torch.cat(
                [feat[:, :feat_len] for feat, feat_len in zip(inputs["input_features"], inputs["feature_lens"])], dim=1
            )
        elif self.model_type in ["qwen2_audio_encoder", "whisper"]:
            del inputs["attention_mask"]
        return inputs

    def process_output(self, output, output_lengths):
        if self.model_type == "qwen3_omni_moe_audio_encoder":
            output = output.split(output_lengths.tolist(), dim=0)
            output = pad_sequence(output, batch_first=True)
        elif self.model_type in ["qwen2_audio_encoder", "whisper"]:
            if isinstance(output_lengths, tuple):
                output_lengths = output_lengths[1]
            output = output[:, : output_lengths.max(), :]
        return output, output_lengths

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def output_dim(self):
        if self.model_type == "qwen3_omni_moe_audio_encoder":
            return self.model.config.output_dim
        elif self.model_type in ["qwen2_audio_encoder", "whisper"]:
            return self.model.config.d_model
        elif self.model_type == "wavlm":
            return self.model.config.hidden_size
        else:
            raise NotImplementedError(f"output_dim not implemented for model type {self.model_type}")


class FusedAudioEncoder(nn.Module):
    def __init__(self, feature_extractors: List[AudioEncoder], output_dim: int):
        super().__init__()
        self.feature_extractors = nn.ModuleList(feature_extractors)
        self.feat_proj = nn.Sequential(
            nn.Linear(sum([feature_extractor.output_dim for feature_extractor in self.feature_extractors]), output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, audios: List[float["t"]]):
        feats_list, feats_lengths_list = [], []
        for feature_extractor in self.feature_extractors:
            feats, feats_lengths = feature_extractor(audios)
            feats_list.append(feats)
            feats_lengths_list.append(feats_lengths)
        feats_list, feats_lengths_list = self.interpolate_to_longest(feats_list, feats_lengths_list)
        feats, feats_lengths = torch.cat(feats_list, dim=-1), feats_lengths_list[0]
        feats = self.feat_proj(feats)
        return feats, feats_lengths

    def interpolate_to_longest(self, feats_list: List[float["b t d"]], feats_lengths_list: List[torch.Tensor]):
        bsz = feats_list[0].shape[0]
        new_feats_list, new_feats_lengths_list = [], []
        for i in range(0, len(self.feature_extractors)):
            new_feats, new_feats_lengths = [], []
            for b in range(0, bsz):
                max_length = max([feats_lengths[b] for feats_lengths in feats_lengths_list])
                feat, feat_len = feats_list[i][b], feats_lengths_list[i][b]
                new_feat = (
                    F.interpolate(
                        feat[:feat_len].transpose(0, 1).unsqueeze(0),
                        size=max_length,
                        mode="linear",
                        align_corners=False,
                    )
                    .transpose(1, 2)
                    .squeeze(0)
                )
                new_feats.append(new_feat)
                new_feats_lengths.append(max_length)
            new_feats_list.append(pad_sequence(new_feats, batch_first=True))
            new_feats_lengths_list.append(torch.tensor(new_feats_lengths, device=self.device))
        return new_feats_list, new_feats_lengths_list

    @property
    def device(self):
        return next(self.parameters()).device
