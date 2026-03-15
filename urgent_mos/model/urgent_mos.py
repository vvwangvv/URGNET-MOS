from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from huggingface_hub import preupload_lfs_files
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn as nn
from urgent_mos.model.audio_encoder import FusedAudioEncoder
from urgent_mos.utils import lengths2padding_mask
from urgent_mos.model.common import RangeActivation


class ComparativeScorePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pooler: nn.Module,
        encoder: nn.Module,
        num_heads: int = 4,
        min_cmos_for_tie: float = 0.5,
        regression_weight: float = 0.5,
        classification_weight: float = 0.5,
    ):
        super().__init__()
        self.min_cmos_for_tie = min_cmos_for_tie
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.pooler = pooler
        self.encoder = encoder
        self.classification_head = nn.Linear(input_dim, 3)
        self.regression_head = nn.Linear(input_dim, 1)
        self.loss_fn_classification = nn.CrossEntropyLoss(reduction="none")
        self.loss_fn_regression = nn.MSELoss(reduction="none")
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        assert self.regression_weight + self.classification_weight == 1

    def forward(
        self, feats1: float["b t d"], feats1_lengths: int["b"], feats2: float["b t d"], feats2_lengths: int["b"]
    ) -> tuple[float["b"], float["b"]]:
        if self.encoder is not None:
            feats1 = self.encoder(feats1, src_key_padding_mask=lengths2padding_mask(feats1_lengths))
            feats2 = self.encoder(feats2, src_key_padding_mask=lengths2padding_mask(feats2_lengths))
        cross_attn_output, _ = self.cross_attn(
            feats1, feats2, feats2, key_padding_mask=lengths2padding_mask(feats2_lengths)
        )
        pooled, _ = self.pooler(cross_attn_output, feats1_lengths)
        preds_cls = preds_reg = None
        if self.regression_weight > 0:
            preds_reg = self.regression_head(pooled).squeeze(-1)
        if self.classification_weight > 0:
            preds_cls = self.classification_head(pooled)
        return preds_reg, preds_cls

    def compute_loss(
        self, preds_reg: float["b"], preds_cls: float["b"], targets: float["b"]
    ) -> tuple[torch.Tensor, dict, dict]:
        # targets: ground-truth CMOS (B,); derive class labels: 0=A wins, 1=B wins, 2=tie
        B = targets.shape[0]
        targets_cls = torch.zeros(B, dtype=torch.long, device=targets.device)
        targets_cls[targets < -self.min_cmos_for_tie] = 1  # B wins
        targets_cls[targets.abs() < self.min_cmos_for_tie] = 2  # tie
        loss_reg = self.loss_fn_regression(preds_reg.squeeze(-1), targets).mean()
        loss_cls = self.loss_fn_classification(preds_cls, targets_cls).mean()
        loss = loss_reg * self.regression_weight + loss_cls * self.classification_weight

        correct_reg = ((preds_reg > 0) == (targets > 0)).sum()
        correct_cls = (preds_cls.argmax(dim=-1) == targets_cls).sum()

        info = {
            "loss_regression": loss_reg.detach().cpu().item(),
            "loss_classification": loss_cls.detach().cpu().item(),
            "acc_batch_regression": correct_reg / B,
            "acc_batch_classification": correct_cls / B,
        }
        others = {
            "correct_items_regression": correct_reg.detach().cpu().item(),
            "correct_items_classification": correct_cls.detach().cpu().item(),
            "total_items": B,
        }
        return loss, info, others

    @property
    def device(self):
        return next(self.parameters()).device


class AbsoluteScorePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pooler: nn.Module,
        encoder: nn.Module,
        min_value: float,
        max_value: float,
        criterion: str = "mse",
    ):
        super().__init__()
        self.pooler = pooler
        self.encoder = encoder
        self.prediction_head = nn.Linear(input_dim, 1)
        self.activation = RangeActivation(min_value=min_value, max_value=max_value)
        if criterion == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif criterion == "huber":
            self.loss_fn = nn.HuberLoss(reduction="none", delta=0.5)

    def forward(self, feats: float["b t d"], feats_lengths: int["b"], return_frame_scores: bool = False):
        if self.encoder is not None:
            feats = self.encoder(feats, src_key_padding_mask=lengths2padding_mask(feats_lengths))
        pooled, attn_and_before_pooling = self.pooler(feats, feats_lengths, return_frame_scores=return_frame_scores)
        preds = self.activation(self.prediction_head(pooled).squeeze(-1))

        if return_frame_scores:
            attn, before_pooling = attn_and_before_pooling
            attn = attn.detach().cpu().tolist()
            preds_frame_scores = (
                self.activation(self.prediction_head(before_pooling).squeeze(-1)).detach().cpu().tolist()
            )
            attn = [attn_item[: feat_length.item()] for attn_item, feat_length in zip(attn, feats_lengths)]
            preds_frame_scores = [
                preds_frame_score[: feat_length.item()]
                for preds_frame_score, feat_length in zip(preds_frame_scores, feats_lengths)
            ]
            return preds, (attn, preds_frame_scores)
        return preds, None

    def compute_loss(self, preds, targets: float["b"]) -> tuple[torch.Tensor, dict, dict]:
        targets_mask = ~torch.isnan(targets)
        preds, targets = preds[targets_mask], targets[targets_mask]
        loss = self.loss_fn(preds, targets).mean()
        return loss, {}, {}


class UrgentMOS(nn.Module):
    def __init__(
        self,
        feature_extractor: FusedAudioEncoder,
        category2encoder: Dict[str, torch.nn.Module],
        category2metrics: Dict[str, Dict[str, float]],
        metric2comparative_score_predictor: Dict[str, ComparativeScorePredictor],
        metric2absolute_score_predictor: Dict[str, AbsoluteScorePredictor],
        shared_encoder: torch.nn.Module | None = None,
        use_symmetric_preference: bool = True,
        absolute_score_loss_weight: float = 0.5,
        comparative_score_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.category2encoder = torch.nn.ModuleDict(category2encoder)
        self.shared_encoder = shared_encoder
        self.metric2comparative_score_predictor = torch.nn.ModuleDict(metric2comparative_score_predictor)
        self.metric2absolute_score_predictor = torch.nn.ModuleDict(metric2absolute_score_predictor)

        self.category2metrics = category2metrics
        self.use_symmetric_preference = use_symmetric_preference
        self.all_metrics = {k: v for metrics in category2metrics.values() for k, v in metrics.items()}

        self.absolute_score_loss_weight = absolute_score_loss_weight
        self.comparative_score_loss_weight = comparative_score_loss_weight
        assert self.absolute_score_loss_weight + self.comparative_score_loss_weight == 1

    def _collect_absolute_score_items(
        self,
        feats: float["b t d"],
        feats_lengths: float["b"],
        items: List[float["b t d"], int["b"], dict[str, float["b"]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        feats = torch.stack([feats[i] for i, _ in items])
        feats_lengths = torch.stack([feats_lengths[i] for i, _ in items])
        feats = feats[:, : feats_lengths.max().item(), :]
        targets: dict[str, float["b"]] = {
            m: torch.tensor([item[1].get(m, float("nan")) for item in items], dtype=torch.float32, device=self.device)
            for m in self.all_metrics
        }
        return feats, feats_lengths, targets

    def _collect_comparative_score_items(
        self,
        feats: float["b t d"],
        feats_lengths: float["b"],
        items: List[int, int, dict[str, float]],
    ) -> Optional[Tuple[float["b t d"], float["b"], float["b t d"], float["b"], Dict[str, float["b"]]]]:
        feats1 = torch.stack([feats[i] for i, j, _ in items])
        feats1_lengths = torch.stack([feats_lengths[i] for i, j, _ in items])
        feats2 = torch.stack([feats[j] for i, j, _ in items])
        feats2_lengths = torch.stack([feats_lengths[j] for i, j, _ in items])
        max_length = max(feats1_lengths.max().item(), feats2_lengths.max().item())
        # NOTE: here we truncate the feats to the max length of BOTH feats1 and feats2
        #       the max of feats1_lengths might be larger than feats1 actually length, but it's for convenience of symmetric preference
        feats1, feats2 = feats1[:, :max_length, :], feats2[:, :max_length, :]
        comparative_metrics = [m for m in self.all_metrics if self.all_metrics[m].get("comparative_score", False)]
        targets: dict[str, float["b"]] = {
            m: torch.tensor([item[2].get(m, float("nan")) for item in items], dtype=torch.float32, device=self.device)
            for m in comparative_metrics
        }
        return feats1, feats1_lengths, feats2, feats2_lengths, targets

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        audios: List[float["t"]],
        absolute_score_items: List[Tuple[int, dict[str, float]]],
        comparative_score_items: List[Tuple[int, int, float]],
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        feats, feats_lengths = self.feature_extractor(audios)
        if self.shared_encoder is not None:
            feats = self.shared_encoder(feats, src_key_padding_mask=lengths2padding_mask(feats_lengths))
        loss, info, others = 0.0, {}, {}

        # Keep original features for use in both absolute and comparative scoring
        original_feats, original_feats_lengths = feats, feats_lengths

        if len(absolute_score_items) != 0 and self.absolute_score_loss_weight > 0:
            feats, feats_lengths, targets = self._collect_absolute_score_items(
                feats, feats_lengths, absolute_score_items
            )
            valid_metrics = [metric for metric in targets.keys() if not torch.isnan(targets[metric]).all()]
            metric2preds = self._predict_absolute_scores_from_feats(feats, feats_lengths, valid_metrics)
            loss_, info_, others_ = self._compute_absolute_score_loss(metric2preds, targets)
            others["absolute_score"] = others_
            others["absolute_score"]["metric2preds"] = metric2preds
            info.update({f"absolute_score_{key}": value for key, value in info_.items()})
            loss += loss_ * self.absolute_score_loss_weight

        if len(comparative_score_items) != 0 and self.comparative_score_loss_weight > 0:
            feats1, feats1_lengths, feats2, feats2_lengths, targets = self._collect_comparative_score_items(
                original_feats, original_feats_lengths, comparative_score_items
            )
            valid_metrics = [metric for metric in targets.keys() if not torch.isnan(targets[metric]).all()]
            metric2preds_reg, metric2preds_cls = self._predict_comparative_scores_from_feats(
                feats1, feats1_lengths, feats2, feats2_lengths, valid_metrics
            )
            loss_, info_, others_ = self._compute_comparative_score_loss(metric2preds_reg, metric2preds_cls, targets)
            others["comparative_score"] = others_
            info.update({f"comparative_score_{key}": value for key, value in info_.items()})
            loss += loss_ * self.comparative_score_loss_weight

        info["loss"] = loss.detach().cpu().item()
        return loss, info, others

    def _predict_absolute_scores_from_feats(
        self,
        feats: float["b t d"],
        feats_lengths: float["b"],
        requested_metrics: List[str] | None = None,
        return_frame_scores: bool = False,
    ) -> dict[str, torch.Tensor]:
        padding_mask = lengths2padding_mask(feats_lengths)
        metric2preds = {}
        for category, category_metrics in self.category2metrics.items():
            active_metrics = set(category_metrics)
            if requested_metrics is not None:
                active_metrics = set(category_metrics) & set(requested_metrics)
            if not active_metrics:
                continue
            category_feats = self.category2encoder[category](feats, src_key_padding_mask=padding_mask)
            for metric_name in active_metrics:
                preds, attn_and_frame_scores = self.metric2absolute_score_predictor[metric_name](
                    category_feats, feats_lengths, return_frame_scores
                )
                metric2preds[metric_name] = preds
                if attn_and_frame_scores is not None:
                    attn, preds_frame_scores = attn_and_frame_scores
                    metric2preds[f"{metric_name}_attn"] = attn
                    metric2preds[f"{metric_name}_frame_scores"] = preds_frame_scores
        return metric2preds

    def _compute_absolute_score_loss(
        self,
        metric2preds: dict[str, float["b"]],
        targets: dict[str, float["b"]],
    ) -> tuple[torch.Tensor, dict]:
        loss, info, num_category_loss = 0.0, {}, 0

        for category, metric2info in self.category2metrics.items():
            category_loss, num_metric_loss = 0.0, 0
            for metric in metric2info:
                if metric not in targets or metric not in metric2preds:
                    continue
                score_predictor = self.metric2absolute_score_predictor[metric]
                num_metric_loss += 1
                loss_, _, _ = score_predictor.compute_loss(metric2preds[metric], targets[metric])
                category_loss += loss_
                info.update({f"loss_{metric}": loss_.detach().cpu().item()})
            if num_metric_loss > 0:
                category_loss = category_loss / num_metric_loss
                loss += category_loss
                num_category_loss += 1
        loss = loss / num_category_loss
        info["loss_total"] = loss.detach().cpu().item()

        return loss, info, {}

    def _predict_comparative_scores_from_feats(
        self,
        feats1: float["b t d"],
        feats1_lengths: float["b"],
        feats2: float["b t d"],
        feats2_lengths: float["b"],
        requested_metrics: List[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        metric2preds_reg, metric2preds_cls = {}, {}

        b = feats1.shape[0]
        feats = torch.cat([feats1, feats2], dim=0)
        feats_lengths = torch.cat([feats1_lengths, feats2_lengths], dim=0)

        for category, category_metrics in self.category2metrics.items():
            # category_metrics: metric name -> config dict (min, max, optional comparative_score)
            active_metrics = {m for m in category_metrics if category_metrics[m].get("comparative_score", False)}
            if requested_metrics is not None:
                active_metrics = active_metrics & set(requested_metrics)
            if not active_metrics:
                continue
            feats_category = self.category2encoder[category](
                feats, src_key_padding_mask=lengths2padding_mask(feats_lengths)
            )
            feats1_category, feats2_category = feats_category[:b], feats_category[b:]
            feats1_category_lengths, feats2_category_lengths = feats1_lengths, feats2_lengths
            if self.use_symmetric_preference and self.training:
                feats1_category, feats2_category = torch.cat([feats1_category, feats2_category], dim=0), torch.cat(
                    [feats2_category, feats1_category], dim=0
                )
                feats1_category_lengths, feats2_category_lengths = torch.cat(
                    [feats1_category_lengths, feats2_category_lengths], dim=0
                ), torch.cat([feats2_category_lengths, feats1_category_lengths], dim=0)
            else:
                max1, max2 = feats1_lengths.max().item(), feats2_lengths.max().item()
                feats1_category, feats2_category = feats1_category[:, :max1, :], feats2_category[:, :max2, :]

            for metric in active_metrics:
                metric2preds_reg[metric], metric2preds_cls[metric] = self.metric2comparative_score_predictor[metric](
                    feats1_category, feats1_category_lengths, feats2_category, feats2_category_lengths
                )
        return metric2preds_reg, metric2preds_cls

    def _compute_comparative_score_loss(
        self,
        metric2preds_reg: dict[str, float["b"]],
        metric2preds_cls: dict[str, float["b"]],
        targets: dict[str, float["b"]],
    ) -> tuple[torch.Tensor, dict]:
        loss, info, num_category_loss = 0.0, {}, 0
        others = {
            "metric2total_items": {},
            "metric2correct_items_regression": {},
            "metric2correct_items_classification": {},
        }
        for category, metric2info in self.category2metrics.items():
            category_loss, num_metric_loss = 0.0, 0
            for metric in metric2info:
                if metric not in targets or metric not in metric2preds_reg or metric not in metric2preds_cls:
                    continue
                score_predictor = self.metric2comparative_score_predictor[metric]
                num_metric_loss += 1
                # Use local copy to avoid modifying the original targets dict
                target_metric = targets[metric]
                if self.use_symmetric_preference and self.training:
                    target_metric = torch.cat([target_metric, -target_metric], dim=0)
                preds_reg, preds_cls = metric2preds_reg[metric], metric2preds_cls[metric]
                metric_loss, metric_info, metric_others = score_predictor.compute_loss(
                    preds_reg, preds_cls, target_metric
                )
                info.update({f"{metric}_{key}": value for key, value in metric_info.items()})
                others["metric2total_items"][metric] = metric_others["total_items"]
                others["metric2correct_items_regression"][metric] = metric_others["correct_items_regression"]
                others["metric2correct_items_classification"][metric] = metric_others["correct_items_classification"]
                category_loss += metric_loss
            if num_metric_loss > 0:
                category_loss = category_loss / num_metric_loss
                loss += category_loss
                num_category_loss += 1
        loss = loss / num_category_loss
        info["loss_total"] = loss.detach().cpu().item()
        return loss, info, others

    def predict_comparative_scores_from_audio_pairs(
        self, audios1: List[float["t"]], audios2: List[float["t"]]
    ) -> tuple[dict[str, float["b"]], dict[str, float["b"]]]:
        feats1, feats1_lengths = self.feature_extractor(audios1)
        feats2, feats2_lengths = self.feature_extractor(audios2)
        if self.shared_encoder is not None:
            feats1 = self.shared_encoder(feats1, src_key_padding_mask=lengths2padding_mask(feats1_lengths))
            feats2 = self.shared_encoder(feats2, src_key_padding_mask=lengths2padding_mask(feats2_lengths))
        metric2preds_reg, metric2preds_cls = self._predict_comparative_scores_from_feats(
            feats1, feats1_lengths, feats2, feats2_lengths
        )
        return metric2preds_reg, metric2preds_cls

    def predict_absolute_scores(
        self, audios: List[float["t"]], absolute_score_items=None, return_frame_scores: bool = False, **kwargs
    ) -> dict[str, float["b"]]:
        feats, feats_lengths = self.feature_extractor(audios)
        if self.shared_encoder is not None:
            feats = self.shared_encoder(feats, src_key_padding_mask=lengths2padding_mask(feats_lengths))
        metric2preds = self._predict_absolute_scores_from_feats(
            feats, feats_lengths, return_frame_scores=return_frame_scores
        )
        return metric2preds

    def predict_comparative_scores(
        self, audios: List[float["t"]], comparative_score_items: List[Tuple[int, int, float]], **kwargs
    ) -> tuple[dict[str, float["b"]], dict[str, float["b"]]]:
        feats, feats_lengths = self.feature_extractor(audios)
        if self.shared_encoder is not None:
            feats = self.shared_encoder(feats, src_key_padding_mask=lengths2padding_mask(feats_lengths))
        feats1, feats1_lengths, feats2, feats2_lengths, targets = self._collect_comparative_score_items(
            feats, feats_lengths, comparative_score_items
        )
        metric2preds_reg, metric2preds_cls = self._predict_comparative_scores_from_feats(
            feats1, feats1_lengths, feats2, feats2_lengths
        )
        return metric2preds_reg, metric2preds_cls


def build_category_encoders(
    category2metrics: Dict[str, Dict[str, Any]],
    encoder_cfg: Union[DictConfig, Dict[str, Any]],
) -> Dict[str, torch.nn.Module]:
    return {category: instantiate(encoder_cfg) for category in category2metrics.keys()}


def build_absolute_score_predictors(
    category2metrics: Dict[str, Dict[str, Any]],
    input_dim: int,
    encoder_cfg: Union[DictConfig, Dict[str, Any]],
    pooler_cfg: Union[DictConfig, Dict[str, Any]],
    criterion: str = "mse",
) -> Dict[str, AbsoluteScorePredictor]:
    metric2predictor = {
        metric: AbsoluteScorePredictor(
            pooler=instantiate(pooler_cfg),
            encoder=instantiate(encoder_cfg),
            input_dim=input_dim,
            min_value=cfg.get("min", -math.inf),
            max_value=cfg.get("max", math.inf),
            criterion=criterion,
        )
        for metric2cfg in category2metrics.values()
        for metric, cfg in metric2cfg.items()
    }
    return metric2predictor


def build_comparative_score_predictors(
    category2metrics: Dict[str, Dict[str, Any]],
    input_dim: int,
    encoder_cfg: Union[DictConfig, Dict[str, Any]],
    pooler_cfg: Union[DictConfig, Dict[str, Any]],
    num_heads: int = 4,
    min_cmos_for_tie: float = 0.5,
    regression_weight: float = 0.5,
    classification_weight: float = 0.5,
) -> Dict[str, ComparativeScorePredictor]:
    metric2predictor = {
        metric: ComparativeScorePredictor(
            pooler=instantiate(pooler_cfg),
            encoder=instantiate(encoder_cfg),
            input_dim=input_dim,
            num_heads=num_heads,
            min_cmos_for_tie=min_cmos_for_tie,
            regression_weight=regression_weight,
            classification_weight=classification_weight,
        )
        for metric2cfg in category2metrics.values()
        for metric, cfg in metric2cfg.items()
        if cfg.get("comparative_score", False)
    }
    return metric2predictor
