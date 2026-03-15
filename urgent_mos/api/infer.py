from __future__ import annotations

from pathlib import Path
import logging
from typing import List, Tuple, Union

import numpy as np
import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import instantiate
from tqdm import tqdm

from urgent_mos.utils import load_model_from_checkpoint, get_audio_duration

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

AudioInput = Union[str, Path, np.ndarray, torch.Tensor]
AudioPairInput = Tuple[
    Union[str, Path, np.ndarray, torch.Tensor],
    Union[str, Path, np.ndarray, torch.Tensor],
]


def _to_serializable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def _is_path(x) -> bool:
    return isinstance(x, (str, Path))


def _normalize_audio(x: AudioInput, sample_rate: int | None = None) -> torch.Tensor:
    """Convert path, ndarray, or tensor to 1D float tensor at TARGET_SAMPLE_RATE."""
    if _is_path(x):
        audio, sample_rate = torchaudio.load(str(x))
        audio = audio.mean(dim=0).float()
    elif isinstance(x, np.ndarray):
        audio = torch.from_numpy(np.asarray(x, dtype=np.float32))
    elif isinstance(x, torch.Tensor):
        audio = x
    else:
        raise TypeError(f"Unsupported audio type: {type(x)}. Use path, numpy array, or torch tensor.")
    audio = audio.float()
    if audio.dim() > 1:
        audio = audio.mean(dim=0)
    sample_rate = TARGET_SAMPLE_RATE if sample_rate is None else sample_rate

    if sample_rate != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, TARGET_SAMPLE_RATE)
    return audio


# ── single entry builders (dataset-compatible) ─────────────────────────────────
def _audio_inputs_to_entries(
    audio_inputs: List[AudioInput],
    sample_rate: int | List[int] | None,
) -> List[dict]:
    """Build dataset entries: path → {audio_path, uid, duration}; array/tensor → {audio, uid}."""

    sample_rate_list = sample_rate
    if sample_rate_list is None:
        sample_rate_list = [TARGET_SAMPLE_RATE] * len(audio_inputs)
    elif isinstance(sample_rate_list, int):
        sample_rate_list = [sample_rate_list] * len(audio_inputs)

    if len(sample_rate_list) != len(audio_inputs):
        raise ValueError(f"sample_rate length != inputs length: {len(sample_rate_list)} != {len(audio_inputs)}")

    entries = []
    for idx, x in enumerate(audio_inputs):
        if _is_path(x):
            entries.append(
                {
                    "audio_path": str(x),
                    "uid": idx,
                    "duration": get_audio_duration(x),
                }
            )
        else:
            entries.append(
                {
                    "audio": _normalize_audio(x, sample_rate_list[idx]),
                    "uid": idx,
                }
            )
    return entries


def _pair_inputs_to_entries(
    audio_pair_inputs: List[AudioPairInput],
    sample_rate: int | List[Tuple[int, int]] | None,
) -> List[dict]:
    """Build dataset entries: path pairs → {audio_paths, uids, durations}; else → {audios, uids}."""
    n = len(audio_pair_inputs)
    if sample_rate is None:
        srs = [(TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE)] * n
    elif isinstance(sample_rate, int):
        srs = [(sample_rate, sample_rate)] * n
    else:
        if len(sample_rate) != n:
            raise ValueError(f"sample_rate length ({len(sample_rate)}) must match number of pairs ({n})")
        srs = list(sample_rate)

    entries = []
    for idx, (a, b) in enumerate(audio_pair_inputs):
        if _is_path(a) and _is_path(b):
            entries.append(
                {
                    "audio_paths": (str(a), str(b)),
                    "uids": [idx, idx],
                    "durations": (get_audio_duration(a), get_audio_duration(b)),
                }
            )
        else:
            assert not _is_path(a) and not _is_path(b), "Audio pair inputs must both be paths or in-memory tensors"
            entries.append(
                {
                    "audios": (_normalize_audio(a, srs[idx][0]), _normalize_audio(b, srs[idx][1])),
                    "uids": [idx, idx],
                }
            )
    return entries


def _dataloader_overrides(
    batch_frames: int | None = None,
    num_workers: int | None = None,
) -> dict:
    """Build kwargs to override model.config.dataloader (for OOM control and acceleration)."""
    overrides = {}
    if batch_frames is not None:
        overrides["batch_frame_per_gpu"] = batch_frames
    if num_workers is not None:
        overrides["num_workers"] = num_workers
    return overrides


def _run_absolute_dataloader(
    model,
    entries: List[dict],
    return_frame_scores: bool = False,
    batch_frames: int | None = None,
    num_workers: int | None = None,
) -> List[dict]:
    accelerator = Accelerator()
    kwargs = {"data": entries, "is_train": False, **_dataloader_overrides(batch_frames, num_workers)}
    dataloader = instantiate(model.config.dataloader, **kwargs, _convert_="all")
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    uid2result: dict[int, dict] = {}
    for batch in tqdm(dataloader):
        metric2preds = model.predict_absolute_scores(batch["audios"], return_frame_scores=return_frame_scores)
        for i, uid in enumerate(batch["uids"]):
            uid2result[uid] = {k: _to_serializable(v[i]) for k, v in metric2preds.items()}
    return [uid2result[i] for i in range(len(entries))]


def _run_pairs_dataloader(
    model,
    entries: List[dict],
    batch_frames: int | None = None,
    num_workers: int | None = None,
) -> List[dict]:
    accelerator = Accelerator()
    kwargs = {"data": entries, "is_train": False, **_dataloader_overrides(batch_frames, num_workers)}
    dataloader = instantiate(model.config.dataloader, **kwargs, _convert_="all")
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    uid2result: dict[int, dict] = {}
    for batch in tqdm(dataloader):
        metric2preds_reg, _ = model.predict_comparative_scores(**batch)
        for i, uid in enumerate(batch["uids"][::2]):
            uid2result[uid] = {k: _to_serializable(v[i]) for k, v in metric2preds_reg.items()}
    return [uid2result[i] for i in range(len(entries))]


# ── public API ─────────────────────────────────────────────────────────────────
@torch.inference_mode()
def infer(
    model,
    audio_inputs: List[AudioInput],
    sample_rate: int | List[int] | None = None,
    batch_frames: int | None = None,
    num_workers: int = 4,
    return_frame_scores: bool = False,
) -> List[dict]:
    """Run absolute score (MOS) inference on a list of audio inputs.

    Each input may be a path (str or Path), a numpy array, or a torch tensor (1D waveform).
    Paths and arrays/tensors use the same dynamic dataloader; in-memory inputs
    use num_workers=0 internally.

    Args:
        model: Loaded UrgentMOS model (e.g. from load_model_from_checkpoint).
        audio_inputs: List of paths, numpy arrays, or tensors; one item per audio.
        sample_rate: Sample rate for array/tensor inputs only. Single int for all,
            or list of ints per item. Ignored for path inputs. Default 16000.
        batch_frames: Max audio frames per batch (dynamic batching). Lower to avoid
            CUDA OOM. If None, uses model.config.dataloader.batch_frame_per_gpu.
        num_workers: Dataloader workers for path-based loading. Ignored for
            in-memory inputs. If None, uses config value.
        return_frame_scores: If True, include per-frame scores in the output dicts.

    Returns:
        List of dicts, one per input. Each dict maps metric name to score (float or
        list of floats when return_frame_scores is True).
    """
    if not audio_inputs:
        return []
    entries = _audio_inputs_to_entries(audio_inputs, sample_rate)
    return _run_absolute_dataloader(
        model,
        entries,
        return_frame_scores=return_frame_scores,
        batch_frames=batch_frames,
        num_workers=num_workers,
    )


@torch.inference_mode()
def infer_pairs(
    model,
    audio_pair_inputs: List[AudioPairInput],
    sample_rate: int | List[Tuple[int, int]] | None = None,
    batch_frames: int | None = None,
    num_workers: int = 0,
) -> List[dict]:
    """Run comparative (pair) inference on (reference, system) audio pairs.

    Each pair is (audio_a, audio_b); each element may be a path (str or Path),
    a numpy array, or a torch tensor (1D waveform). Uses the same dynamic
    dataloader as infer for efficient batching.

    Args:
        model: Loaded UrgentMOS model (e.g. from load_model_from_checkpoint).
        audio_pair_inputs: List of (audio_a, audio_b) pairs.
        sample_rate: Sample rate for array/tensor elements only. Single int for
            all, or list of (sr_a, sr_b) per pair. Ignored for path inputs.
            Default 16000.
        batch_frames: Max audio frames per batch. Lower to avoid CUDA OOM.
            If None, uses model.config.dataloader.batch_frame_per_gpu.
        num_workers: Dataloader workers for path-based loading. Ignored for
            in-memory inputs. If None, uses config value.

    Returns:
        List of dicts, one per pair. Each dict maps metric name to predicted
        comparative score (e.g. CMOS).
    """
    if not audio_pair_inputs:
        return []
    entries = _pair_inputs_to_entries(audio_pair_inputs, sample_rate)
    return _run_pairs_dataloader(
        model,
        entries,
        batch_frames=batch_frames,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import json

    audio_paths = []
    with open("data/bc19/test/data.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            audio_paths.append(item["audio_path"])
    checkpoint = "exp/f1s1c5m1_C1M1_pool_metric_token_noug/model_11000.pt"
    model = load_model_from_checkpoint(checkpoint, "cuda")
    result = infer(model, audio_paths, return_frame_scores=True)
    print(result)

    audio_pair_paths = []
    with open("data/chime-7-udase-eval/test/data_pairs.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            audio_pair_paths.append(item["audio_paths"])
    checkpoint = "exp/f1c1m5_d_ref/model_11000.pt"
    model = load_model_from_checkpoint(checkpoint, "cuda")
    result = infer_pairs(model, audio_pair_paths)
    print(result)
