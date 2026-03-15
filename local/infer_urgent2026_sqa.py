#!/usr/bin/env python3
"""
Run URGNET-MOS inference on the urgent-challenge/urgent2026-sqa HuggingFace dataset.

Produces a submission-ready predictions.scp (space-delimited, headerless):
  sample_id score

ACR: absolute score (MOS 1--5). CCR: comparative score (CMOS -3 to +3).

Usage (from repo root):
  python local/infer_urgent2026_sqa.py --checkpoint path/to/model.pt --output predictions.scp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from urgent_mos.api.infer import infer, infer_pairs
from urgent_mos.utils import load_model_from_checkpoint

DATASET_ID = "urgent-challenge/urgent2026-sqa"


def _get_audio_tensor(row, key: str):
    """Extract 1D float waveform and sample rate from a dataset row (torchcodec or dict)."""
    audio_col = row[key]
    if hasattr(audio_col, "get_all_samples"):
        # torchcodec-style: AudioDecoder
        samples = audio_col.get_all_samples()
        waveform = samples.data.squeeze(0).float()
        sr = getattr(samples, "sample_rate", row.get("sample_rate", 16000))
    elif isinstance(audio_col, dict):
        # dict with array + sampling_rate
        waveform = torch.from_numpy(audio_col["array"].astype("float32"))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        sr = audio_col.get("sampling_rate", row.get("sample_rate", 16000))
    else:
        raise TypeError(f"Unexpected audio type for key {key!r}: {type(audio_col)}")
    return waveform, sr


METRIC = "mos_overall"


def run_acr(model, dataset, batch_frames: int | None):
    """Run absolute (ACR) inference; return list of (sample_id, score)."""
    sample_ids = []
    audios = []
    sample_rates = []
    for i in tqdm(range(len(dataset)), desc="Loading ACR"):
        row = dataset[i]
        sample_ids.append(row["sample_id"])
        wav, sr = _get_audio_tensor(row, "audio")
        audios.append(wav)
        sample_rates.append(sr)
    results = infer(
        model,
        audios,
        sample_rate=sample_rates,
        batch_frames=batch_frames,
        num_workers=0,
    )
    return [(sid, results[i][METRIC]) for i, sid in enumerate(sample_ids)]


def run_ccr(model, dataset, batch_frames: int | None):
    """Run comparative (CCR) inference; return list of (sample_id, score)."""
    sample_ids = []
    audios_a = []
    audios_b = []
    sample_rates = []
    for i in tqdm(range(len(dataset)), desc="Loading CCR"):
        row = dataset[i]
        sample_ids.append(row["sample_id"])
        wa, sra = _get_audio_tensor(row, "audio_a")
        wb, srb = _get_audio_tensor(row, "audio_b")
        audios_a.append(wa)
        audios_b.append(wb)
        sample_rates.append((sra, srb))
    pairs = list(zip(audios_a, audios_b))
    results = infer_pairs(
        model,
        pairs,
        sample_rate=sample_rates,
        batch_frames=batch_frames,
        num_workers=0,
    )
    return [(sid, results[i][METRIC]) for i, sid in enumerate(sample_ids)]


def main():
    parser = argparse.ArgumentParser(
        description="Run URGNET-MOS on urgent2026-sqa (ACR + CCR) and write predictions.scp"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="urgent-challenge/urgent-mos-f1c1m5dref",
        help="Path to model checkpoint (e.g. model.pt or HuggingFace repo id)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.scp",
        help="Output path for space-delimited predictions (default: predictions.scp)",
    )
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=None,
        help="Max audio frames per batch (default: from config). Lower to avoid OOM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.eval()

    def to_scalar(score):
        if isinstance(score, (int, float)):
            return float(score)
        if hasattr(score, "item"):
            return float(score.item())
        return float(score[0])

    lines = []
    acr_ds = load_dataset(DATASET_ID, "acr", split="test")
    for sid, score in run_acr(model, acr_ds, args.batch_frames):
        lines.append(f"{sid} {to_scalar(score)}")
    ccr_ds = load_dataset(DATASET_ID, "ccr", split="test")
    for sid, score in run_ccr(model, ccr_ds, args.batch_frames):
        lines.append(f"{sid} {to_scalar(score)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} predictions to {out_path}")


if __name__ == "__main__":
    main()
