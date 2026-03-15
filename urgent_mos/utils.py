from __future__ import annotations

from typing import Optional, Union
from pathlib import Path
from omegaconf import OmegaConf

from hydra.utils import instantiate
import torch
from huggingface_hub import hf_hub_download
from torchcodec.decoders import AudioDecoder


def lengths2padding_mask(lengths: int["b"], max_len: Optional[int] = None) -> bool["b t"]:
    """
    Make padding mask from lengths.
    True = pad
    """
    B = lengths.size(0)
    T = int(max_len if max_len is not None else lengths.max().item())
    idx = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
    return idx >= lengths.unsqueeze(1)


def mask2lens(mask: int["b n"]) -> int["b"]:
    return mask.sum(dim=1)


def default(value, default_value):
    return value if value is not None else default_value


def exists(value):
    return value is not None


def override(d: dict, **kwargs):
    """
    Override dictionary d with keyword arguments (includig None).
    """
    for k, v in kwargs.items():
        d[k] = v
    return d


def calculate_metrics(preds: list[dict], refs: list[dict]) -> dict[str, dict[str, float]]:
    """
    preds: list of {"uid": str, "system_id": str, "value": float}
    lables: list of {"uid": str, "system_id": str, "value": float}
    """
    import numpy as np
    import pandas as pd
    import scipy.stats

    df_pred = pd.DataFrame(preds).rename(columns={"value": "pred"})
    df_ref = pd.DataFrame(refs).rename(columns={"value": "ref"})

    df_pred = df_pred.merge(df_ref, on=["uid", "system_id"], how="left")
    if df_pred["ref"].isna().any():
        missing_rows = df_pred[df_pred["ref"].isna()]
        raise ValueError(f"Missing refs for some predictions:\n{missing_rows}")

    utt_pred = df_pred.sort_values(by=["uid"])["pred"].to_numpy(dtype=float)
    utt_ref = df_pred.sort_values(by=["uid"])["ref"].to_numpy(dtype=float)

    sys_df = df_pred.groupby("system_id", as_index=False).agg(pred=("pred", "mean"), ref=("ref", "mean"))

    sys_pred = sys_df.sort_values(by=["system_id"])["pred"].to_numpy(dtype=float)
    sys_ref = sys_df.sort_values(by=["system_id"])["ref"].to_numpy(dtype=float)

    def metrics(preds: float["b"], refs: float["b"]) -> dict[str, float]:
        mse = np.mean((preds - refs) ** 2)
        lcc = np.corrcoef(preds, refs)[0, 1]
        srcc = scipy.stats.spearmanr(preds, refs).statistic
        ktau = scipy.stats.kendalltau(preds, refs).statistic
        return {"mse": mse, "lcc": lcc, "srcc": srcc, "ktau": ktau}

    return {"utt": metrics(utt_pred, utt_ref), "sys": metrics(sys_pred, sys_ref)}


def load_model_from_checkpoint(
    checkpoint_path: Union[Path, str],
    device: Union[str, torch.device] = "cuda",
) -> torch.nn.Module:

    # download if checkpoint is a huggingface hub repo id
    if not Path(checkpoint_path).is_file():
        checkpoint_path = hf_hub_download(checkpoint_path, "model.pt")

    checkpoint_path = Path(checkpoint_path)

    if Path(checkpoint_path).is_dir():
        checkpoint_path = Path(checkpoint_path) / "model.pt"

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    config = OmegaConf.create(checkpoint["config"])
    model = instantiate(config.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).float()

    setattr(model, "config", config)

    return model


def get_audio_duration(audio_path):
    audio_path = audio_path if isinstance(audio_path, str) else audio_path.as_posix()
    decoder = AudioDecoder(audio_path)
    meta = decoder.metadata
    if meta.duration_seconds_from_header is not None:
        duration = meta.duration_seconds_from_header
    else:
        samples = decoder.get_all_samples()
        duration = samples.duration_seconds
    return duration
