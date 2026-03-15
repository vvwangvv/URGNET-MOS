from __future__ import annotations

import json
import logging
import math
import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


class UrgentMOS_Dataset(Dataset):
    def __init__(
        self,
        datasets: list[Path | str] | list[dict],
        category2metrics: dict[str, list[str]],
        sample_rate: int = 16000,
        pairing_scope: str = "reference",  # "reference" | "corpus" | "any" | "none",
        is_train: bool = True,
    ):
        self.entries = []
        self.sample_rate = sample_rate
        self.category2metrics = category2metrics
        self.dataset_to_reference2items = {}
        self.is_train = is_train

        self.metrics_with_comparative_score = []
        for metrics in category2metrics.values():
            for metric_name, metric_info in metrics.items():
                if metric_info.get("comparative_score", False):
                    self.metrics_with_comparative_score.append(metric_name)

        if isinstance(datasets[0], dict):
            self.entries = datasets[:]
            for item in self.entries:
                item["dataset"] = "default"
        else:
            for dataset in datasets:
                dataset_hash = hash(Path(dataset).resolve())
                with open(dataset, "r") as f:
                    for line in tqdm(f, desc=f"Loading {dataset}"):
                        item = json.loads(line)
                        item["dataset"] = dataset_hash
                        if "durations" not in item and "duration" not in item:
                            raise ValueError(f"Item {item} has no durations or duration")
                        self.entries.append(item)
        if is_train and pairing_scope != "none":
            for item in tqdm(self.entries, desc="Indexing reference pairs..."):
                if not self.item_has_comparative_score(item):
                    continue
                dataset, reference_id = item["dataset"], item["reference_id"]
                if dataset not in self.dataset_to_reference2items:
                    self.dataset_to_reference2items[dataset] = {}
                if reference_id not in self.dataset_to_reference2items[dataset]:
                    self.dataset_to_reference2items[dataset][reference_id] = []
                self.dataset_to_reference2items[dataset][reference_id].append(item)

        self.pairing_scope = pairing_scope

    def __len__(self):
        return len(self.entries)

    def process_item(self, item):
        """Load or use in-memory audio; return dict with 'audio' (1D float tensor), uid, etc.

        In-memory tensor is assumed pre-normalized (infer API); ndarray is converted and
        resampled if item has sample_rate. Path-loaded audio is loaded and resampled as needed.
        """
        sr = None
        if "audio" in item:
            raw = item["audio"]
            if isinstance(raw, torch.Tensor):
                audio = raw.float()
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)
            else:
                audio = torch.from_numpy(np.asarray(raw, dtype=np.float32))
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)
                sr = item.get("sample_rate", self.sample_rate)
        else:
            try:
                audio, sr = torchaudio.load(item["audio_path"])
                audio = audio.mean(dim=0)
            except Exception as e:
                logging.warning(f"Error loading audio {item['audio_path']}: {e}")
                return None

        if sr is not None and sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        out = {
            "audio": audio.float(),
            "audio_path": item.get("audio_path"),
            "reference_id": item.get("reference_id", None),
            "system_id": item.get("system_id", None),
            "uid": item.get("uid", None),
        }
        if "metrics" in item:
            out["metrics"] = {
                name: item.get("metrics", {}).get(name, float("nan"))
                for name in chain.from_iterable(self.category2metrics.values())
            }
        return out

    def __getitem__(self, idx):
        item = self.entries[idx]
        item1 = item2 = comparative_metrics = None
        if "audio_paths" in item:
            item1, item2, comparative_metrics = (
                {"audio_path": item["audio_paths"][0], "uid": item.get("uids", [None, None])[0]},
                {"audio_path": item["audio_paths"][1], "uid": item.get("uids", [None, None])[1]},
                item.get("metrics", {}),
            )
        elif "audios" in item:
            # In-memory pair (e.g. from infer API): pre-normalized tensors, no sample_rate
            uids = item.get("uids", [idx, idx])
            item1 = {"audio": item["audios"][0], "uid": uids[0]}
            item2 = {"audio": item["audios"][1], "uid": uids[1]}
            comparative_metrics = item.get("metrics", {})
        else:
            item1, item2 = item, self.find_pair(item)
            if item2 is not None:
                comparative_metrics = {
                    name: item1["metrics"].get(name, float("nan")) - item2["metrics"].get(name, float("nan"))
                    for name in self.metrics_with_comparative_score
                }
                if all(math.isnan(v) for v in comparative_metrics.values()):
                    comparative_metrics = None

        item1_dict = self.process_item(item1)
        item2_dict = self.process_item(item2) if item2 is not None else None

        return item1_dict, item2_dict, comparative_metrics

    def item_has_comparative_score(self, item):
        if "durations" in item:
            # NOTE: "durations" indicate the item is a pair
            return False
        return any(
            not math.isnan(item.get("metrics", {}).get(metric, float("nan")))
            for metric in self.metrics_with_comparative_score
        )

    def find_pair(self, item):
        if not self.is_train or self.pairing_scope == "none" or not self.item_has_comparative_score(item):
            return None
        if self.pairing_scope == "any":
            candidate_items = [entry for entry in self.entries if self.item_has_comparative_score(entry)]
        elif self.pairing_scope == "corpus":
            candidate_items = list(chain.from_iterable(self.dataset_to_reference2items[item["dataset"]].values()))
        elif self.pairing_scope == "reference":
            candidate_items = list(self.dataset_to_reference2items[item["dataset"]][item["reference_id"]])
        candidate_items = candidate_items[:]
        random.shuffle(candidate_items)
        try:
            pair = next(ci for ci in candidate_items if item["duration"] >= ci["duration"])  # prevent OOM
            return pair
        except StopIteration:
            return None

    def get_frame_len(self, idx):
        item = self.entries[idx]
        if "duration" in item:
            return int(item["duration"] * self.sample_rate)
        if "durations" in item:
            return int(max(item["durations"]) * self.sample_rate)
        if "audio" in item:
            return _audio_numel(item["audio"])
        if "audios" in item:
            return max(_audio_numel(a) for a in item["audios"])
        raise ValueError(f"Entry has no duration/audio/audios: {list(item.keys())}")


def _audio_numel(a) -> int:
    """Number of samples in audio (tensor or array)."""
    return a.numel() if isinstance(a, torch.Tensor) else len(a)


def collate_fn(batch):
    audio_list, absolute_score_items, comparative_score_items = [], [], []
    uids, system_ids = [], []
    for item1, item2, comparative_metrics in batch:
        if item1 is None:
            continue
        audio_list.append(item1["audio"])
        idx = len(audio_list) - 1
        uids.append(item1.get("uid", None))
        system_ids.append(item1.get("system_id", None))
        if "metrics" in item1:
            absolute_score_items.append((idx, item1["metrics"]))

        if item2 is not None:
            audio_list.append(item2["audio"])
            system_ids.append(item2.get("system_id", None))
            uids.append(item2.get("uid", None))
            if "metrics" in item2:
                absolute_score_items.append((idx + 1, item2["metrics"]))
            comparative_score_items.append((idx, idx + 1, comparative_metrics))

    batch_dict = {
        "audios": audio_list,
        "uids": uids,
        "system_ids": system_ids,
        "comparative_score_items": comparative_score_items,
        "absolute_score_items": absolute_score_items,
    }
    return batch_dict


# https://github.com/SWivid/F5-TTS/blob/605fa13b42b40e860961bac8ce30fe49f02dfa0d/src/f5_tts/model/dataset.py#L165
class DynamicBatchSampler(Sampler):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, dataset: Dataset, frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0
        self.dataset = dataset

        indices, batches = [], []
        logging.info("Sorting dataset by frame lengths... This can be slow if duration was not precomputed")
        for idx in tqdm(range(len(dataset)), desc="Sorting dataset... "):
            indices.append((idx, dataset.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    logging.warning(
                        f"Single sample with {frame_len} frames exceeds the frames_threshold of {self.frames_threshold}, dropping it."
                    )
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


def _data_has_in_memory_audio(data: list) -> bool:
    """True if data is list of dicts with in-memory audio (no path loading)."""
    if not data or not isinstance(data[0], dict):
        return False
    first = data[0]
    return "audio" in first or "audios" in first


def build_dataloader(
    data: list[Path] | list[dict],
    category2metrics: dict[str, dict[str, float]],
    sample_rate: int = 16000,
    pairing_scope: str = "corpus",
    batch_frame_per_gpu: int = 480000,
    max_samples_per_gpu: int = 32,
    num_workers: int = 4,
    prefetch: int = 100,
    seed: int = 42,
    is_train: bool = True,
) -> DataLoader:
    if _data_has_in_memory_audio(data):
        num_workers = 0
        prefetch = None

    dataset = UrgentMOS_Dataset(
        data,
        category2metrics,
        sample_rate,
        pairing_scope=pairing_scope,
        is_train=is_train,
    )
    batch_sampler = DynamicBatchSampler(
        dataset,
        batch_frame_per_gpu,
        max_samples=max_samples_per_gpu,
        random_seed=seed if is_train else None,
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        pin_memory=num_workers != 0,
        persistent_workers=num_workers != 0,
        batch_sampler=batch_sampler,
    )
    return dataloader
