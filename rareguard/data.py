import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import stft


def prepare(manifest, output, window=4096, stride=2048, representation="waveform"):
    path = Path(manifest).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"path", "key", "label", "client", "machine", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires columns: {sorted(required)}")
    if min(window, stride) < 1:
        raise ValueError("Window and stride must be positive")
    groups = {}
    seen = set()
    items = []
    for row in rows:
        if row["split"] not in ("train", "audit", "val", "test"):
            raise ValueError("Split must be train, audit, val or test")
        identity = row["machine"]
        if identity in groups and groups[identity] != (row["split"], row["client"]):
            raise ValueError(f"Machine {identity} crosses splits or clients")
        groups[identity] = (row["split"], row["client"])
        source = path.parent / row["path"]
        file_key = (str(source.resolve()), row["key"])
        if file_key in seen:
            raise ValueError(f"Duplicate signal source: {source}")
        seen.add(file_key)
        if source.suffix == ".wav":
            _, values = wavfile.read(source)
            values = values.astype(np.float32)
            if values.ndim == 2:
                values = values.mean(1)
        elif source.suffix == ".mat":
            values = loadmat(source, simplify_cells=True)
            for component in row["key"].split("/"):
                values = values[int(component)] if component.isdigit() else values[component]
        elif source.suffix == ".npy":
            values = np.load(source, allow_pickle=False)
        else:
            raise ValueError("Use WAV, MAT or NPY recordings")
        values = np.asarray(values, dtype=np.float32).squeeze()
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Expected a finite one-dimensional recording: {source}")
        for start in range(0, len(values) - window + 1, stride):
            segment = values[start:start + window]
            if representation == "spectrum":
                _, _, spectrum = stft(segment, nperseg=256, noverlap=128)
                feature = np.log1p(np.abs(spectrum)).astype(np.float32)
            else:
                feature = segment[None, :]
            items.append((feature, int(row["label"]), row["client"], row["machine"], row["split"]))
    if not items:
        raise ValueError("No complete windows in the manifest")
    x = np.stack([item[0] for item in items])
    split = np.array([item[4] for item in items])
    train = split == "train"
    if not train.any():
        raise ValueError("No training recordings")
    mean = x[train].mean(axis=(0, 2), keepdims=True)
    standard = x[train].std(axis=(0, 2), keepdims=True).clip(1e-6)
    x = (x - mean) / standard
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=x, y=np.array([item[1] for item in items], dtype=np.int64),
                        client=np.array([item[2] for item in items]),
                        machine=np.array([item[3] for item in items]), split=split,
                        mean=mean, std=standard, representation=representation)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("x", "y", "client", "machine", "split")}
    if data["x"].ndim != 3 or not np.isfinite(data["x"]).all():
        raise ValueError("Expected finite signal/features [N, channels, length]")
    if any(len(value) != len(data["x"]) for value in data.values()):
        raise ValueError("Inconsistent sample metadata")
    if set(data["split"]) != {"train", "audit", "val", "test"}:
        raise ValueError("All train/audit/val/test splits are required")
    labels = np.unique(data["y"])
    if labels.tolist() != list(range(len(labels))):
        raise ValueError("Labels must be consecutive integers starting at zero")
    for machine in np.unique(data["machine"]):
        index = data["machine"] == machine
        if len(np.unique(data["split"][index])) != 1 or len(np.unique(data["client"][index])) != 1:
            raise ValueError(f"Machine {machine} crosses splits or clients")
    data["x"] = data["x"].astype(np.float32)
    data["y"] = data["y"].astype(np.int64)
    return data
