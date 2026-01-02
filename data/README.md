# Data Directory

> **DO NOT COMMIT DATA FILES TO THIS REPOSITORY**

This directory is for local dataset references only. All data files are `.gitignore`d.

## Expected Structure for LocalNpzAdapter

The `LocalNpzAdapter` expects a directory with one `.npz` file per subject:

```
data/
  README.md           # This file
  raw/                # Raw dataset directory
    subject001.npz
    subject002.npz
    subject003.npz
    ...
```

### NPZ File Contents

Each `.npz` file **MUST** contain:

| Field | Type | Description |
|-------|------|-------------|
| `ppg` | `np.ndarray` | PPG signal array, shape `(N,)` or `(1, N)` |
| `bioz` | `np.ndarray` | Bioimpedance signal array, shape `(N,)` or `(1, N)` |
| `fs_ppg` | `float` | PPG sampling rate in Hz |
| `fs_bioz` | `float` | Bioimpedance sampling rate in Hz |

Optional fields:

| Field | Type | Description |
|-------|------|-------------|
| `t` | `np.ndarray` | Timestamps array |
| `step` | `np.ndarray` | Protocol step/level array (aligned in time) |
| `t_step` | `np.ndarray` | Timestamps aligned with `step` (if different from `t`) |
| `metadata` | `dict` | Additional metadata dictionary |

Note: multi-channel signals are not supported in v0.x. Select a single channel
or average before saving.

### Step Alignment Contract

- If `step` is provided, it must be aligned with `t` (same length), or provide
  its own `t_step` (same length as `step`).
- If no timestamps are provided, `step` length must match the reference signal
  length after resampling, otherwise preprocessing will raise an error.

### Example: Creating a Valid Dataset

```python
import numpy as np

# Create sample data
fs_ppg = 100.0  # 100 Hz
fs_bioz = 50.0  # 50 Hz
duration = 60   # 60 seconds

ppg = np.sin(2 * np.pi * 1.2 * np.arange(int(fs_ppg * duration)) / fs_ppg)
bioz = np.sin(2 * np.pi * 0.3 * np.arange(int(fs_bioz * duration)) / fs_bioz)
t = np.arange(int(fs_ppg * duration)) / fs_ppg
step = np.repeat([1, 2, 3, 4], int(fs_ppg * duration / 4))

# Save as .npz
np.savez(
    "data/raw/subject001.npz",
    ppg=ppg,
    bioz=bioz,
    fs_ppg=fs_ppg,
    fs_bioz=fs_bioz,
    t=t,
    step=step,
    metadata={"age": 25, "gender": "M"},
)
```

## Validating Your Dataset

Use the `opencr data fetch` command to validate your dataset:

```bash
opencr data fetch data/raw --output runs/fetch --adapter npz
```

This will:
1. Validate the directory structure
2. Check that all required fields are present
3. Generate a `data_card.json` with dataset metadata

## Anti-Leakage Policy

> **WARNING**: All train/val/test splits MUST be performed at the **SUBJECT level**.

This prevents data leakage where samples from the same subject appear in both training and test sets. The preprocessing pipeline enforces this automatically.
