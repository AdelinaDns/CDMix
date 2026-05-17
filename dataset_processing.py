"""
LPIS PixelSet Pipeline — Full End-to-End
=========================================
GEE asset → Cloud Score+ composites → GCS export → local assembly
→ temporal interpolation → PyTorch Dataset

Single config block at the top. Run:
    python pipeline.py

Stages (each is skipped if output already exists — safe to resume):
    1. GEE export   : sampleRegions() per composite per batch → GCS CSVs
    2. GCS download : pull CSVs to local staging folder
    3. Assembly     : CSVs → (T, C, S) uint16 .npy + META/
    4. Interpolation: fill fully-cloudy date slices via linear interp
    5. Dataset      : LPISPixelSetDataset + CombinedLPISDataset + DataLoader

Requirements:
    pip install earthengine-api google-cloud-storage geemap tqdm pandas numpy torch
    pip install cupy-cuda12x   # optional GPU acceleration (match your CUDA version)
"""

import ee
import os
import json
import time
import logging
import warnings
import numpy as np
import pandas as pd
import torch
from datetime import date as _date
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    log.info("CuPy found — GPU acceleration enabled.")
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not found — using CPU NumPy. "
                  "Install: pip install cupy-cuda12x")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← edit this block only
# ══════════════════════════════════════════════════════════════════════════════

YEAR             = 2024
GEE_PROJECT      = "disertatie-494713"
GCS_BUCKET       = "disertatie-494713-lpis"
GCS_PREFIX       = "slovakia_section1"
OUTPUT_PATH      = "./PixelSet-Slovakia-S1"
LPIS_ASSET       = "projects/disertatie-494713/assets/Section_1_Slovakia"
PARCEL_ID_PROP   = "ID_PARCEL"
LABEL_NAMES      = ["CODE_GROUP"]

BATCH_SIZE       = 500    # polygons per GEE export task
TILE_SCALE       = 8      # increase to 16 if GEE raises memory errors
CS_THRESHOLD     = 0.60   # Cloud Score+ cs_cdf threshold
MAX_CONCURRENT   = 20     # simultaneous GEE export tasks
POLL_INTERVAL    = 30     # seconds between task status polls
N_CSV_WORKERS    = min(os.cpu_count(), 16)
N_INTERP_WORKERS = min(os.cpu_count(), 16)

# Dataset filtering & sampling
MAX_PIXELS       = 500    # parcels with S > MAX_PIXELS are randomly subsampled
                           # keeps large parcels but limits GPU memory per batch
                           # set to None to disable subsampling
MIN_PIXELS       = 3      # parcels with fewer pixels are excluded

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# ══════════════════════════════════════════════════════════════════════════════
#  DATES  — year-agnostic bimonthly grid
# ══════════════════════════════════════════════════════════════════════════════

def build_bimonthly_dates(year: int) -> dict:
    """
    24 composite target dates: 1st and 15th of every month.
    Indices 0-23 consistent across years — only the year changes.
    """
    dates = {}
    idx = 0
    for month in range(1, 13):
        for day in [1, 15]:
            dates[idx] = int(_date(year, month, day).strftime("%Y%m%d"))
            idx += 1
    return dates


COMPOSITE_DATES = build_bimonthly_dates(YEAR)
N_DATES         = len(COMPOSITE_DATES)   # always 24
N_BANDS         = len(S2_BANDS)          # always 10
DATE_TO_IDX     = {v: k for k, v in COMPOSITE_DATES.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — GEE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _yyyymmdd_to_ee(d: int) -> ee.Date:
    s = str(d)
    return ee.Date(f"{s[:4]}-{s[4:6]}-{s[6:8]}")


def _build_date_windows() -> list:
    """
    Compute [start, end) window for each composite date as midpoints between
    adjacent target dates. Edge dates extend ±7 days outward.
    """
    def to_days(d: int) -> int:
        s = str(d)
        return (_date(int(s[:4]), int(s[4:6]), int(s[6:8])) -
                _date(2000, 1, 1)).days

    items    = sorted(COMPOSITE_DATES.items())
    day_nums = [to_days(v) for _, v in items]
    windows  = []

    for i, (idx, _) in enumerate(items):
        half_before = 7 if i == 0 else (day_nums[i] - day_nums[i - 1]) // 2
        half_after  = 7 if i == len(items) - 1 else (day_nums[i + 1] - day_nums[i]) // 2
        center = _yyyymmdd_to_ee(COMPOSITE_DATES[idx])
        windows.append((idx,
                        center.advance(-half_before, "day"),
                        center.advance(half_after,   "day")))
    return windows


def _build_composite(start: ee.Date, end: ee.Date, aoi: ee.Geometry) -> ee.Image:
    """
    Cloud-masked median composite using linkCollection — the correct GEE
    pattern for joining Cloud Score+ to S2. Avoids the system:index join
    silently failing inside .map() on the server side.
    """
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
          .linkCollection(cs_plus, ["cs_cdf"]))

    def mask_and_select(image):
        return (image
                .updateMask(image.select("cs_cdf").gte(CS_THRESHOLD))
                .select(S2_BANDS)
                .copyProperties(image, ["system:time_start"]))

    masked = s2.map(mask_and_select)

    # Zero fallback for windows with no valid scenes (e.g. winter cloud cover)
    # These all-zero pixels are caught and filled by interpolation in Stage 4
    empty = ee.Image.constant([0] * len(S2_BANDS)).rename(S2_BANDS).toUint16()

    composite = ee.Algorithms.If(
        masked.size().gt(0),
        masked.median().select(S2_BANDS).unmask(0).toUint16(),
        empty
    )
    return ee.Image(composite)


def _batch_fc(fc: ee.FeatureCollection) -> list:
    """
    Split FC into batches using toList(count, offset).
    Deterministic and guaranteed complete — every parcel appears exactly once.
    Fixes the non-deterministic toList() issue that caused ~2650 missing parcels.
    """
    total = fc.size().getInfo()
    log.info("Asset: %d parcels → %d batches of %d",
             total, -(-total // BATCH_SIZE), BATCH_SIZE)

    batches = []
    for start in range(0, total, BATCH_SIZE):
        batch = ee.FeatureCollection(fc.toList(BATCH_SIZE, start))
        batches.append(batch)
    return batches


def _make_task(composite: ee.Image, fc_batch: ee.FeatureCollection,
               date_int: int, composite_idx: int, batch_idx: int) -> ee.batch.Task:
    samples = composite.sampleRegions(
        collection=fc_batch,
        properties=[PARCEL_ID_PROP] + LABEL_NAMES,
        scale=10,
        tileScale=TILE_SCALE,
        geometries=False,
    ).map(lambda f: f.set("composite_date", date_int))

    return ee.batch.Export.table.toCloudStorage(
        collection=samples,
        description=f"lpis_d{composite_idx:02d}_b{batch_idx:04d}",
        bucket=GCS_BUCKET,
        fileNamePrefix=f"{GCS_PREFIX}/date_{composite_idx:02d}/batch_{batch_idx:04d}",
        fileFormat="CSV",
        selectors=[PARCEL_ID_PROP, "composite_date"] + LABEL_NAMES + S2_BANDS,
    )


def _wait_for_tasks(tasks: list):
    pending = {t.id: t for t in tasks}
    while pending:
        time.sleep(POLL_INTERVAL)
        still = {}
        for tid, task in list(pending.items()):
            status = task.status()
            state  = status["state"]
            if state == "COMPLETED":
                log.info("  ✓ %s", tid[:20])
            elif state in ("FAILED", "CANCELLED"):
                log.error("  ✗ %s  %s", tid[:20],
                          status.get("error_message", ""))
            else:
                still[tid] = task
        pending = still
        if pending:
            log.info("  %d tasks still running...", len(pending))


def run_gee_export():
    """Stage 1: submit all GEE export tasks and wait for completion."""
    staging   = Path(OUTPUT_PATH) / "_csv_staging"
    done_flag = staging / ".gee_export_done"
    if done_flag.exists():
        log.info("Stage 1 skipped — GEE export already complete.")
        return

    log.info("=== Stage 1: GEE Export ===")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

    lpis_fc = ee.FeatureCollection(LPIS_ASSET)
    aoi     = lpis_fc.geometry().bounds()
    batches = _batch_fc(lpis_fc)
    windows = _build_date_windows()

    all_tasks = []
    for composite_idx, (window_idx, start, end) in enumerate(windows):
        date_int  = COMPOSITE_DATES[window_idx]
        composite = _build_composite(start, end, aoi)
        for batch_idx, batch_fc in enumerate(batches):
            all_tasks.append(_make_task(composite, batch_fc, date_int,
                                        composite_idx, batch_idx))

    log.info("Submitting %d tasks in waves of %d...",
             len(all_tasks), MAX_CONCURRENT)

    for i in range(0, len(all_tasks), MAX_CONCURRENT):
        wave = all_tasks[i:i + MAX_CONCURRENT]
        for t in wave:
            t.start()
            time.sleep(0.3)
        log.info("Wave %d/%d submitted — waiting...",
                 i // MAX_CONCURRENT + 1,
                 -(-len(all_tasks) // MAX_CONCURRENT))
        _wait_for_tasks(wave)

    staging.mkdir(parents=True, exist_ok=True)
    done_flag.touch()
    log.info("Stage 1 complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — GCS DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _get_gcs_client():
    """
    GCS client via gcloud ADC.
    Run once before pipeline: gcloud auth application-default login
    """
    from google.cloud import storage
    return storage.Client(project=GEE_PROJECT)


def run_gcs_download():
    """Stage 2: download all CSVs from GCS to local staging folder."""
    staging   = Path(OUTPUT_PATH) / "_csv_staging"
    done_flag = staging / ".download_done"
    if done_flag.exists():
        log.info("Stage 2 skipped — CSVs already downloaded.")
        return

    log.info("=== Stage 2: GCS Download ===")
    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blobs  = list(bucket.list_blobs(prefix=GCS_PREFIX))
    log.info("Downloading %d files from gs://%s/%s ...",
             len(blobs), GCS_BUCKET, GCS_PREFIX)

    for blob in tqdm(blobs, desc="GCS download"):
        rel  = blob.name[len(GCS_PREFIX):].lstrip("/")
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

    done_flag.touch()
    log.info("Stage 2 complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for b in S2_BANDS:
        if b in df.columns:
            df[b] = pd.to_numeric(df[b], errors="coerce")
    return df


def _build_parcel_array(parcel_df: pd.DataFrame):
    """
    Build (T, C, S) uint16 array matching original PixelSet format exactly:
      - S fixed from first valid date
      - NaN pixels stripped per date
      - shorter dates zero-padded on right
      - dtype uint16, no 65535
    """
    xp = cp if CUPY_AVAILABLE else np

    date_pixels = {}
    for didx in range(N_DATES):
        sub = parcel_df[parcel_df["date_idx"] == didx][S2_BANDS].dropna(how="any")
        if len(sub) == 0:
            date_pixels[didx] = None
        else:
            arr = np.clip(sub.values.T.astype(np.float32), 0, 65534).astype(np.uint16)
            date_pixels[didx] = arr  # (C, S_t)

    # S fixed from first valid date — matches original prepare_dataset.py
    S = None
    for didx in range(N_DATES):
        if date_pixels[didx] is not None:
            S = date_pixels[didx].shape[1]
            break

    if S is None:
        return None

    out = xp.zeros((N_DATES, N_BANDS, S), dtype=xp.uint16)
    for didx in range(N_DATES):
        px = date_pixels[didx]
        if px is None:
            continue
        s_t = px.shape[1]
        if s_t >= S:
            out[didx] = xp.array(px[:, :S], dtype=xp.uint16)
        else:
            out[didx, :, :s_t] = xp.array(px, dtype=xp.uint16)

    return cp.asnumpy(out) if CUPY_AVAILABLE else out


def run_assembly():
    """Stage 3: assemble CSVs into (T, C, S) uint16 .npy files."""
    data_dir  = Path(OUTPUT_PATH) / "DATA"
    done_flag = Path(OUTPUT_PATH) / ".assembly_done"
    if done_flag.exists():
        log.info("Stage 3 skipped — assembly already complete.")
        return

    log.info("=== Stage 3: Assembly ===")
    csv_dir = Path(OUTPUT_PATH) / "_csv_staging"
    paths   = [str(p) for p in csv_dir.rglob("*.csv")]
    log.info("Loading %d CSVs with %d workers...", len(paths), N_CSV_WORKERS)

    dfs = []
    with ProcessPoolExecutor(max_workers=N_CSV_WORKERS) as ex:
        futures = {ex.submit(_load_csv, p): p for p in paths}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Loading CSVs"):
            try:
                dfs.append(fut.result())
            except Exception as e:
                log.warning("CSV load failed: %s", e)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df[PARCEL_ID_PROP] = full_df[PARCEL_ID_PROP].astype(str)
    full_df["date_idx"]     = full_df["composite_date"].map(DATE_TO_IDX)
    log.info("Loaded %d pixel rows.", len(full_df))

    data_dir.mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT_PATH) / "META").mkdir(parents=True, exist_ok=True)

    grouped = full_df.groupby(PARCEL_ID_PROP)
    sizes   = {}
    labels  = {l: {} for l in LABEL_NAMES}
    skipped = 0

    for pid in tqdm(full_df[PARCEL_ID_PROP].unique(), desc="Building .npy"):
        arr = _build_parcel_array(grouped.get_group(pid))
        if arr is None:
            skipped += 1
            continue
        np.save(data_dir / pid, arr)
        sizes[pid] = int(arr.shape[2])
        row = grouped.get_group(pid).iloc[0]
        for l in LABEL_NAMES:
            if l in row:
                labels[l][pid] = int(row[l]) if pd.notna(row[l]) else None

    meta = Path(OUTPUT_PATH) / "META"
    _write_json(meta / "dates.json",  COMPOSITE_DATES)
    _write_json(meta / "labels.json", labels)
    _write_json(meta / "sizes.json",  sizes)

    log.info("Assembled %d parcels, skipped %d.", len(sizes), skipped)
    done_flag.touch()
    log.info("Stage 3 complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — TEMPORAL INTERPOLATION
# ══════════════════════════════════════════════════════════════════════════════

def _interpolate_array(arr: np.ndarray) -> np.ndarray:
    """
    Per-pixel linear interpolation along the time axis.
    - Fully-zero date slices filled from bracketing valid dates
    - Edge gaps: forward/backward fill (flat extrapolation)
    - Valid pixels never modified
    - dtype uint16 preserved; interpolated values clamped to [1, 65534]
    - Pixels missing across ALL dates remain zero
    """
    T, C, S = arr.shape
    t_axis  = np.arange(T, dtype=np.float32)
    out     = arr.astype(np.float32).reshape(T, C * S)
    valid   = (arr > 0).reshape(T, C * S)

    for n in range(C * S):
        vmask   = valid[:, n]
        valid_t = t_axis[vmask]
        valid_v = out[:, n][vmask]
        if len(valid_t) == 0 or len(valid_t) == T:
            continue
        out[:, n] = np.interp(t_axis, valid_t, valid_v)

    result = np.clip(out, 1, 65534).astype(np.uint16).reshape(T, C, S)

    # Re-zero pixels missing across all dates
    all_missing = ~(arr > 0).any(axis=0)  # (C, S)
    result[:, all_missing] = 0

    return result


def _interp_worker(args: tuple):
    in_path, out_path = args
    pid = Path(in_path).stem
    try:
        arr = np.load(in_path)
        if not any((arr[t] == 0).all() for t in range(arr.shape[0])):
            np.save(out_path, arr)
            return pid, "skipped"
        np.save(out_path, _interpolate_array(arr))
        return pid, "ok"
    except Exception as e:
        return pid, f"error: {e}"


def run_interpolation():
    """Stage 4: fill fully-cloudy date slices via linear interpolation."""
    interp_root = Path(str(OUTPUT_PATH) + "_interp")
    done_flag   = interp_root / ".interp_done"
    if done_flag.exists():
        log.info("Stage 4 skipped — interpolation already complete.")
        return

    log.info("=== Stage 4: Temporal Interpolation ===")

    import shutil
    (interp_root / "DATA").mkdir(parents=True, exist_ok=True)
    (interp_root / "META").mkdir(parents=True, exist_ok=True)
    for f in (Path(OUTPUT_PATH) / "META").glob("*.json"):
        shutil.copy2(f, interp_root / "META" / f.name)

    npy_files = sorted((Path(OUTPUT_PATH) / "DATA").glob("*.npy"))
    tasks = [(str(f), str(interp_root / "DATA" / f.name))
             for f in npy_files]
    log.info("Interpolating %d files with %d workers...",
             len(tasks), N_INTERP_WORKERS)

    stats = {"ok": 0, "skipped": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=N_INTERP_WORKERS) as ex:
        futures = {ex.submit(_interp_worker, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Interpolating"):
            pid, status = fut.result()
            if "error" in status:
                log.warning("Parcel %s — %s", pid, status)
            stats[status.split(":")[0]] += 1

    log.info("Done: %d filled | %d no-gap | %d errors",
             stats["ok"], stats["skipped"], stats.get("error", 0))
    done_flag.touch()
    log.info("Stage 4 complete → %s", interp_root)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class LPISPixelSetDataset(Dataset):
    """
    PyTorch Dataset for assembled PixelSet data.

    Loads (T, C, S) uint16 arrays, casts to float32, divides by 10000.
    Compatible with both GEE-assembled files and original PASTIS files.

    Args:
        root        : dataset root (contains DATA/ and META/)
        label_name  : property to use as class label (default: CODE_GROUP)
        max_pixels  : if set, parcels with S > max_pixels are randomly
                      subsampled at each __getitem__ call. Keeps large parcels
                      but limits GPU memory. Different sample each epoch =
                      free spatial data augmentation. Recommended: 500.
        min_pixels  : skip parcels with fewer pixels than this. Default: 5.
        transform   : optional callable applied to each sample dict
    """

    def __init__(self, root: str,
                 label_name: str = "CODE_GROUP",
                 max_pixels: int = None,
                 min_pixels: int = 1,
                 transform=None):
        self.root       = Path(root)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.transform  = transform

        with open(self.root / "META" / "labels.json") as f:
            all_labels = json.load(f)
        with open(self.root / "META" / "sizes.json") as f:
            self.sizes = json.load(f)

        self.label_map = all_labels.get(label_name, {})
        data_dir       = self.root / "DATA"

        self.parcel_ids = [
            pid for pid in self.sizes
            if (data_dir / f"{pid}.npy").exists()
            and pid in self.label_map
            and self.label_map[pid] is not None
            and int(self.sizes[pid]) >= min_pixels
        ]

        unique            = sorted(set(int(self.label_map[p])
                                       for p in self.parcel_ids))
        self.class_to_idx = {c: i for i, c in enumerate(unique)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        if max_pixels is not None:
            n_large = sum(1 for p in self.parcel_ids
                          if int(self.sizes[p]) > max_pixels)
            log.info("Dataset: %d parcels | %d classes | "
                     "%d parcels subsampled to max_pixels=%d | %s",
                     len(self.parcel_ids), len(self.class_to_idx),
                     n_large, max_pixels, root)
        else:
            log.info("Dataset: %d parcels | %d classes | %s",
                     len(self.parcel_ids), len(self.class_to_idx), root)

    def __len__(self) -> int:
        return len(self.parcel_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.parcel_ids[idx]
        arr = np.load(self.root / "DATA" / f"{pid}.npy")  # (T, C, S) uint16
        S   = arr.shape[2]

        # Random pixel subsampling for large parcels.
        # Different random subset each epoch = spatial data augmentation.
        if self.max_pixels is not None and S > self.max_pixels:
            indices = np.random.choice(S, self.max_pixels, replace=False)
            indices.sort()              # preserve relative spatial order
            arr = arr[:, :, indices]
            S   = self.max_pixels

        data = torch.clamp(
            torch.from_numpy(arr.astype(np.float32) / 10000.0),
            0.0, 1.0                    # guard against DN > 10000
        )
        label = torch.tensor(
            self.class_to_idx[int(self.label_map[pid])],
            dtype=torch.long
        )
        sample = {"data": data, "label": label, "parcel_id": pid, "size": S}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = torch.zeros(len(self.class_to_idx))
        for pid in self.parcel_ids:
            counts[self.class_to_idx[int(self.label_map[pid])]] += 1
        w = 1.0 / counts.clamp(min=1)
        return w / w.sum()


class CombinedLPISDataset(Dataset):
    """
    Merge multiple PixelSet datasets (e.g. original PASTIS + GEE Slovakia).
    Builds unified class_to_idx from the union of all label sets.

    Args:
        roots       : list of dataset root paths
        label_name  : label property (must be consistent across datasets)
        max_pixels  : passed to each LPISPixelSetDataset
        min_pixels  : passed to each LPISPixelSetDataset
        transform   : optional callable on each sample dict
    """

    def __init__(self, roots: list,
                 label_name: str = "CODE_GROUP",
                 max_pixels: int = None,
                 min_pixels: int = 1,
                 transform=None):
        self.transform = transform
        self.ds_list   = [
            LPISPixelSetDataset(r, label_name,
                                max_pixels=max_pixels,
                                min_pixels=min_pixels)
            for r in roots
        ]

        all_classes       = sorted(set(c for ds in self.ds_list
                                       for c in ds.class_to_idx))
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.samples = [(di, pid)
                        for di, ds in enumerate(self.ds_list)
                        for pid in ds.parcel_ids]

        log.info("CombinedDataset: %d parcels | %d classes | %d sources",
                 len(self.samples), len(self.class_to_idx), len(roots))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        di, pid   = self.samples[idx]
        ds        = self.ds_list[di]
        child_idx = ds.parcel_ids.index(pid)
        sample    = ds[child_idx]
        sample["source"] = di
        return sample

    def get_class_weights(self) -> torch.Tensor:
        counts = torch.zeros(len(self.class_to_idx))
        for di, pid in self.samples:
            ds = self.ds_list[di]
            counts[self.class_to_idx[int(ds.label_map[pid])]] += 1
        w = 1.0 / counts.clamp(min=1)
        return w / w.sum()


def collate_fn_pad(batch: list) -> dict:
    """
    Pad variable-S tensors within a batch to the largest S in that batch.
    The 'size' field carries real pixel counts for attention masking
    (e.g. PSE-TAE uses this to ignore padding in the pooling step).
    """
    max_s  = max(s["data"].shape[2] for s in batch)
    T, C   = batch[0]["data"].shape[:2]
    B      = len(batch)
    padded = torch.zeros(B, T, C, max_s, dtype=torch.float32)
    labels = torch.zeros(B, dtype=torch.long)
    sizes, pids = [], []

    for i, s in enumerate(batch):
        si = s["data"].shape[2]
        padded[i, :, :, :si] = s["data"]
        labels[i] = s["label"]
        sizes.append(si)
        pids.append(s["parcel_id"])

    return {
        "data"      : padded,               # (B, T, C, S_max)
        "label"     : labels,               # (B,)
        "size"      : torch.tensor(sizes),  # (B,) real pixel count per sample
        "parcel_id" : pids,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def validate(root: str, n: int = 200):
    files = list((Path(root) / "DATA").glob("*.npy"))
    if not files:
        log.warning("No .npy files found in %s", root)
        return
    log.info("Validating %d files (sample=%d) in %s", len(files), n, root)
    s_vals, zero_slices, bad = [], [], []
    for f in files[:n]:
        arr = np.load(f)
        s_vals.append(arr.shape[2])
        zero_slices.append(sum((arr[t] == 0).all()
                               for t in range(arr.shape[0])))
        if arr.dtype != np.uint16:
            bad.append(f.name)
    log.info("T=%d C=%d  S min=%d max=%d mean=%.1f",
             N_DATES, N_BANDS,
             min(s_vals), max(s_vals), sum(s_vals) / len(s_vals))
    log.info("Fully-zero date slices/parcel — mean=%.2f  (0 = fully interpolated)",
             sum(zero_slices) / len(zero_slices))
    if bad:
        log.warning("Wrong dtype: %s", bad)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Stages 1-4: data preparation ─────────────────────────────────────────
    run_gee_export()
    run_gcs_download()
    run_assembly()
    run_interpolation()

    # ── Stage 5: validate and build DataLoader ────────────────────────────────
    interp_root = str(OUTPUT_PATH) + "_interp"
    validate(interp_root)

    # Single dataset (Slovakia only)
    dataset = LPISPixelSetDataset(
        root=interp_root,
        label_name="CODE_GROUP",
        max_pixels=MAX_PIXELS,   # large parcels subsampled, not excluded
        min_pixels=MIN_PIXELS,   # very small parcels excluded
    )

    # Combined with original PASTIS dataset — uncomment if available
    # dataset = CombinedLPISDataset(
    #     roots=["./PixelSet-S2-2017-T31TFM", interp_root],
    #     label_name="CODE_GROUP",
    #     max_pixels=MAX_PIXELS,
    #     min_pixels=MIN_PIXELS,
    # )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_pad,
    )

    batch  = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=== Ready for training ===")
    log.info("Batch data   : %s  float32", tuple(batch["data"].shape))
    log.info("Batch labels : %s  long",    tuple(batch["label"].shape))
    log.info("Value range  : [%.4f, %.4f]",
             batch["data"].min(), batch["data"].max())
    log.info("Device       : %s", device)
    log.info("Classes      : %d", len(dataset.class_to_idx))

    weights   = dataset.get_class_weights().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    log.info("CrossEntropyLoss with %d class weights ready.", len(weights))