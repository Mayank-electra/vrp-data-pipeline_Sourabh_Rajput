"""
Data loader module for ingesting, merging, aligning, and parsing EV telemetry CSV datasets.
"""

import glob
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def generate_synthetic_vrp_data(num_rows: int = 500, output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Generates a realistic synthetic VRP / EV telemetry dataset for testing or fallback.
    """
    np.random.seed(42)
    timestamps = pd.date_range("2026-10-27 08:00:00+00:00", periods=num_rows, freq="10s")
    
    # Simulate speed and battery dynamics
    speed = np.clip(np.random.normal(loc=35.0, scale=18.0, size=num_rows), 0, 120)
    soc = np.clip(95.0 - np.cumsum(np.random.exponential(scale=0.015, size=num_rows)), 10.0, 100.0)
    motor_rpm = speed * 45.0 + np.random.normal(0, 50, size=num_rows)
    pack_voltage = 330.0 + (soc / 100.0) * 45.0 + np.random.normal(0, 1.5, size=num_rows)
    pack_current = (speed / 120.0) * 80.0 + np.random.normal(5, 4, size=num_rows)
    power_kw = (pack_voltage * pack_current) / 1000.0
    
    min_cell_mv = 3100 + (soc / 100.0) * 350 + np.random.normal(0, 10, size=num_rows)
    cell_delta_mv = np.clip(np.random.exponential(scale=6.0, size=num_rows), 1.0, 40.0)
    max_cell_mv = min_cell_mv + cell_delta_mv
    
    # Introduce some realistic missingness in peripheral sensors
    batt_temp = np.random.normal(28.0, 2.5, size=num_rows)
    coolant_in = np.where(np.random.rand(num_rows) > 0.4, np.nan, 26.0 + np.random.normal(0, 1, size=num_rows))
    coolant_out = np.where(np.random.rand(num_rows) > 0.5, np.nan, 29.0 + np.random.normal(0, 1, size=num_rows))
    
    # High missingness column (>60% missing) to test dropping
    ccs_station_a = np.where(np.random.rand(num_rows) > 0.15, np.nan, np.random.uniform(50, 150, size=num_rows))
    
    df = pd.DataFrame({
        "ts_utc_original": [t.isoformat() if i % 2 == 0 else t.strftime("%Y-%m-%dT%H:%M:%SZ") for i, t in enumerate(timestamps)],
        "soc": soc,
        "pack_voltage": pack_voltage,
        "pack_current": pack_current,
        "power_kw": power_kw,
        "speed_kmh": speed,
        "motor_rpm": motor_rpm,
        "min_cell_mv": min_cell_mv,
        "max_cell_mv": max_cell_mv,
        "cell_delta_mv": cell_delta_mv,
        "batt_temp": batt_temp,
        "coolant_in": coolant_in,
        "coolant_out": coolant_out,
        "ccs_station_a": ccs_station_a,
        "is_charging": np.random.choice(["FALSE", "TRUE"], size=num_rows, p=[0.9, 0.1]),
        "drive_mode": np.random.choice(["ECO", "NORMAL", "SPORT"], size=num_rows, p=[0.3, 0.5, 0.2]),
    })
    
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info(f"Generated synthetic VRP dataset at {out_p} with {len(df)} rows.")
        
    return df


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Standardizes timestamps handling ISO formats with +00:00, Z, epoch ms, or standard strings.
    Converts to UTC datetime.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)

    # Check if numeric (epoch milliseconds or seconds)
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().mean() > 0.8:
        # Check scale: epoch ms (~1e12) vs seconds (~1e9)
        max_val = numeric_series.dropna().max()
        if max_val > 1e11:
            return pd.to_datetime(numeric_series, unit="ms", utc=True)
        elif max_val > 1e8:
            return pd.to_datetime(numeric_series, unit="s", utc=True)

    # Mixed ISO strings (e.g. 2026-10-27T04:39:24.159+00:00 and 2026-10-27T04:39:24.159Z)
    clean_series = series.astype(str).str.replace("Z", "+00:00", regex=False)
    try:
        parsed = pd.to_datetime(clean_series, utc=True, format="ISO8601", errors="coerce")
    except Exception:
        parsed = pd.to_datetime(clean_series, utc=True, format="mixed", errors="coerce")
    
    # If any NaTs remain, try row-wise parsing fallback
    if parsed.isna().sum() > 0:
        mask = parsed.isna()
        parsed.loc[mask] = pd.to_datetime(clean_series.loc[mask], utc=True, errors="coerce")
        
    return parsed


def load_dataset(
    input_path: Optional[Union[str, Path, List[str]]] = None,
    auto_generate_if_missing: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads one or multiple CSV files, aligns mismatched columns, parses timestamps,
    and returns a consolidated DataFrame with loading metadata.
    """
    metadata = {
        "files_loaded": [],
        "initial_rows": 0,
        "initial_cols": 0,
        "columns_found": [],
        "has_timestamp": False,
        "timestamp_col": None,
        "time_range": None,
    }

    file_paths: List[Path] = []

    if input_path is None or (isinstance(input_path, str) and not input_path.strip()):
        # Try raw directory
        raw_csvs = list(RAW_DATA_DIR.glob("*.csv"))
        if raw_csvs:
            file_paths = [sorted(raw_csvs)[0]]
        elif auto_generate_if_missing:
            sample_path = RAW_DATA_DIR / "sample_vrp.csv"
            generate_synthetic_vrp_data(num_rows=500, output_path=sample_path)
            file_paths = [sample_path]
    elif isinstance(input_path, list):
        for p in input_path:
            file_paths.extend([Path(x) for x in glob.glob(str(p))])
    elif isinstance(input_path, (str, Path)):
        p_str = str(input_path)
        if "," in p_str:
            for part in p_str.split(","):
                file_paths.extend([Path(x) for x in glob.glob(part.strip())])
        elif os.path.isdir(p_str):
            file_paths.extend(sorted(Path(p_str).glob("*.csv")))
        elif "*" in p_str:
            file_paths.extend([Path(x) for x in sorted(glob.glob(p_str))])
        else:
            file_paths = [Path(p_str)]

    # If no files found, check fallback
    existing_files = [f for f in file_paths if f.exists()]
    if not existing_files:
        if auto_generate_if_missing:
            fallback = RAW_DATA_DIR / "sample_vrp.csv"
            generate_synthetic_vrp_data(num_rows=500, output_path=fallback)
            existing_files = [fallback]
        else:
            raise FileNotFoundError(f"No CSV files found matching: {input_path}")

    dfs = []
    for fp in existing_files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            if "source_file" not in df.columns:
                df["source_file"] = fp.name
            dfs.append(df)
            metadata["files_loaded"].append(str(fp))
            logger.info(f"Loaded {fp.name}: {df.shape[0]} rows, {df.shape[1]} columns.")
        except Exception as e:
            logger.error(f"Error loading {fp}: {e}")
            raise

    # Align columns across multiple files
    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        # Align via pd.concat (automatically unions columns)
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        logger.info(f"Merged {len(dfs)} files into DataFrame with shape {combined_df.shape}.")

    # Identify and standardize timestamp column
    ts_candidates = ["ts_utc_original", "ts", "timestamp", "datetime", "time", "date_time", "ts_utc"]
    matched_ts = None
    for cand in ts_candidates:
        if cand in combined_df.columns:
            matched_ts = cand
            break

    if matched_ts is not None:
        parsed_ts = parse_timestamp_series(combined_df[matched_ts])
        if parsed_ts.notna().sum() > 0:
            combined_df["ts_utc"] = parsed_ts
            # Sort chronologically
            combined_df = combined_df.sort_values(by="ts_utc").reset_index(drop=True)
            metadata["has_timestamp"] = True
            metadata["timestamp_col"] = "ts_utc"
            valid_ts = combined_df["ts_utc"].dropna()
            if not valid_ts.empty:
                metadata["time_range"] = {
                    "start": str(valid_ts.min()),
                    "end": str(valid_ts.max())
                }
            logger.info(f"Standardized timestamp column '{matched_ts}' to 'ts_utc' ({metadata['time_range']}).")

    metadata["initial_rows"] = int(combined_df.shape[0])
    metadata["initial_cols"] = int(combined_df.shape[1])
    metadata["columns_found"] = list(combined_df.columns)

    return combined_df, metadata
