#!/usr/bin/env python3
"""Data Preprocessor MCP Server - ML-ready data preprocessing for educational use."""

import os
import sys
import uuid
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("data-preprocessor-server")

mcp = FastMCP("data-preprocessor")

_datasets = {}

DATA_DIR = os.environ.get("DATA_DIR", "/data")
MAX_ROWS = int(os.environ.get("MAX_PREVIEW_ROWS", "20"))


def _no_dataset(dataset_id):
    """Return True if the dataset_id is missing or not loaded."""
    return not dataset_id.strip() or dataset_id not in _datasets


def _get_dataset_state(dataset_id):
    """Return the stored state for a dataset_id."""
    return _datasets[dataset_id]


def _dtype_str(df):
    """Return a compact string summary of column dtypes."""
    return ", ".join(f"{c}({str(t)})" for c, t in df.dtypes.items())


# ============================================================
# TOOL 1: load_dataset
# ============================================================
@mcp.tool()
async def load_dataset(filename: str = "", file_type: str = "") -> str:
    """Load a dataset from /data directory. Supports csv, excel, parquet, json."""
    if not filename.strip():
        return "filename is required."
    filepath = os.path.join(DATA_DIR, filename.strip())
    if not os.path.isfile(filepath):
        try:
            files = os.listdir(DATA_DIR)
        except Exception:
            files = []
        return f"File not found: {filepath}\nAvailable in {DATA_DIR}: {', '.join(files) or 'none'}"
    ext = file_type.strip().lower() or os.path.splitext(filename)[1].lower().lstrip(".")
    try:
        if ext in ("csv", "txt"):
            df = pd.read_csv(filepath)
        elif ext in ("xlsx", "xls", "excel"):
            df = pd.read_excel(filepath)
        elif ext in ("parquet", "pq"):
            df = pd.read_parquet(filepath)
        elif ext == "json":
            df = pd.read_json(filepath)
        else:
            return f"Unsupported type '{ext}'. Supported: csv, excel, parquet, json."

        dataset_id = str(uuid.uuid4())
        _datasets[dataset_id] = {
            "dataset": df.copy(),
            "original_dataset": df.copy(),
            "dataset_name": filename.strip()
        }

        return (
            f"Loaded '{filename.strip()}'\n"
            f"dataset_id: {dataset_id}\n"
            f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n"
            f"Columns: {', '.join(df.columns.tolist())}\n"
            f"Types: {_dtype_str(df)}"
        )
    except Exception as e:
        logger.error(f"load_dataset: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 2: get_info
# ============================================================
@mcp.tool()
async def get_info(dataset_id: str = "") -> str:
    """Show dataset info: shape, column types, null counts, and memory usage."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        dataset_name = state["dataset_name"]
        rows, cols = df.shape
        lines = [
            f"Dataset: {dataset_name}",
            f"dataset_id: {dataset_id}",
            f"Shape: {rows} rows × {cols} columns",
            "",
            f"{'Column':<28} {'Type':<15} {'Non-Null':<12} {'Nulls':<8}",
            f"{'-'*65}",
        ]
        for col in df.columns:
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            lines.append(f"{col:<28} {str(df[col].dtype):<15} {non_null:<12} {null_count:<8}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"get_info: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 3: preview_data
# ============================================================
@mcp.tool()
async def preview_data(dataset_id: str = "", n_rows: str = "5", position: str = "head") -> str:
    """Preview first or last N rows of the dataset. position: head or tail."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    try:
        state = _get_dataset_state(dataset_id)
        df_current = state["dataset"]
        dataset_name = state["dataset_name"]
        n = int(n_rows.strip()) if n_rows.strip() else 5
        n = max(1, min(n, MAX_ROWS))
        pos = position.strip().lower()
        df = df_current.tail(n) if pos == "tail" else df_current.head(n)
        return f"{pos.upper()} {n} rows of '{dataset_name}':\n{df.to_string(index=True)}"
    except Exception as e:
        logger.error(f"preview_data: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 4: get_statistics
# ============================================================
@mcp.tool()
async def get_statistics(dataset_id: str = "", columns: str = "") -> str:
    """Show descriptive statistics. Pass comma-separated column names or leave empty for all."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        dataset_name = state["dataset_name"]
        if columns.strip():
            cols = [c.strip() for c in columns.split(",")]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                return f"Columns not found: {', '.join(missing)}"
            df = df[cols]
        stats = df.describe(include="all").round(4)
        return f"Statistics for '{dataset_name}':\n{stats.to_string()}"
    except Exception as e:
        logger.error(f"get_statistics: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 5: get_missing_summary
# ============================================================
@mcp.tool()
async def get_missing_summary(dataset_id: str = "") -> str:
    """Show missing value count and percentage for each column."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        dataset_name = state["dataset_name"]
        total = len(df)
        lines = [
            f"Missing Values in '{dataset_name}' ({total} rows total):",
            "",
            f"{'Column':<28} {'Missing':<10} {'Percent':<10} {'Status'}",
            f"{'-'*58}",
        ]
        has_missing = False
        for col in df.columns:
            missing = df[col].isna().sum()
            pct = (missing / total * 100) if total > 0 else 0.0
            status = "OK" if missing == 0 else "HAS NULLS"
            if missing > 0:
                has_missing = True
            lines.append(f"{col:<28} {missing:<10} {pct:.1f}%{'':<5} {status}")
        if not has_missing:
            lines.append("\nNo missing values found!")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"get_missing_summary: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 6: drop_columns
# ============================================================
@mcp.tool()
async def drop_columns(dataset_id: str = "", columns: str = "") -> str:
    """Drop specified columns from the dataset. Provide comma-separated column names."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    if not columns.strip():
        return "columns is required. Provide comma-separated column names."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {', '.join(missing)}\nAvailable: {', '.join(df.columns.tolist())}"
        state["dataset"] = df.drop(columns=cols)
        df = state["dataset"]
        return f"Dropped: {', '.join(cols)}\nNew shape: {df.shape[0]} rows × {df.shape[1]} cols\nRemaining: {', '.join(df.columns.tolist())}"
    except Exception as e:
        logger.error(f"drop_columns: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 7: select_columns
# ============================================================
@mcp.tool()
async def select_columns(dataset_id: str = "", columns: str = "") -> str:
    """Keep only the specified columns. Provide comma-separated column names."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    if not columns.strip():
        return "columns is required. Provide comma-separated column names."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {', '.join(missing)}\nAvailable: {', '.join(df.columns.tolist())}"
        state["dataset"] = df[cols]
        df = state["dataset"]
        return f"Selected {len(cols)} columns: {', '.join(cols)}\nNew shape: {df.shape[0]} rows × {df.shape[1]} cols"
    except Exception as e:
        logger.error(f"select_columns: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 8: drop_duplicates
# ============================================================
@mcp.tool()
async def drop_duplicates(dataset_id: str = "", subset: str = "", keep: str = "first") -> str:
    """Remove duplicate rows. subset: comma-separated columns (empty=all). keep: first, last, or none."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        before = len(df)
        subset_cols = [c.strip() for c in subset.split(",") if c.strip()] if subset.strip() else None
        keep_val = keep.strip().lower() if keep.strip() else "first"
        if keep_val == "none":
            keep_val = False
        elif keep_val not in ("first", "last"):
            keep_val = "first"
        state["dataset"] = df.drop_duplicates(subset=subset_cols, keep=keep_val)
        df = state["dataset"]
        removed = before - len(df)
        return f"Removed {removed} duplicate rows\nNew shape: {df.shape[0]} rows × {df.shape[1]} cols"
    except Exception as e:
        logger.error(f"drop_duplicates: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 9: drop_rows_with_missing
# ============================================================
@mcp.tool()
async def drop_rows_with_missing(dataset_id: str = "", how: str = "any", subset: str = "") -> str:
    """Drop rows with missing values. how: any (default) or all. subset: optional comma-separated columns."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        before = len(df)
        how_val = how.strip().lower() if how.strip() in ("any", "all") else "any"
        subset_cols = [c.strip() for c in subset.split(",") if c.strip()] if subset.strip() else None
        state["dataset"] = df.dropna(how=how_val, subset=subset_cols)
        df = state["dataset"]
        removed = before - len(df)
        return f"Dropped {removed} rows with missing values (how='{how_val}')\nNew shape: {df.shape[0]} rows × {df.shape[1]} cols"
    except Exception as e:
        logger.error(f"drop_rows_with_missing: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 10: fill_missing
# ============================================================
@mcp.tool()
async def fill_missing(dataset_id: str = "", column: str = "", strategy: str = "mean", value: str = "") -> str:
    """Fill missing values in a column. strategy: mean, median, mode, constant, ffill, bfill."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found.\nAvailable: {', '.join(df.columns.tolist())}"
    try:
        strat = strategy.strip().lower() if strategy.strip() else "mean"
        before = df[col].isna().sum()
        if strat == "mean":
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
        elif strat == "median":
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
        elif strat == "mode":
            mode_vals = df[col].mode()
            if mode_vals.empty:
                return f"Cannot compute mode for '{col}' (all values are null)."
            df[col] = df[col].fillna(mode_vals[0])
        elif strat == "constant":
            if not value.strip():
                return "value is required when strategy is 'constant'."
            fill_val = value.strip()
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    fill_val = float(fill_val)
                    if fill_val == int(fill_val):
                        fill_val = int(fill_val)
                except ValueError:
                    pass
            df[col] = df[col].fillna(fill_val)
        elif strat == "ffill":
            df[col] = df[col].ffill()
        elif strat == "bfill":
            df[col] = df[col].bfill()
        else:
            return f"Unknown strategy '{strat}'. Use: mean, median, mode, constant, ffill, bfill."
        state["dataset"] = df
        after = df[col].isna().sum()
        return f"Filled {before - after} missing values in '{col}' using strategy='{strat}'\nRemaining nulls in '{col}': {after}"
    except Exception as e:
        logger.error(f"fill_missing: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 11: rename_column
# ============================================================
@mcp.tool()
async def rename_column(dataset_id: str = "", old_name: str = "", new_name: str = "") -> str:
    """Rename a column in the dataset."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not old_name.strip() or not new_name.strip():
        return "Both old_name and new_name are required."
    if old_name.strip() not in df.columns:
        return f"Column '{old_name}' not found.\nAvailable: {', '.join(df.columns.tolist())}"
    try:
        state["dataset"] = df.rename(columns={old_name.strip(): new_name.strip()})
        df = state["dataset"]
        return f"Renamed '{old_name}' → '{new_name}'\nColumns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        logger.error(f"rename_column: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 12: filter_rows
# ============================================================
@mcp.tool()
async def filter_rows(dataset_id: str = "", column: str = "", operator: str = "==", value: str = "") -> str:
    """Filter rows by condition. operator: ==, !=, >, <, >=, <=, contains, startswith, endswith, isnull, notnull."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        before = len(df)
        op = operator.strip().lower()
        ser = df[col]
        if op == "isnull":
            mask = ser.isna()
        elif op == "notnull":
            mask = ser.notna()
        elif op == "contains":
            mask = ser.astype(str).str.contains(value.strip(), na=False)
        elif op == "startswith":
            mask = ser.astype(str).str.startswith(value.strip(), na=False)
        elif op == "endswith":
            mask = ser.astype(str).str.endswith(value.strip(), na=False)
        else:
            try:
                v_num = float(value.strip())
                if op == "==":
                    mask = ser == v_num
                elif op == "!=":
                    mask = ser != v_num
                elif op == ">":
                    mask = ser > v_num
                elif op == "<":
                    mask = ser < v_num
                elif op == ">=":
                    mask = ser >= v_num
                elif op == "<=":
                    mask = ser <= v_num
                else:
                    return f"Unknown operator '{op}'. Use: ==, !=, >, <, >=, <=, contains, startswith, endswith, isnull, notnull."
            except (ValueError, TypeError):
                v_str = value.strip()
                if op == "==":
                    mask = ser == v_str
                elif op == "!=":
                    mask = ser != v_str
                else:
                    return f"Operator '{op}' requires a numeric column. For strings use: ==, !=, contains, startswith, endswith."
        state["dataset"] = df[mask].reset_index(drop=True)
        kept = len(state["dataset"])
        return f"Filter: '{col}' {op} '{value}'\nKept {kept} rows, removed {before - kept} rows"
    except Exception as e:
        logger.error(f"filter_rows: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 13: encode_categorical
# ============================================================
@mcp.tool()
async def encode_categorical(dataset_id: str = "", column: str = "", method: str = "label") -> str:
    """Encode a categorical column. method: label (integer codes) or onehot (dummy columns)."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        m = method.strip().lower() if method.strip() else "label"
        if m == "label":
            le = LabelEncoder()
            null_mask = df[col].isna()
            non_null_vals = df.loc[~null_mask, col].astype(str)
            encoded_arr = le.fit_transform(non_null_vals)
            result_arr = np.full(len(df), np.nan, dtype=float)
            result_arr[~null_mask.values] = encoded_arr.astype(float)
            df[col + "_encoded"] = result_arr
            state["dataset"] = df
            mapping = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
            return f"Label encoded '{col}' → '{col}_encoded'\nMapping: {mapping}"
        elif m == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            state["dataset"] = pd.concat([df, dummies], axis=1)
            new_cols = dummies.columns.tolist()
            return f"One-hot encoded '{col}' → {len(new_cols)} new columns\nNew columns: {', '.join(new_cols)}"
        else:
            return f"Unknown method '{m}'. Use: label or onehot."
    except Exception as e:
        logger.error(f"encode_categorical: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 14: normalize_column
# ============================================================
@mcp.tool()
async def normalize_column(dataset_id: str = "", column: str = "", method: str = "minmax") -> str:
    """Normalize a numeric column. method: minmax (0-1 range) or zscore (mean=0, std=1)."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        m = method.strip().lower() if method.strip() else "minmax"
        values = df[[col]].copy()
        new_col = f"{col}_normalized"
        if m == "minmax":
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)
            df[new_col] = scaled
            state["dataset"] = df
            orig_min = float(values[col].min())
            orig_max = float(values[col].max())
            return f"Min-max normalized '{col}' → '{new_col}'\nOriginal range: [{orig_min:.4f}, {orig_max:.4f}] → [0.0, 1.0]"
        elif m == "zscore":
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values)
            df[new_col] = scaled
            state["dataset"] = df
            orig_mean = float(values[col].mean())
            orig_std = float(values[col].std())
            return f"Z-score normalized '{col}' → '{new_col}'\nOriginal: mean={orig_mean:.4f}, std={orig_std:.4f} → mean≈0, std≈1"
        else:
            return f"Unknown method '{m}'. Use: minmax or zscore."
    except Exception as e:
        logger.error(f"normalize_column: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 15: convert_column_type
# ============================================================
@mcp.tool()
async def convert_column_type(dataset_id: str = "", column: str = "", target_type: str = "") -> str:
    """Convert a column to a different data type. target_type: int, float, str, bool, datetime."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip() or not target_type.strip():
        return "Both column and target_type are required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        t = target_type.strip().lower()
        before_dtype = str(df[col].dtype)
        if t == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif t == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif t == "str":
            df[col] = df[col].astype(str)
        elif t == "bool":
            df[col] = df[col].astype(bool)
        elif t == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            return f"Unknown type '{t}'. Use: int, float, str, bool, datetime."
        state["dataset"] = df
        after_dtype = str(df[col].dtype)
        return f"Converted '{col}': {before_dtype} → {after_dtype}"
    except Exception as e:
        logger.error(f"convert_column_type: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 16: feature_engineering_date
# ============================================================
@mcp.tool()
async def feature_engineering_date(dataset_id: str = "", column: str = "", features: str = "year,month,day") -> str:
    """Extract date/time features from a datetime column. features: year, month, day, hour, minute, dayofweek, quarter, weekofyear, dayofyear."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        feat_list = [f.strip().lower() for f in features.split(",") if f.strip()]
        valid = ["year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter", "weekofyear", "dayofyear"]
        invalid = [f for f in feat_list if f not in valid]
        if invalid:
            return f"Invalid features: {', '.join(invalid)}\nValid options: {', '.join(valid)}"
        created = []
        for feat in feat_list:
            new_col = f"{col}_{feat}"
            if feat == "year":
                df[new_col] = dt_series.dt.year
            elif feat == "month":
                df[new_col] = dt_series.dt.month
            elif feat == "day":
                df[new_col] = dt_series.dt.day
            elif feat == "hour":
                df[new_col] = dt_series.dt.hour
            elif feat == "minute":
                df[new_col] = dt_series.dt.minute
            elif feat == "second":
                df[new_col] = dt_series.dt.second
            elif feat == "dayofweek":
                df[new_col] = dt_series.dt.dayofweek
            elif feat == "quarter":
                df[new_col] = dt_series.dt.quarter
            elif feat == "weekofyear":
                df[new_col] = dt_series.dt.isocalendar().week.astype(int)
            elif feat == "dayofyear":
                df[new_col] = dt_series.dt.dayofyear
            created.append(new_col)
        state["dataset"] = df
        return f"Extracted {len(created)} features from '{col}'\nNew columns: {', '.join(created)}"
    except Exception as e:
        logger.error(f"feature_engineering_date: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 17: compute_correlation
# ============================================================
@mcp.tool()
async def compute_correlation(dataset_id: str = "", columns: str = "", method: str = "pearson") -> str:
    """Compute correlation matrix. columns: comma-separated (empty=all numeric). method: pearson, spearman, kendall."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        if columns.strip():
            cols = [c.strip() for c in columns.split(",") if c.strip()]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                return f"Columns not found: {', '.join(missing)}"
            df = df[cols]
        m = method.strip().lower() if method.strip() else "pearson"
        if m not in ("pearson", "spearman", "kendall"):
            return f"Unknown method '{m}'. Use: pearson, spearman, kendall."
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return "No numeric columns available for correlation."
        corr = numeric_df.corr(method=m).round(4)
        return f"{m.title()} Correlation Matrix:\n{corr.to_string()}"
    except Exception as e:
        logger.error(f"compute_correlation: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 18: detect_outliers
# ============================================================
@mcp.tool()
async def detect_outliers(dataset_id: str = "", column: str = "", method: str = "iqr", threshold: str = "", remove: str = "false") -> str:
    """Detect outliers in a numeric column. method: iqr (default threshold 1.5) or zscore (default threshold 3.0). remove: true/false."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        m = method.strip().lower() if method.strip() else "iqr"
        ser = pd.to_numeric(df[col], errors="coerce")
        if m == "iqr":
            thresh = float(threshold.strip()) if threshold.strip() else 1.5
            q1 = ser.quantile(0.25)
            q3 = ser.quantile(0.75)
            iqr_val = q3 - q1
            lower = q1 - thresh * iqr_val
            upper = q3 + thresh * iqr_val
            outlier_mask = (ser < lower) | (ser > upper)
            detail = f"Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr_val:.4f}, bounds=[{lower:.4f}, {upper:.4f}]"
        elif m == "zscore":
            thresh = float(threshold.strip()) if threshold.strip() else 3.0
            z_scores = (ser - ser.mean()) / ser.std()
            outlier_mask = z_scores.abs() > thresh
            detail = f"mean={ser.mean():.4f}, std={ser.std():.4f}, threshold=|z|>{thresh}"
        else:
            return f"Unknown method '{m}'. Use: iqr or zscore."
        n_outliers = int(outlier_mask.sum())
        sample_vals = ser[outlier_mask].tolist()[:10]
        result = (
            f"Outlier Detection in '{col}' ({m.upper()}):\n"
            f"{detail}\n"
            f"Outliers found: {n_outliers} ({n_outliers / len(ser) * 100:.1f}%)\n"
            f"Sample outlier values: {sample_vals}"
        )
        should_remove = remove.strip().lower() in ("true", "yes", "1")
        if should_remove and n_outliers > 0:
            before = len(df)
            state["dataset"] = df[~outlier_mask].reset_index(drop=True)
            df = state["dataset"]
            result += f"\nRemoved {before - len(df)} outlier rows\nNew shape: {df.shape[0]} rows × {df.shape[1]} cols"
        elif should_remove:
            result += "\nNo outliers to remove."
        return result
    except Exception as e:
        logger.error(f"detect_outliers: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 19: apply_log_transform
# ============================================================
@mcp.tool()
async def apply_log_transform(dataset_id: str = "", column: str = "", method: str = "log1p") -> str:
    """Apply log transformation to a numeric column. method: log (natural log) or log1p (log(1+x), handles zeros)."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        m = method.strip().lower() if method.strip() else "log1p"
        ser = pd.to_numeric(df[col], errors="coerce")
        new_col = f"{col}_{m}"
        if m == "log":
            neg_count = int((ser <= 0).sum())
            if neg_count > 0:
                return f"Column '{col}' has {neg_count} values ≤ 0. Use method='log1p' instead, or filter first."
            df[new_col] = np.log(ser)
        elif m == "log1p":
            neg_count = int((ser < -1).sum())
            if neg_count > 0:
                return f"Column '{col}' has {neg_count} values < -1. Cannot apply log1p."
            df[new_col] = np.log1p(ser)
        else:
            return f"Unknown method '{m}'. Use: log or log1p."
        state["dataset"] = df
        new_ser = df[new_col]
        return f"Applied {m} to '{col}' → '{new_col}'\nNew stats: min={new_ser.min():.4f}, max={new_ser.max():.4f}, mean={new_ser.mean():.4f}"
    except Exception as e:
        logger.error(f"apply_log_transform: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 20: bin_column
# ============================================================
@mcp.tool()
async def bin_column(dataset_id: str = "", column: str = "", bins: str = "5", labels: str = "", strategy: str = "equal_width") -> str:
    """Bin a numeric column into categories. strategy: equal_width or quantile. labels: comma-separated (optional)."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        n_bins = int(bins.strip()) if bins.strip() else 5
        label_list = [lb.strip() for lb in labels.split(",") if lb.strip()] if labels.strip() else None
        if label_list and len(label_list) != n_bins:
            return f"Number of labels ({len(label_list)}) must match number of bins ({n_bins})."
        ser = pd.to_numeric(df[col], errors="coerce")
        new_col = f"{col}_binned"
        strat = strategy.strip().lower() if strategy.strip() else "equal_width"
        if strat == "equal_width":
            df[new_col] = pd.cut(ser, bins=n_bins, labels=label_list)
        elif strat == "quantile":
            df[new_col] = pd.qcut(ser, q=n_bins, labels=label_list, duplicates="drop")
        else:
            return f"Unknown strategy '{strat}'. Use: equal_width or quantile."
        state["dataset"] = df
        value_counts = df[new_col].value_counts().sort_index()
        return f"Binned '{col}' → '{new_col}' ({strat}, {n_bins} bins)\nDistribution:\n{value_counts.to_string()}"
    except Exception as e:
        logger.error(f"bin_column: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 21: sort_data
# ============================================================
@mcp.tool()
async def sort_data(dataset_id: str = "", column: str = "", ascending: str = "true") -> str:
    """Sort the dataset by a column. ascending: true (default) or false for descending order."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    df = state["dataset"]
    if not column.strip():
        return "column is required."
    col = column.strip()
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        asc = ascending.strip().lower() not in ("false", "no", "0")
        state["dataset"] = df.sort_values(by=col, ascending=asc).reset_index(drop=True)
        df = state["dataset"]
        direction = "ascending" if asc else "descending"
        return f"Sorted by '{col}' ({direction})\nFirst 3 values: {df[col].head(3).tolist()}\nLast 3 values: {df[col].tail(3).tolist()}"
    except Exception as e:
        logger.error(f"sort_data: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 22: export_dataset
# ============================================================
@mcp.tool()
async def export_dataset(dataset_id: str = "", filename: str = "", file_type: str = "csv") -> str:
    """Export the current dataset to /data directory. file_type: csv, excel, parquet, json."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    if not filename.strip():
        return "filename is required."
    try:
        state = _get_dataset_state(dataset_id)
        df = state["dataset"]
        filepath = os.path.join(DATA_DIR, filename.strip())
        ft = file_type.strip().lower() if file_type.strip() else "csv"
        if ft == "csv":
            df.to_csv(filepath, index=False)
        elif ft in ("xlsx", "excel"):
            df.to_excel(filepath, index=False)
        elif ft in ("parquet", "pq"):
            df.to_parquet(filepath, index=False)
        elif ft == "json":
            df.to_json(filepath, orient="records", indent=2)
        else:
            return f"Unsupported type '{ft}'. Use: csv, excel, parquet, json."
        file_size = os.path.getsize(filepath) / 1024
        return f"Exported to {filepath}\nFile size: {file_size:.1f} KB\nShape: {df.shape[0]} rows × {df.shape[1]} cols"
    except Exception as e:
        logger.error(f"export_dataset: {e}")
        return f"Error: {str(e)}"


# ============================================================
# TOOL 23: reset_dataset
# ============================================================
@mcp.tool()
async def reset_dataset(dataset_id: str = "") -> str:
    """Reset the dataset to the originally loaded version, undoing all changes."""
    if _no_dataset(dataset_id):
        return "No dataset loaded for the provided dataset_id. Use load_dataset first."
    state = _get_dataset_state(dataset_id)
    if state["original_dataset"] is None:
        return "No original dataset snapshot available."
    try:
        state["dataset"] = state["original_dataset"].copy()
        df = state["dataset"]
        dataset_name = state["dataset_name"]
        return f"Dataset reset to original '{dataset_name}'\nShape: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        logger.error(f"reset_dataset: {e}")
        return f"Error: {str(e)}"


# ============================================================
# SERVER STARTUP
# ============================================================
if __name__ == "__main__":
    logger.info("Starting Data Preprocessor MCP server...")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"MAX_PREVIEW_ROWS: {MAX_ROWS}")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)