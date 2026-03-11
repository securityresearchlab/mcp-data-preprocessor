# Data Preprocessor MCP Server

Advanced ML-ready data preprocessing for educational and development use.

## Purpose

This MCP server provides a secure, containerised interface for AI assistants to load, explore, clean, transform, and export datasets using `pandas`, `numpy`, `scikit-learn`, `pyarrow`, and `openpyxl`.

## Features — 23 Tools

### Data Loading and Inspection

- `load_dataset` — Load a file from your data directory (`CSV`, `Excel`, `Parquet`, `JSON`)
- `get_info` — Shape, column types, null counts, memory usage
- `preview_data` — First or last `N` rows of the dataset
- `get_statistics` — Descriptive statistics (`count`, `mean`, `std`, `min`, `max`, etc.)
- `get_missing_summary` — Per-column missing value count and percentage

### Data Cleaning

- `drop_columns` — Remove one or more columns
- `select_columns` — Keep only specified columns
- `drop_duplicates` — Remove duplicate rows, optionally on a subset of columns
- `drop_rows_with_missing` — Drop rows that contain null values
- `fill_missing` — Fill nulls using `mean`, `median`, `mode`, `constant`, `ffill`, or `bfill`
- `rename_column` — Rename a single column
- `filter_rows` — Filter rows using:
  - `==`
  - `!=`
  - `>`
  - `<`
  - `>=`
  - `<=`
  - `contains`
  - `startswith`
  - `endswith`
  - `isnull`
  - `notnull`
- `detect_outliers` — Detect, and optionally remove, outliers via `IQR` or `Z-score`

### Type Conversion and Encoding

- `convert_column_type` — Cast a column to `int`, `float`, `str`, `bool`, or `datetime`
- `encode_categorical` — Label encoding or one-hot (dummy) encoding

### Normalisation and Transforms

- `normalize_column` — Min-max (`0–1`) or Z-score (`mean = 0`, `std = 1`) scaling
- `apply_log_transform` — Apply `log` or `log1p` to a numeric column

### Feature Engineering

- `feature_engineering_date` — Extract `year`, `month`, `day`, `hour`, `minute`, `dayofweek`, `quarter`, `weekofyear`, and `dayofyear` from a datetime column
- `bin_column` — Bin a numeric column using equal-width or quantile strategy

### Sorting and Export

- `sort_data` — Sort dataset by a column in ascending or descending order
- `export_dataset` — Export to `CSV`, `Excel`, `Parquet`, or `JSON`
- `reset_dataset` — Reset dataset to its originally loaded state

## Prerequisites

- Docker Desktop, with Docker MCP Toolkit enabled
- Docker MCP CLI plugin, with the `docker mcp` command available
- A local data directory to mount into the container

## Installation

### Step 1 — Create Your Data Directory

```bash
mkdir -p ~/mcp-data
# Copy datasets you want to work with into this directory
cp mydata.csv ~/mcp-data/
```

### Step 2 — Build the Docker Image

Build the MCP server image from the project directory.

```bash
cd /path/to/data-preprocessor
docker build -t data-preprocessor-mcp-server .
```

### Step 3 — Create the Custom Catalog

Create your custom Docker MCP catalog file.

```bash
mkdir -p ~/.docker/mcp/catalogs
nano ~/.docker/mcp/catalogs/custom.yaml
```

Paste the following, replacing `YOUR_USERNAME` with your macOS username:

```yaml
version: 2
name: custom
displayName: Custom MCP Servers
registry:
  data-preprocessor:
    description: "ML-ready data preprocessing: cleaning, encoding, normalisation, feature engineering"
    title: "Data Preprocessor"
    type: server
    dateAdded: "2026-03-06T00:00:00Z"
    image: data-preprocessor-mcp-server:latest
    ref: ""
    readme: ""
    toolsUrl: ""
    source: ""
    upstream: ""
    icon: ""
    volumes:
      - source: "/Users/YOUR_USERNAME/mcp-data"
        target: /data
    tools:
      - name: load_dataset
      - name: get_info
      - name: preview_data
      - name: get_statistics
      - name: get_missing_summary
      - name: drop_columns
      - name: select_columns
      - name: drop_duplicates
      - name: drop_rows_with_missing
      - name: fill_missing
      - name: rename_column
      - name: filter_rows
      - name: encode_categorical
      - name: normalize_column
      - name: convert_column_type
      - name: feature_engineering_date
      - name: compute_correlation
      - name: detect_outliers
      - name: apply_log_transform
      - name: bin_column
      - name: sort_data
      - name: export_dataset
      - name: reset_dataset
    metadata:
      category: automation
      tags:
        - data-preprocessing
        - machine-learning
        - pandas
        - scikit-learn
        - csv
      license: MIT
      owner: local
```

### Step 4 — Update the Registry

Open your local Docker MCP registry file:

```bash
nano ~/.docker/mcp/registry.yaml
```

Add the following under the existing `registry:` key:

```yaml
data-preprocessor:
  ref: ""
```

### Step 5 — Verify

Check that the MCP server is visible:

```bash
docker mcp server list
```

## Usage Examples

You can send requests like:

```text
"Load the file sales.csv from my data directory"
```

Uses:

```text
load_dataset(filename="sales.csv")
```

```text
"Show me the first 10 rows"
```

Uses:

```text
preview_data(n_rows="10", position="head")
```

```text
"Summarise missing values in the dataset"
```

Uses:

```text
get_missing_summary()
```

```text
"Fill missing values in the Age column using the median"
```

Uses:

```text
fill_missing(column="Age", strategy="median")
```

```text
"Drop the columns ID and Timestamp"
```

Uses:

```text
drop_columns(columns="ID, Timestamp")
```

```text
"Remove duplicate rows"
```

Uses:

```text
drop_duplicates()
```

```text
"Encode the Category column using label encoding"
```

Uses:

```text
encode_categorical(column="Category", method="label")
```

```text
"Normalise the Price column using min-max scaling"
```

Uses:

```text
normalize_column(column="Price", method="minmax")
```

```text
"Extract year, month and day from the OrderDate column"
```

Uses:

```text
feature_engineering_date(column="OrderDate", features="year,month,day")
```

```text
"Show the correlation matrix"
```

Uses:

```text
compute_correlation()
```

```text
"Detect outliers in the Revenue column using IQR"
```

Uses:

```text
detect_outliers(column="Revenue", method="iqr")
```

```text
"Remove outliers from the Revenue column"
```

Uses:

```text
detect_outliers(column="Revenue", method="iqr", remove="true")
```

```text
"Apply log1p transform to the Salary column"
```

Uses:

```text
apply_log_transform(column="Salary", method="log1p")
```

```text
"Bin the Age column into 4 equal-width groups"
```

Uses:

```text
bin_column(column="Age", bins="4", strategy="equal_width")
```

```text
"Export the cleaned dataset as cleaned_sales.csv"
```

Uses:

```text
export_dataset(filename="cleaned_sales.csv", file_type="csv")
```

```text
"Reset to the original dataset"
```

Uses:

```text
reset_dataset()
```

## Architecture

```text
AI Client → MCP Gateway → Data Preprocessor Container → /data volume
                                     ↕
                            ~/mcp-data (your local files)
```

All data stays local. No internet access is required. No API keys are needed.

## Development

### Local Testing (Without Docker)

```bash
export DATA_DIR="$(pwd)/test-data"
export MAX_PREVIEW_ROWS=20
mkdir -p test-data
python data_preprocessor_server.py
```

### Rebuild After Changes

```bash
docker build -t data-preprocessor-mcp-server .
```

### Adding New Tools

1. Add a function with the `@mcp.tool()` decorator
2. Use a single-line docstring only, because multi-line docstrings can cause gateway panic
3. Default all parameters to `""`, never `None`
4. Return a formatted string
5. Add the tool name to your `custom.yaml` catalog entry
6. Rebuild the Docker image

## License

MIT License
