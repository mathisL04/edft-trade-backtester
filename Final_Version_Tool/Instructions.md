# ⚡📈 EDFT Trade Backtesting & Validation Tool

## Overview

This project is a robust, Python-based pipeline developed to **backtest**, **validate**, and **analyze EDF Energy's Power and Gas trades**. It integrates with internal systems like **FIS Aligne** (for trade logs) and **Snowflake** (for internal valuations), compares traded prices with **EDFT benchmark curves**, and flags anomalies or data integrity issues.

Visualizations and data exports are directly linked to **Tableau dashboards**, allowing seamless, dynamic updates of daily and historical trade performance.

> **Note**: The pipeline does **not** currently fetch data automatically from FIS Aligne. Users must manually download the daily trade logs from Aligne and store them locally (e.g., `R:\ZAINET_CLIENT\ODBS\`) before execution. Automation of this ingestion step is under consideration.

---

## 🚀 Key Features

### 📅 Date-Driven Trade Import
- Users specify a **start** and **end** date (e.g., `01JUL25` to `17JUL25`). The tool automatically looks for corresponding **FIS Aligne CSV trade logs** stored locally (for Power and Gas) using the `report_date = trade_date + 1` logic.
- Validates file existence, skips weekends, and supports multi-day batch processing.

### 🧾 Trade Filtering & Labeling
- **Filters** invalid trades (e.g., zero volume or hedge value).
- **Flags prompt trades** (Volume = Valuation & delivery < 2 days).
- **Removes mirrored trades** (positive/negative volume duplicates).
- **Labels** trades by:
  - Season (`WIN`, `SUM`)
  - Quarter (`Q1`, `Q2`, ...`)
  - Month and Week
  - Load Shape (`Baseload`, `Peakload`, `Offpeak`)
  - Trade granularity (`SinglePeriod`, `MultiPeriod`, `MULTI`)

### ❄️ Snowflake Integration
- Securely connects to **EDF Energy's Snowflake environment** using the `snowflake-connector-python` library and `externalbrowser` authentication.
- Queries the `POWER` and `GAS` price tables for EDFT reference prices based on `TradeDate_EDFT_Label`.
- Merges EDFT prices with trades using `ProductLabel` and `LoadShape`, with **fallback logic** to baseload if needed.

### 📊 Trade Metrics
- Computes:
  - Traded vs. Internal Price
  - Traded vs. EDFT Benchmark Price
  - Absolute & Weighted price differences (PnL indicators)

### 📈 Trade Flagging System
A separate script applies three anomaly detection techniques:

#### 🔹 Z-Score Flagging
- Detects extreme deviations from the mean:
  \[ Z = \frac{x - \mu}{\sigma} \]
  - Flags trades where `|Z| > 2.5`.

#### 🔹 Robust Z-Score (MAD)
- More resistant to outliers:
  \[ \text{Robust Z} = \frac{x - \text{median}}{1.4826 \cdot \text{MAD}} \]
  - Flags trades where `|Robust Z| > 2.5`.

#### 🔹 Isolation Forest (ML-Based)
- Detects anomalous price behavior using unsupervised ML.
- Trained per day on `EDFT_Diff` values.

### ✅ Combined Flagging
- Trades receive a `Flag_Count` (0–3), based on how many methods identify them as anomalies.
- Final dataset includes:
  - Trade ID, Date, Traded Price, EDFT Price, EDFT Diff
  - Z-Score, Robust Z-Score, Isolation Forest prediction
  - ProductLabel, LoadShape, Volume

### 📤 Output Pipeline
- Labeled trade CSVs exported to `R:\Project 1\Trades Labelling & Filtering`.
- Flagged trades uploaded to Snowflake table: `DB_WMS_PRD.MDR_SANDBOX.POWER_GAS_TRADE_FLAG`.
- Tableau dashboards are connected directly to this table.

---

## 🧩 Workflow Summary

```text
[User: Download Daily FIS Aligne Files]
        ↓
[Run Script 1: Labelling & Filtering Tool]
        ↓
[Fetch EDFT Prices from Snowflake]
        ↓
[Merge, Clean, Compute PnL]
        ↓
[Export Labeled CSVs to R:\Project 1]
        ↓
[Run Script 2: Flagging Tool]
        ↓
[Upload Flagged Trades to Snowflake]
        ↓
[View Trades in Tableau]
```

---

## ⚙️ How to Use

### 🔧 Step 1: Manual Preparation
- Download daily trade files from FIS Aligne.
- Save to: `R:\ZAINET_CLIENT\ODBS\MDR_ELEC_TRADES_<reportdate>.csv` and `MDR_GAS_TRADES_<reportdate>.csv`.

### ▶️ Step 2: Run Labelling & Filtering Script
```bash
python flag_trade_labeller.py
```
- Prompts for:
  - `START DATE (Format: DDMMMYY)`
  - `END DATE (Format: DDMMMYY)`
- Output saved to `R:\Project 1\Trades Labelling & Filtering\output_power_trades_<range>.csv`.

### ▶️ Step 3: Run Flagging Script
```bash
python flag_trade_outliers.py
```
- Reads the labeled CSVs.
- Applies Z-Score, MAD, and Isolation Forest logic.
- Uploads flagged trades to Snowflake.

### 🔑 Snowflake Setup (Modify in Both Scripts)
Replace connection values as needed:
```python
conn = snowflake.connector.connect(
    user='your.email@edfenergy.com',
    account='ACCOUNT_NAME',
    warehouse='WAREHOUSE_NAME',
    database='DB_NAME',
    schema='SCHEMA_NAME',
    authenticator='externalbrowser',
    role='YOUR_ROLE'
)
```
Ensure your user has access to the following tables:
- `FLK_DUB_DB_DATALAKE_PRD.STAGING_WMS_EDFT.POWER`
- `FLK_DUB_DB_DATALAKE_PRD.STAGING_WMS_EDFT.GAS`
- `DB_WMS_PRD.MDR_SANDBOX.POWER_GAS_TRADE_FLAG`

---

## ⚙️ Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- openpyxl
- snowflake-connector-python

Install with:
```bash
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```text
📂 R:\Project 1\
 ├── Trades Labelling & Filtering
 │     └── output_power_trades_<range>.csv
 │     └── output_gas_trades_<range>.csv
 ├── Trades Flagging
 │     └── [Optional combined or Excel output]

📂 R:\ZAINET_CLIENT\ODBS\
 └── MDR_ELEC_TRADES_<reportdate>.csv
 └── MDR_GAS_TRADES_<reportdate>.csv
```

---

## 🧠 Notes & Enhancements

- **FIS Aligne Automation**: Future enhancement aims to automate log fetching (currently manual due to legacy constraints).
- **MULTI Trades**: Ambiguous period trades (`MULTI`) are excluded from Tableau.
- **Snowflake Upload**: Uses `write_pandas()` for fast upload.
- **Flagging Logic**: Modular design enables tuning or method replacement.

---

## 📬 Contact

For questions or contributions:
**Mathis Laurent**  
`mathislaurent04@gmail.com`

---

> “Powerful insights come from clean data.”
