# edft-trade-backtester
power-gas-trade-recon:   **Automated pipeline for validating and analyzing EDF Energy power and gas trades using Snowflake and FIS Aligne, with PnL backtesting and trade quality checks.**


# ⚡📈 EDF Trade Backtesting & Validation Tool

This project is a fully automated Python-based pipeline for **backtesting**, **validating**, and **analyzing EDF Energy’s Power and Gas trades**. It connects to internal systems including **FIS Aligne** (trade log source) and **Snowflake** (internal valuation database), compares traded prices to EDFT benchmark curves, and flags unusual pricing patterns or data integrity issues. The tool also visualizes significant PnL outcomes and trade risks.

---

## 🔍 Key Features

- **📅 Date-Driven Execution**  
  Users input a trade date (e.g., `13JUN25`) and the tool dynamically locates the correct CSV trade files for that day. It validates format, filters out weekends, and ensures data availability before proceeding.

- **❄️ Snowflake Integration**  
  Automatically fetches historical EDFT valuation curves for Power and Gas via Snowflake SQL queries. The number of days back is dynamically computed from the trade date.

- **🏷️ Intelligent Labeling**  
  Trade delivery windows are automatically labeled with:
  - Delivery season (`WIN` / `SUM`)
  - Quarter (`Q1`, `Q2`...)
  - Month and week
  - Load shape (`Baseload`, `Peakload`, `Offpeak`)
  - Trade granularity (`SinglePeriod` vs `MULTI`)

- **📉 PnL Calculation and Risk Detection**  
  Calculates traded price, internal valuation, and compares to EDFT reference pricing:
  - Computes absolute and weighted price differences
  - Flags trades with large discrepancies (e.g., >£10,000 PnL or >£2 price delta)
  - Detects suspicious trades with `Volume == Forward Valuation` (often prompt)

- **📊 Visual Analytics**  
  - PnL scatter plots (Power and Gas)
  - Color-coded profit/loss markers
  - Annotated trade IDs and value labels
  - Legends with summary counts:
    - PnL > £10k
    - PnL > £100k
    - Profit / Loss counts

---

## Z-Score Based Outlier Flagging

Once the trade data is matched with EDFT benchmark prices and cleaned, we apply a statistical flagging system using **Z-score analysis**. This method detects trades whose price deviation is significantly different from the average.

### 🔹 Z-Score Formula

We calculate the Z-score for each trade using the formula:


Where:
- `x` = `EDFT_Diff` = `TradedPrice − EDFT_Price`
- `μ` = mean of `EDFT_Diff` values in the dataset
- `σ` = standard deviation of `EDFT_Diff`

---

### 🔹 Flagging Logic

A trade is flagged as an outlier if:


This threshold filters trades with extreme deviations from the market benchmark (EDFT price), relative to the distribution of all trades on that day.

---

### 🔹 Why Z-Score? (vs. Fixed Thresholds)

The Z-score method adapts to the statistical nature of each dataset. Compared to using a fixed value threshold (e.g. `price difference > £2`), it offers:

- **Adaptiveness** – scales with the dataset's internal variability
- **Robustness** – highlights outliers regardless of trade volume or product
- **Consistency** – applicable across different dates, products, and markets

---

### 🔹 Output

Flagged trades are exported to Excel files with:
- Computed Z-scores
- `True/False` flag indicator
- Color-coded rows for rapid review

--- 

## Flow of Tool:
## ⚙️ Processing Flow

```text
[User Input Date]
       ↓
[Load Trade Logs]
       ↓
[Clean & Filter Trades]
       ↓
[Fetch EDFT Prices]
       ↓
[Match, Analyze, Flag]


## ⚙️ Dependencies

- Python 3.8+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `snowflake-connector-python`

---


