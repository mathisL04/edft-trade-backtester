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

## Flow of Tool:

[User Input: Trade Date]  
         │  
         ▼  
[Load Trade Files from Local Directory]  
         │  
         ▼  
[Ignore Weekend Dates and Future Dates]  
         │  
         ▼  
[Filter Out Invalid Trades]  
  (Zero volume / Zero hedge / Prompt trades / Past delivery trades)  
         │  
         ▼  
[Fetch EDFT Reference Prices from Snowflake]  
         │  
         ▼  
[Label Each Trade]  
  (Quarter / Month / Week / Load Shape / Product Label)  
         │  
         ▼  
[Match Trade Product + Load Shape with EDFT Prices]  
         │  
         ▼  
[Compare Traded Price vs EDFT Price]  
         │  
         ▼  
[Calculate EDFT_Diff = TradedPrice − EDFT_Price]  
         │  
         ▼  
[Export Cleaned Power & Gas Trade Files as CSV]  
         │  
         ▼  
[Load Cleaned CSV Files for Z-Score Flagging]  
         │  
         ▼  
[Calculate Z-score of EDFT_Diff for Each Trade]  
         │  
         ▼  
[Flag Trades where |Z-score| > 2.5]  
         │  
         ▼  
[Export Excel File with Highlighted Outliers]


## ⚙️ Dependencies

- Python 3.8+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `snowflake-connector-python`

---


