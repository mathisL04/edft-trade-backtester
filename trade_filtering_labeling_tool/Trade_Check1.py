import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import snowflake.connector


# -----------------------------------------------------------------------------

# Date Request
selected_date = input("Selected Trade Date (Format: DDMMMYY): ").upper()

try:
    datetime.strptime(selected_date, "%d%b%y")
except ValueError:
    print("Invalid date format. Please use format like '10JAN24'.")
    exit()


# === Load Power Trade File ===
file_path_power = rf'C:\ZAINET_CLIENT\ODBS\MDR_ELEC_TRADES_{selected_date}.csv'
# === Load GAS Trade File ===
file_path_gas = rf'C:\ZAINET_CLIENT\ODBS\MDR_GAS_TRADES_{selected_date}.csv'




# Check if files exist
if not os.path.exists(file_path_power):
    print(f"⚠️ Power trade file not found for date {selected_date}: {file_path_power}")
    exit()

if not os.path.exists(file_path_gas):
    print(f"⚠️ Gas trade file not found for date {selected_date}: {file_path_gas}")
    exit()

# Ignore weekends
filename_date_str = selected_date


try:
    file_date = datetime.strptime(filename_date_str, "%d%b%y")
except ValueError:
    print("Could not parse the file date correctly.")
    exit()

if file_date.weekday() in [5, 6]:
    print("No data for this date (weekend).")
    exit()


# Creating time variable for Query 
selected_datetime = datetime.strptime(selected_date, "%d%b%y")
today_datetime = datetime.today()

# Compute how many days ago that date is from today
delta_days = (today_datetime - selected_datetime).days

# Sanity check
if delta_days < 0:
    print("Selected date is in the future. Cannot proceed.")
    exit()


# --------------------------------------------------------------------------------
# SQL fetching part 

# Connect to my snowflake account to access data set 
conn = snowflake.connector.connect(
    user='mathis.laurent@edfenergy.com',
    account='YSQZARO-WQ16105',
    warehouse='WH_PERSONA_WMS_PRD',
    database='FLK_DUB_DB_DATALAKE_PRD',
    schema='STAGING_WMS_EDFT',
    authenticator='externalbrowser'
)


# Query to fetch and filter the EDFT data using SQL
query_power = f"""
SELECT 
    PRODUCT, LOAD_SHAPE, PRICE
FROM POWER
WHERE PUBLICATION_DATE = CURRENT_DATE() - {delta_days}
  AND MARKET = 'UNITED KINGDOM'
  AND PRICE_TYPE = 'MID';
"""    


cursor = conn.cursor()
cursor.execute(query_power)

# Create DataFrame from Snowflake results
df_edft = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])
df_edft.columns = ['ProductLabel', 'LoadShape', 'EDFT_Price']
df_edft['EDFT_Price'] = df_edft['EDFT_Price'].astype(float)

# Show the first rows
#print("\n=== EDFT Price Data ===")
#print(df_edft.head(10))


# === EDFT GAS PRICE FETCH (commented out for now) ===
query_gas = f"""
SELECT 
    PUBLICATION_DATE, PRODUCT, PRICE
FROM GAS
WHERE PUBLICATION_DATE = CURRENT_DATE() - {delta_days}
"""


cursor = conn.cursor()
cursor.execute(query_gas)

df_edft_gas = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])
df_edft_gas.columns = ['PublicationDate', 'ProductLabel', 'EDFT_Price']
df_edft_gas = df_edft_gas.drop(columns=['PublicationDate'])
df_edft_gas['EDFT_Price'] = df_edft_gas['EDFT_Price'].astype(float)


#print("\n=== EDFT GAS Price Data ===")
#print(df_edft_gas.head())


cursor.close()
conn.close()


# --------------------------------------------------------------------------------
# Filtering & cleaning Trade data set 
# Load the Excel file
df_power = pd.read_csv(file_path_power)
df_gas = pd.read_csv(file_path_gas)

# Filter relevant columns
df_power = df_power[['TNUM', 'TRADE_DATE', 'BUY_SELL',
                     'DELIVERY START', 'DELIVERY END',
                     'DEL TIMESTART', 'DEL TIMEEND',
                     'VOLUME', 'HEDGE_VALUE', 'FORWARD CURVE VALUATION']]

df_gas = df_gas[['TNUM', 'TRADE_DATE', 'BUY_SELL',
                     'DELIVERY START', 'DELIVERY END',
                     'DEL TIMESTART', 'DEL TIMEEND',
                     'VOLUME', 'HEDGE_VALUE', 'FORWARD CURVE VALUATION']]


# Rename columns for clarity
df_power.columns = ['TradeID', 'TradeDate', 'TradeType',
                    'StartDate', 'EndDate', 'StartTime', 'EndTime',
                    'Volume', 'HedgePrice', 'ForwardValuation']

df_gas.columns = ['TradeID', 'TradeDate', 'TradeType', 
                  'StartDate', 'EndDate', 'StartTime', 'EndTime', 
                  'Volume', 'HedgePrice', 'ForwardValuation']


# --------------------------------------------------------------------------------
# Check for invalid data & prompt trades (removal)

# === Flag invalid trades ===
df_power['Invalid_ZeroVolume'] = df_power['Volume'] == 0
df_power['Invalid_ZeroHedge'] = df_power['HedgePrice'] == 0
df_power['InvalidTrade'] = df_power['Invalid_ZeroVolume'] | df_power['Invalid_ZeroHedge']

# Store TradeIDs of invalid trades for tracking
invalid_trade_ids = df_power[df_power['InvalidTrade']]['TradeID'].tolist()


# Filter out zero-volume or zero-value trades
df_power = df_power[df_power['Volume'] != 0].copy()
df_power = df_power[df_power['HedgePrice'] != 0]

df_gas = df_gas[df_gas['Volume'] != 0]
df_gas = df_gas[df_gas['HedgePrice'] != 0]


# Convert dates to datetime for Power
df_power['TradeDate'] = pd.to_datetime(df_power['TradeDate'])
df_power['StartDate'] = pd.to_datetime(df_power['StartDate'])
df_power['EndDate'] = pd.to_datetime(df_power['EndDate'])
# Convert dates to datetime for Gas
df_gas['TradeDate'] = pd.to_datetime(df_gas['TradeDate'])
df_gas['StartDate'] = pd.to_datetime(df_gas['StartDate'])
df_gas['EndDate'] = pd.to_datetime(df_gas['EndDate'])

# --------------------------------------------------------------------------------
# Calculation and convertion in correct format

# Calculate MWh prices
df_power['TradedPrice'] = df_power['HedgePrice'] / df_power['Volume'] # Traded price that will be compared to internal price and later to EDFT price
df_power['InternalPrice_per_MWh'] = df_power['ForwardValuation'] / df_power['Volume'] # Forward valuation price 

df_gas['TradedPrice'] = df_gas['HedgePrice'] / df_gas['Volume']
df_gas['InternalPrice_per_MWh'] = df_gas['ForwardValuation'] / df_gas['Volume']


# Check if the Volume is equal to the Forward Curve valuation
df_power['Flagged_VolEqVal'] = df_power['Volume'] == df_power['ForwardValuation']
df_gas['Flagged_VolEqVal'] = df_gas['Volume'] == df_gas['ForwardValuation']
#print(f"Row count: {len(df_power)}")


# Flaging and removing the prompt trades (Volume == Forward Curve valuation && Day to delivery << 2 days)
df_power['Flagged'] = df_power['Flagged_VolEqVal']  # Assuming 'Flagged_VolEqVal' is the intended condition
df_power['DaysToDelivery'] = (df_power['EndDate'] - df_power['TradeDate']).dt.total_seconds() / (3600 * 24) # Compute period to delivery time
df_power['PromptTrade'] = (df_power['Flagged']) & (df_power['DaysToDelivery'] < 2)   # Label Prompt trades among the flagged ones
flagged_non_prompt_trades = df_power[(df_power['Flagged']) & (~df_power['PromptTrade'])] #  Filter to keep only flagged trades that are NOT prompt

# Remove flagged trades with DaysToDelivery < 2
df_power = df_power[~((df_power['Flagged']) & (df_power['DaysToDelivery'] < 2))]


# Count after removing prompt trades but keeping future trades 
#print(f"Row count: {len(df_power)}")


# --------------------------------------------------------------------------------

# Calculate price difference metrics 
df_power['PriceDiff'] = df_power['TradedPrice'] - df_power['InternalPrice_per_MWh'] 
df_power['AbsPriceDiff'] = df_power['PriceDiff'].abs()
df_power['WeightedImpact_£'] = df_power['PriceDiff'] * df_power['Volume']

df_gas['PriceDiff'] = df_gas['TradedPrice'] - df_gas['InternalPrice_per_MWh']
df_gas['AbsPriceDiff'] = df_gas['PriceDiff'].abs()
df_gas['WeightedImpact_£'] = df_gas['PriceDiff'] * df_gas['Volume']

# --------------------------------------------------------------------------------


# === Optional flagging (metrics to be changed) ===
# === MODIFIABLE PARAMETER ===
df_power['Warning'] = (
    (df_power['AbsPriceDiff'] > 3) &
    (df_power['Volume'] > 10000)
)

df_gas['Warning'] = (
    (df_gas['AbsPriceDiff'] > 3) &
    (df_gas['Volume'] > 10000)
)

# Mismatch type
df_power['MismatchType'] = df_power['PriceDiff'].apply(lambda x: 'Overpaying' if x > 0 else 'Undervaluing')
df_gas['MismatchType'] = df_gas['PriceDiff'].apply(lambda x: 'Overpaying' if x > 0 else 'Undervaluing')

# Time horizon: days between trade and delivery 
df_power['DaysToDelivery'] = (df_power['StartDate'] - df_power['TradeDate']).dt.days

# --------------------------------------------------------------------------------

# === Labelling Part ===

# Matching work to label trade period of trade log data to the EDFT data set 
def generate_edft_labels(row):
    start = row['StartDate']
    end = row['EndDate']

    # Year, month, week
    start_year, start_month, start_week = start.year, start.month, start.isocalendar().week
    end_year, end_month, end_week = end.year, end.month, end.isocalendar().week

    # Quarter function (labeling trades by quarters)
    def get_quarter_label(year, month):
        if month <= 3:
            return f"{year}Q1"
        elif month <= 6:
            return f"{year}Q2"
        elif month <= 9:
            return f"{year}Q3"
        else:
            return f"{year}Q4"


    # Label generation
    start_quarter = get_quarter_label(start_year, start_month)
    end_quarter = get_quarter_label(end_year, end_month)
    start_month_label = f"{start_year}{start_month:02d}"
    end_month_label = f"{end_year}{end_month:02d}"
    start_week_label = f"{start_year}W{start_week:02d}"
    end_week_label = f"{end_year}W{end_week:02d}"


    # Single or multi-period flags
    is_single_quarter = start_quarter == end_quarter
    is_single_month = start_month_label == end_month_label
    is_single_week = start_week_label == end_week_label
   

    return pd.Series({
        'QuarterLabel': start_quarter if is_single_quarter else f"{start_quarter}–{end_quarter}",
        'MonthLabel': start_month_label if is_single_month else f"{start_month_label}–{end_month_label}",
        'WeekLabel': start_week_label if is_single_week else f"{start_week_label}–{end_week_label}",
        'LabelType': (
            'SingleQuarter' if is_single_quarter else
            'SingleMonth' if is_single_month else
            'MultiPeriod'
        )
    })



# Apply the EDFT label function
df_power = df_power.join(df_power.apply(generate_edft_labels, axis=1))


# --------------------------------------------------------------------------------
# === Load Shape Labellinbg ===
# === Infer Load Shape ===
def infer_load_shape(row):
    try:
        start = datetime.strptime(row['StartTime'], '%H:%M').time()
        end = datetime.strptime(row['EndTime'], '%H:%M').time()
    except:
        return 'Unknown'

    if start == end and start == datetime.strptime('23:00', '%H:%M').time():
        return 'Baseload'  # Baseload: 23:00 to 23:00  

    start_min = start.hour * 60 + start.minute
    end_min = end.hour * 60 + end.minute

    # Load Shape labelling condition 

    if start_min == 0 and end_min >= 1439:
        return 'Baseload'
    elif 420 <= start_min <= 600 and 1020 <= end_min <= 1140:
        return 'Peakload'
    else:
        return 'Offpeak'

df_power['LoadShape'] = df_power.apply(infer_load_shape, axis=1)


# Detect full season

def detect_full_season(start_date, end_date):
    # Full WINTER: Oct 1 (Y) to Mar 31 (Y+1)
    if (start_date.month == 10 and start_date.day == 1 and 
        end_date.month == 3 and end_date.day == 31 and 
        end_date.year == start_date.year + 1):
        return f"{start_date.year}WIN"

    # Full SUMMER: Apr 1 to Sep 30 (same year)
    elif (start_date.month == 4 and start_date.day == 1 and 
          end_date.month == 9 and end_date.day == 30 and 
          start_date.year == end_date.year):
        return f"{start_date.year}SUM"

    return None



# === Create Product Label ===
def get_product_label(row):
    # Season label
    season_label = detect_full_season(row['StartDate'], row['EndDate'])
    if season_label:
        return season_label

    # Then check other granular labels
    if row['StartDate'] == row['EndDate']:
        return row['StartDate'].strftime('%Y%m%d')  # One-day trade
    elif row['WeekLabel'] and '–' not in row['WeekLabel']:
        return row['WeekLabel']
    elif row['MonthLabel'] and '–' not in row['MonthLabel']:
        return row['MonthLabel']
    elif row['QuarterLabel'] and '–' not in row['QuarterLabel']:
        return row['QuarterLabel']
    else:
        return 'MULTI' # case where it hasnt been labelled anything else


df_power['ProductLabel'] = df_power.apply(get_product_label, axis=1)


multi_count = (df_power['ProductLabel'] == 'MULTI').sum()
print(f'Number of trades labeled as MULTI: {multi_count}')


# Labelling and Shape for Gas:
df_gas['DaysToDelivery'] = (df_gas['StartDate'] - df_gas['TradeDate']).dt.days
df_gas = df_gas.join(df_gas.apply(generate_edft_labels, axis=1))
df_gas['LoadShape'] = df_gas.apply(infer_load_shape, axis=1)
df_gas['ProductLabel'] = df_gas.apply(get_product_label, axis=1)


# ------------------------------------------------------------------------------
# Merge EDFT pricing data to the labeled trade log
df_power = df_power.merge(df_edft, on=['ProductLabel', 'LoadShape'], how='left')
df_gas = df_gas.merge(df_edft_gas, on=['ProductLabel'], how='left')

# --------------------------------------------------------------------------------
# === Main operation (comparing Traded price with EDFT) ===

df_power['EDFT_Diff'] = df_power['TradedPrice'] - df_power['EDFT_Price']
df_power['Abs_EDFT_Diff'] = df_power['EDFT_Diff'].abs() # Take the absolute value (if needed), for PnL we will not use it
df_power['EDFT_Flag'] = df_power['Abs_EDFT_Diff'] > 2  # Set threshold as needed (fixed for now)

df_gas['EDFT_Diff'] = df_gas['TradedPrice'] - df_gas['EDFT_Price']
df_gas['Abs_EDFT_Diff'] = df_gas['EDFT_Diff'].abs()
df_gas['EDFT_Flag'] = df_gas['Abs_EDFT_Diff'] > 2



print("=== POWER vs GAS — Trade Summary Statistics ===")
print(f"Total POWER trades: {len(df_power):>6}  |  Flagged: {df_power['Warning'].sum():>4}  |  % Flagged: {(df_power['Warning'].mean() * 100):.2f}%")
print(f"Total GAS   trades: {len(df_gas):>6}  |  Flagged: {df_gas['Warning'].sum():>4}  |  % Flagged: {(df_gas['Warning'].mean() * 100):.2f}%\n")

# -------------------------------------------------------------------------
# Compute PnL only where EDFT_Price is not missing and types are compatible

df_power['PnL_EDFT'] = df_power.apply(
    lambda row: float(row['EDFT_Diff']) * float(row['Volume'])
    if pd.notnull(row['EDFT_Diff']) and pd.notnull(row['Volume'])
    else None,
    axis=1
)

df_gas['PnL_EDFT'] = df_gas.apply(
    lambda row: float(row['EDFT_Diff']) * float(row['Volume'])
    if pd.notnull(row['EDFT_Diff']) and pd.notnull(row['Volume'])
    else None,
    axis=1
)


# -------------------------------------------------------------------------
# === PnL visualisation: ===

# Filter significant PnL trades 
df_significant_pnl = df_power[df_power['PnL_EDFT'].abs() > 10000].sort_values(by='TradeDate')
df_significant_gas_pnl = df_gas[df_gas['PnL_EDFT'].abs() > 10000].sort_values(by='TradeDate')

power_over_10k = (df_significant_pnl['PnL_EDFT'].abs() > 10000).sum()
power_over_100k = (df_significant_pnl['PnL_EDFT'].abs() > 100000).sum()

gas_over_10k = (df_significant_gas_pnl['PnL_EDFT'].abs() > 10000).sum()
gas_over_100k = (df_significant_gas_pnl['PnL_EDFT'].abs() > 100000).sum()

power_profit_count = (df_significant_pnl['PnL_EDFT'] > 0).sum()
power_loss_count = (df_significant_pnl['PnL_EDFT'] < 0).sum()

gas_profit_count = (df_significant_gas_pnl['PnL_EDFT'] > 0).sum()
gas_loss_count = (df_significant_gas_pnl['PnL_EDFT'] < 0).sum()


# Index for x-axis
power_trade_numbers = range(len(df_significant_pnl))
gas_trade_numbers = range(len(df_significant_gas_pnl))

# Create subplots (2 rows, 1 column)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=False)

# ---- POWER PnL ----
ax1.plot(power_trade_numbers, df_significant_pnl['PnL_EDFT'], linestyle='-', marker='o', color='blue', label='Power PnL')

for i, row in enumerate(df_significant_pnl.itertuples()):
    color = 'green' if row.PnL_EDFT >= 0 else 'red'
    ax1.scatter(i, row.PnL_EDFT, color=color, zorder=3)
    ax1.text(i, row.PnL_EDFT + 500, f"{row.PnL_EDFT:,.0f}", ha='center', va='bottom', fontsize=8, rotation=90)

ax1.set_title('Power Trades PnL > ±£10,000')
ax1.set_ylabel('PnL (£)')
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_xticks(power_trade_numbers)
ax1.set_xticklabels(df_significant_pnl['TradeID'], rotation=45, ha='right')
ax1.legend([f"Total > £10k: {power_over_10k}\n> £100k: {power_over_100k}"], loc='upper left')
ax1.legend([f"Total > £10k: {power_over_10k}\n> £100k: {power_over_100k}\nProfits: {power_profit_count} | Losses: {power_loss_count}"], loc='upper left')

# ---- GAS PnL ----
ax2.plot(gas_trade_numbers, df_significant_gas_pnl['PnL_EDFT'], linestyle='-', marker='o', color='orange', label='Gas PnL')


for i, row in enumerate(df_significant_gas_pnl.itertuples()):
    color = 'green' if row.PnL_EDFT >= 0 else 'red'
    ax2.scatter(i, row.PnL_EDFT, color=color, zorder=3)
    ax2.text(i, row.PnL_EDFT + 500, f"{row.PnL_EDFT:,.0f}", ha='center', va='bottom', fontsize=8, rotation=90)

ax2.set_title('Gas Trades PnL > ±£10,000')
ax2.set_xlabel('Trade ID (chronologically ordered)')
ax2.set_ylabel('PnL (£)')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xticks(gas_trade_numbers)
ax2.set_xticklabels(df_significant_gas_pnl['TradeID'], rotation=45, ha='right')
ax2.legend([f"Total > £10k: {gas_over_10k}\n> £100k: {gas_over_100k}"], loc='upper left')
ax2.legend([f"Total > £10k: {gas_over_10k}\n> £100k: {gas_over_100k}\nProfits: {gas_profit_count} | Losses: {gas_loss_count}"], loc='upper left')

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# === Summary statistics for POWER ===
total_trades = len(df_power)
num_flagged = df_power['Warning'].sum()
flagged_ratio = (num_flagged / total_trades) * 100

#print(f"Total trades: {total_trades}")
#print(f"Flagged trades: {num_flagged}")
#print(f"Percentage flagged: {flagged_ratio:.2f}%")

# === Suspicious rows: Volume == Forward Valuation ===
suspicious_rows = df_power[df_power['Volume'] == df_power['ForwardValuation']]
num_suspicious = len(suspicious_rows)
#print(f"Number of trades where Volume == Forward Valuation: {num_suspicious}")



# === Summary statistics for GAS ===
total_gas_trades = len(df_gas)
num_gas_flagged = df_gas['Warning'].sum()
gas_flagged_ratio = (num_gas_flagged / total_gas_trades) * 100



# === Suspicious gas rows: Volume = Forward Valuation ===
suspicious_gas_rows = df_gas[df_gas['Volume'] == df_gas['ForwardValuation']]
num_suspicious_gas = len(suspicious_gas_rows)
#print(f"Number of GAS trades where Volume == Forward Valuation: {num_suspicious_gas}")


# === Max delay trade ===
if not suspicious_rows.empty:
    max_delay_trade = suspicious_rows[suspicious_rows['DaysToDelivery'] == suspicious_rows['DaysToDelivery'].max()]


print("=== PnL Threshold Check (>|£10,000|) ===")
print(f"POWER trades with PnL > ±£10k: {(df_power['PnL_EDFT'].abs() > 10000).sum()}")
print(f"GAS   trades with PnL > ±£10k: {(df_gas['PnL_EDFT'].abs() > 10000).sum()}\n")

print("=== ⚠️ Data Integrity Issues ===")
print(f"Zero-volume GAS trades: {(df_gas['Volume'] == 0).sum()}")
print(f"Zero hedge value GAS trades: {(df_gas['TradedPrice'] == 0).sum()}")
print(f"Volume == ForwardValuation (POWER): {(df_power['Volume'] == df_power['ForwardValuation']).sum()}")
print(f"Volume == ForwardValuation (GAS): {(df_gas['Volume'] == df_gas['ForwardValuation']).sum()}\n")


# == Key statistics and summary ==


def generate_trade_summary_clean(df, name="DATASET"):
    return {
        "Total Trades": len(df),
        "Volume = ForwardValuation": (df['Volume'] == df['ForwardValuation']).sum(),
        "Prompt Trades (<2d to delivery)": ((df['Volume'] == df['ForwardValuation']) & (df['DaysToDelivery'] < 2)).sum(),
        "Abs(Traded - Internal) > £2": (df['AbsPriceDiff'] > 2).sum(),
        "Volume > £10k": (df['Volume'] > 10000).sum(),
        "PnL (vs EDFT) > £10k": (df['PnL_EDFT'].abs() > 10000).sum(),
        "PnL (vs EDFT) > £100k": (df['PnL_EDFT'].abs() > 100000).sum()
    }

power_stats = generate_trade_summary_clean(df_power, "POWER")
gas_stats = generate_trade_summary_clean(df_gas, "GAS")

def print_summary_vertical(title, stats, icon=""):
    print(f"\n{icon} {title}")
    print("-" * (len(title) + 2 + len(icon)))
    for key, value in stats.items():
        print(f"{key:<40} : {value:,}")

# Final display
print("\n📊 === Summary of Trade Quality ===")
print_summary_vertical("POWER", power_stats, icon="⚡")
print_summary_vertical("GAS", gas_stats, icon="🔥")
