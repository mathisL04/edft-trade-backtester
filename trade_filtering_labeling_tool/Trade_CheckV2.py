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
    selected_datetime = datetime.strptime(selected_date, "%d%b%y")
except ValueError:
    print("Invalid date format. Please use format like '20JUN25'.")
    exit()


# === Load Power Trade File ===
file_path_power = rf'C:\ZAINET_CLIENT\ODBS\MDR_ELEC_TRADES_{selected_date}.csv'
# === Load GAS Trade File ===
file_path_gas = rf'C:\ZAINET_CLIENT\ODBS\MDR_GAS_TRADES_{selected_date}.csv'

# === PRE FILTERING ===

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
delta_days = (today_datetime - selected_datetime).days

# Compute how many days ago that date is from today
edft_reference_date = selected_datetime - pd.Timedelta(days=2)
edft_reference_date_str = edft_reference_date.strftime('%Y-%m-%d')

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
WHERE PUBLICATION_DATE = DATE '{edft_reference_date_str}'
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



query_gas = f"""
SELECT 
    PUBLICATION_DATE, PRODUCT, PRICE
FROM GAS
WHERE PUBLICATION_DATE = DATE '{edft_reference_date_str}'
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
# Load trade file
df_power = pd.read_csv(file_path_power)
df_gas = pd.read_csv(file_path_gas)



# Fix starts here (this replaces a section in your original code)

# --- Time Parsing (Correct placement BEFORE renaming columns)
def format_time_string(val):
    if pd.isnull(val):
        return None
    val_str = str(val).strip()
    try:
        if ':' in val_str:
            parsed = datetime.strptime(val_str, '%H:%M')
            return parsed.strftime('%H:%M')
        val_float = float(val_str)
        hours = int(val_float)
        minutes = int(round((val_float - hours) * 60))
        return f"{hours:02d}:{minutes:02d}"
    except:
        return None

# Apply BEFORE renaming
# Make sure you use the original column names from the CSV

# Power
df_power['StartTime'] = df_power['DEL TIMESTART'].apply(format_time_string)
df_power['EndTime']   = df_power['DEL TIMEEND'].apply(format_time_string)

# Gas
df_gas['StartTime'] = df_gas['DEL TIMESTART'].apply(format_time_string)
df_gas['EndTime']   = df_gas['DEL TIMEEND'].apply(format_time_string)

# Then continue with renaming

# Rename for clarity
df_power = df_power[['TNUM', 'TRADE_DATE', 'BUY_SELL',
                     'DELIVERY START', 'DELIVERY END',
                     'StartTime', 'EndTime',
                     'VOLUME', 'HEDGE_VALUE', 'FORWARD CURVE VALUATION']]

df_power.columns = ['TradeID', 'TradeDate', 'TradeType',
                    'StartDate', 'EndDate', 'StartTime', 'EndTime',
                    'Volume', 'HedgePrice', 'ForwardValuation']

df_gas = df_gas[['TNUM', 'TRADE_DATE', 'BUY_SELL',
                 'DELIVERY START', 'DELIVERY END',
                 'StartTime', 'EndTime',
                 'VOLUME', 'HEDGE_VALUE', 'FORWARD CURVE VALUATION']]

df_gas.columns = ['TradeID', 'TradeDate', 'TradeType',
                  'StartDate', 'EndDate', 'StartTime', 'EndTime',
                  'Volume', 'HedgePrice', 'ForwardValuation']


# === Remove invalid trades ===

# Convert dates to datetime for Power
df_power['TradeDate'] = pd.to_datetime(df_power['TradeDate'])
df_power['StartDate'] = pd.to_datetime(df_power['StartDate'])
df_power['EndDate'] = pd.to_datetime(df_power['EndDate'])

# Convert dates to datetime for Gas
df_gas['TradeDate'] = pd.to_datetime(df_gas['TradeDate'])
df_gas['StartDate'] = pd.to_datetime(df_gas['StartDate'])
df_gas['EndDate'] = pd.to_datetime(df_gas['EndDate'])

# Filter out trades where EndDate is more than 1 day before the selected date
cutoff_date = selected_datetime - pd.Timedelta(days=2)

df_power = df_power[df_power['EndDate'] > cutoff_date].copy()
df_gas = df_gas[df_gas['EndDate'] > cutoff_date].copy()

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

#print("\n Sample of StartTime/EndTime BEFORE LoadShape inference:")
#print(df_power[['StartTime', 'EndTime']].drop_duplicates().head(10))

def infer_load_shape(row):
    try:
        start_time = datetime.strptime(row['StartTime'], '%H:%M').time()
        end_time = datetime.strptime(row['EndTime'], '%H:%M').time()
        start_date = row['StartDate']
        end_date = row['EndDate']
    except Exception:
        return 'UNKNOWN'

    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    duration_days = (end_date - start_date).days

    # === BASELOAD detection ===
    # If trade spans multiple days AND start/end are near midnight (00:00 or 23:00)
    if duration_days >= 1:
        if (start_minutes in [0, 60, 1380]) and (end_minutes in [0, 60, 1380]):
            return 'BASELOAD'

    # Also allow BASELOAD for single-day 00:00–23:59
    if duration_days == 0 and start_minutes == 0 and end_minutes in [1439, 1440]:
        return 'BASELOAD'

    # === PEAKLOAD detection === (daytime 07:00 to 19:00)
    if duration_days == 0 and start_minutes >= 420 and end_minutes <= 1140:
        return 'PEAKLOAD'

    # === Default to OFFPEAK ===
    return 'OFFPEAK'




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
    season_label = detect_full_season(row['StartDate'], row['EndDate'])
    if season_label:
        return season_label

    start_date = row['StartDate']
    end_date = row['EndDate']
    start_time = row['StartTime']
    end_time = row['EndTime']

    if (
        start_time == '00:00' and
        end_time == '00:00' and
        (end_date - start_date).days == 1
    ):
        return start_date.strftime('%Y%m%d')

    if (
        start_time == '05:00' and
        end_time == '05:00' and
        (end_date - start_date).days == 1
    ):
        return start_date.strftime('%Y%m%d')

    if start_date.date() == end_date.date():
        return start_date.strftime('%Y%m%d')

    if row['WeekLabel'] and '–' not in row['WeekLabel']:
        return row['WeekLabel']
    elif row['MonthLabel'] and '–' not in row['MonthLabel']:
        return row['MonthLabel']
    elif row['QuarterLabel'] and '–' not in row['QuarterLabel']:
        return row['QuarterLabel']
    else:
        return 'MULTI'





df_power['ProductLabel'] = df_power.apply(get_product_label, axis=1)


multi_count = (df_power['ProductLabel'] == 'MULTI').sum()
print(f'Number of trades labeled as MULTI: {multi_count}')


# Labelling and Shape for Gas:
df_gas['DaysToDelivery'] = (df_gas['StartDate'] - df_gas['TradeDate']).dt.days
df_gas = df_gas.join(df_gas.apply(generate_edft_labels, axis=1))
df_gas['LoadShape'] = df_gas.apply(infer_load_shape, axis=1)
df_gas['ProductLabel'] = df_gas.apply(get_product_label, axis=1)


# ------------------------------------------------------------------------------
# DIAGNOSTICS
# STEP 1: Normalize input
df_power['ProductLabel'] = df_power['ProductLabel'].astype(str).str.strip()
df_power['LoadShape'] = df_power['LoadShape'].astype(str).str.strip().str.upper()
df_edft['ProductLabel'] = df_edft['ProductLabel'].astype(str).str.strip()
df_edft['LoadShape'] = df_edft['LoadShape'].astype(str).str.strip().str.upper()

# STEP 2: Merge exact match (ProductLabel + LoadShape)
df_power = df_power.merge(
    df_edft.rename(columns={'EDFT_Price': 'EDFT_Price_Matched'}),
    on=['ProductLabel', 'LoadShape'],
    how='left'
)

# STEP 3: Fallback to baseload (for trades where LoadShape ≠ BASELOAD and no price found)
fallback_needed = df_power['EDFT_Price_Matched'].isna() & (df_power['LoadShape'] != 'BASELOAD')

df_power.loc[fallback_needed, 'LoadShape_Fallback'] = 'BASELOAD'
df_power.loc[fallback_needed, 'ProductLabel_Fallback'] = df_power.loc[fallback_needed, 'ProductLabel']

df_edft_baseload = df_edft[df_edft['LoadShape'] == 'BASELOAD'].copy()
df_edft_baseload = df_edft_baseload.rename(columns={
    'ProductLabel': 'ProductLabel_Fallback',
    'EDFT_Price': 'EDFT_Price_Fallback'
})

# STEP 4: Merge fallback baseload price
df_power = df_power.merge(
    df_edft_baseload[['ProductLabel_Fallback', 'EDFT_Price_Fallback']],
    on='ProductLabel_Fallback',
    how='left'
)

# STEP 5: Final column: use exact match if found, otherwise fallback to baseload
df_power['EDFT_Price'] = df_power['EDFT_Price_Matched']
df_power['EDFT_Price'].fillna(df_power['EDFT_Price_Fallback'], inplace=True)

# Cleanup
df_power.drop(columns=[
    'EDFT_Price_Matched',
    'EDFT_Price_Fallback',
    'ProductLabel_Fallback',
    'LoadShape_Fallback'
], inplace=True)



df_gas = df_gas.merge(df_edft_gas, on=['ProductLabel'], how='left')

# --------------------------------------------------------------------------------
# === Main operation (comparing Traded price with EDFT) ===

df_power['EDFT_Diff'] = df_power['TradedPrice'] - df_power['EDFT_Price']
df_power['Abs_EDFT_Diff'] = df_power['EDFT_Diff'].abs() # Take the absolute value (if needed), for PnL we will not use it
df_power['EDFT_Flag'] = df_power['Abs_EDFT_Diff'] > 2  # Set threshold as needed (fixed for now)

df_gas['EDFT_Diff'] = df_gas['TradedPrice'] - df_gas['EDFT_Price']
df_gas['Abs_EDFT_Diff'] = df_gas['EDFT_Diff'].abs()
df_gas['EDFT_Flag'] = df_gas['Abs_EDFT_Diff'] > 2



#print("=== POWER vs GAS — Trade Summary Statistics ===")
#print(f"Total POWER trades: {len(df_power):>6}  |  Flagged: {df_power['Warning'].sum():>4}  |  % Flagged: {(df_power['Warning'].mean() * 100):.2f}%")
#print(f"Total GAS   trades: {len(df_gas):>6}  |  Flagged: {df_gas['Warning'].sum():>4}  |  % Flagged: {(df_gas['Warning'].mean() * 100):.2f}%\n")



# -------------------------------------------------------------------------
# === PnL visualisation: ===




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


'''

print("=== ⚠️ Data Integrity Issues ===")
print(f"Zero-volume GAS trades: {(df_gas['Volume'] == 0).sum()}")
print(f"Zero hedge value GAS trades: {(df_gas['TradedPrice'] == 0).sum()}")
print(f"Zero-volume POWER trades: {(df_power['Volume'] == 0).sum()}")
print(f"Zero hedge value POWER trades: {(df_power['TradedPrice'] == 0).sum()}")
print(f"Volume == ForwardValuation (POWER): {(df_power['Volume'] == df_power['ForwardValuation']).sum()}")
print(f"Volume == ForwardValuation (GAS): {(df_gas['Volume'] == df_gas['ForwardValuation']).sum()}\n")
'''

def print_summary_vertical(title, stats, icon=""):
    print(f"\n{icon} {title}")
    print("-" * (len(title) + 2 + len(icon)))
    for key, value in stats.items():
        formatted_value = f"{value:,.2f}" if "PnL" in key and isinstance(value, (int, float)) else f"{value:,}"
        print(f"{key:<40} : {formatted_value}")

# Final display
print("\n📊 === Summary of Trade Quality ===")
#print_summary_vertical("POWER", power_stats, icon="⚡")
#print_summary_vertical("GAS", gas_stats, icon="🔥")
 # === Final sanity check: Any trades with missing EDFT_Diff? ===

missing_edft_power = df_power['EDFT_Diff'].isna().sum()
missing_edft_gas = df_gas['EDFT_Diff'].isna().sum()

if missing_edft_power > 0:
    print(f"\n⚠️ WARNING: {missing_edft_power} POWER trade(s) have missing EDFT_Diff (likely no EDFT price found).")

if missing_edft_gas > 0:
    print(f"⚠️ WARNING: {missing_edft_gas} GAS trade(s) have missing EDFT_Diff (likely no EDFT price found).")


# -----------------------------------------------------------------------------------------------------------------------
# === File exportation (CSV) ===


export_dir = r'R:\Project 1\Output Trades (Tableau)'
export_columns = [
    'TradeID',
    'TradeDate',
    'TradeType',             
    'TradedPrice',
    'InternalPrice_per_MWh',
    'EDFT_Diff',              
    'AbsPriceDiff',
    'Volume',
    'ProductLabel',
    'LoadShape'
]

power_export_path = os.path.join(export_dir, f'output_power_trades_{selected_date}.csv')
gas_export_path = os.path.join(export_dir, f'output_gas_trades_{selected_date}.csv')

df_power[export_columns].to_csv(power_export_path, index=False)
df_gas[export_columns].to_csv(gas_export_path, index=False)

print(f"Power trade details saved to: {power_export_path}")
print(f"Gas trade details saved to: {gas_export_path}")

# ========================================================================================================

def trade_integrity_summary(df, name="DATASET"):
    total = len(df)

    # Volume = ForwardValuation
    vol_eq_val = (df['Volume'] == df['ForwardValuation']).sum()

    # Prompt trades: flagged trades AND DaysToDelivery < 2
    prompt = ((df['Volume'] == df['ForwardValuation']) & (df['DaysToDelivery'] < 2)).sum()

    # Zero-volume and zero-value trades (counted before filtering)
    zero_volume = (df['Volume'] == 0).sum()
    zero_hedge = (df['HedgePrice'] == 0).sum()

    print(f"\n🔍 Summary of Removed or Flagged Trades — {name}")
    print("----------------------------------------------------")
    print(f"Total trades (before deletion):        {total:,}")
    print(f"Volume == ForwardValuation:            {vol_eq_val:,}")
    print(f"→ Prompt trades (<2d to delivery):     {prompt:,}")
    print(f"Trades with Volume = 0:                {zero_volume:,}")
    print(f"Trades with Hedge Value = 0:           {zero_hedge:,}")


trade_integrity_summary(df_power, name="POWER")
trade_integrity_summary(df_gas, name="GAS")
