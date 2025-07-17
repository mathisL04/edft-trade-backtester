import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font
from statsmodels.robust.scale import mad
from sklearn.ensemble import IsolationForest
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas




conn = snowflake.connector.connect(
    user='mathis.laurent@edfenergy.com',
    account='YSQZARO-WQ16105',
    warehouse='WH_PERSONA_WMS_PRD',
    database='DB_WMS_PRD',
    schema='MDR_SANDBOX',
    authenticator='externalbrowser',
    role='ROL_BUILDER_WMS'
)

conn.cursor().execute("TRUNCATE TABLE DB_WMS_PRD.MDR_SANDBOX.POWER_GAS_TRADE_FLAG")



# --- User input ---
start_date_input = input("START DATE (Format: DDMMMYY): ").upper()
end_date_input = input("END DATE (Format: DDMMMYY): ").upper()

# --- File paths ---
input_base_dir = r'R:\Project 1\Trades Labelling & Filtering'
output_dir = r'R:\Project 1\Trades Flagging'
range_label = f"{start_date_input}_to_{end_date_input}"

files = {
    'POWER': os.path.join(input_base_dir, f'output_power_trades_{range_label}.csv'),
    'GAS': os.path.join(input_base_dir, f'output_gas_trades_{range_label}.csv'),
}

def process_trade_file(input_csv_path, energy_label):
    print(f"\nProcessing {energy_label} data...")

    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f" File not found for {energy_label}: {input_csv_path}")
        return pd.DataFrame()

    if 'EDFT_Diff' not in df.columns or 'TradeID' not in df.columns:
        print(f" Required columns missing in {energy_label} input file: {input_csv_path}")
        return pd.DataFrame()

    # Remove rows with missing EDFT_Diff
    df = df[df['EDFT_Diff'].notna()]
    print(f"  Rows after removing missing EDFT_Diff: {len(df)}")

    # Z-score
    mean_diff = df['EDFT_Diff'].mean()
    std_diff = df['EDFT_Diff'].std(ddof=0)
    df['Zscore'] = (df['EDFT_Diff'] - mean_diff) / std_diff if std_diff != 0 else 0.0
    df['Flag_Zscore'] = df['Zscore'].abs() > 2.5

    # Robust Z-score (MAD)
    median_diff = df['EDFT_Diff'].median()
    mad_diff = mad(df['EDFT_Diff'])
    if mad_diff and not np.isnan(mad_diff):
        robust_z = (df['EDFT_Diff'] - median_diff) / (1.4826 * mad_diff)
        df['Flag_RobustZ'] = robust_z.abs() > 2.5
    else:
        df['Flag_RobustZ'] = False

    # Isolation Forest
    valid_iforest = df['EDFT_Diff'].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid_iforest.empty:
        model = IsolationForest(contamination=0.05, random_state=42)
        preds = model.fit_predict(valid_iforest.values.reshape(-1, 1))
        df['IF_Score'] = pd.Series(index=valid_iforest.index, data=preds)
        df['Flag_IForest'] = df['IF_Score'] == -1
    else:
        df['IF_Score'] = np.nan
        df['Flag_IForest'] = False

    # Total flags
    df['Flag_Count'] = df[['Flag_Zscore', 'Flag_RobustZ', 'Flag_IForest']].sum(axis=1)

    # Add EnergyType
    df['EnergyType'] = energy_label

    # Prepare output columns
    output_cols = ['TradeID', 'EnergyType']
    if 'TradeDate' in df.columns:
        output_cols.append('TradeDate')
    if 'TradedPrice' in df.columns:
        output_cols.append('TradedPrice')
    if 'EDFT_Price' in df.columns:
        output_cols.append('EDFT_Price')
    if 'ProductLabel' in df.columns:
        output_cols.append('ProductLabel')
    if 'LoadShape' in df.columns:
        output_cols.append('LoadShape')
    if 'Volume' in df.columns:
        output_cols.append('Volume')

    output_cols += ['EDFT_Diff', 'Zscore', 'Flag_Zscore', 'Flag_RobustZ', 'Flag_IForest', 'Flag_Count']

    return df[output_cols]

# --- Process and combine ---
df_power = process_trade_file(files['POWER'], 'POWER')
df_gas = process_trade_file(files['GAS'], 'GAS')

df_combined = pd.concat([df_power, df_gas], ignore_index=True)

# Convert TradeDate to datetime for sorting
df_combined['TradeDate'] = pd.to_datetime(df_combined['TradeDate'], errors='coerce').dt.date
df_combined = df_combined.sort_values(by='TradeDate')
float_cols = df_combined.select_dtypes(include=['float']).columns
df_combined[float_cols] = df_combined[float_cols].round(4)


df_combined.columns = [col.upper() for col in df_combined.columns]

success, nchunks, nrows, _ = write_pandas(
    conn,
    df_combined,
    table_name='POWER_GAS_TRADE_FLAG',
    schema='MDR_SANDBOX',
    database='DB_WMS_PRD',
    quote_identifiers=False
)

print(f"\n✅ Uploaded {nrows} rows to Snowflake in {nchunks} chunk(s)")





# Output file path
#combined_output_path = os.path.join(output_dir, f'COMBINED_POWER_GAS_{range_label}_Zscores.xlsx')
#df_combined.to_excel(combined_output_path, index=False)

#print(f"\n Combined POWER and GAS trades saved to:\n{combined_output_path}")
