import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """
    Loads and merges the 4 key datasets from the data/ folder.
    Handles different metadata header lengths (Bloomberg inconsistency).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print(f"📂 Looking for data in: {data_dir}")

    # Map: 'Key': ('filename', rows_to_skip)
    files = {
        'BCOM': ('bcom.xlsx', 6),          # Has 6 rows of metadata
        'SPX': ('spx.xlsx', 6),            # Has 6 rows of metadata
        'Treasury10Y': ('treasury_10y.xlsx', 5), # Has 5 rows of metadata
        'IG_Corp': ('corp_bonds.xlsx', 5)  # Has 5 rows of metadata
    }
    
    merged_df = None
    
    for index_name, (filename, skip_count) in files.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Error: Could not find {filepath}")
            continue
            
        try:
            # Read Excel using the specific skip count for this file
            df = pd.read_excel(filepath, skiprows=skip_count)
            
            # Clean up column names (remove hidden spaces)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Check if we successfully grabbed the header
            if 'PX_LAST' in df.columns:
                cols_to_keep = ['Date', 'PX_LAST']
                df = df[cols_to_keep].copy()
                df.columns = ['Date', f'{index_name}_Price']
                
                # Formatting
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df[f'{index_name}_Price'] = pd.to_numeric(df[f'{index_name}_Price'], errors='coerce')
                
                # Drop empty rows and sort
                df = df.dropna(subset=['Date']).sort_values('Date')
                
                # Merge into the main table
                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on='Date', how='outer')
                    
                print(f"   -> Loaded {filename} (Skipped {skip_count} rows)")
            else:
                print(f"⚠️ Warning: 'PX_LAST' column missing in {filename}. Found: {df.columns.tolist()}")

        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")

    # Final Cleanup
    if merged_df is not None:
        merged_df = merged_df.sort_values('Date').reset_index(drop=True)
        print(f"✅ Success! Loaded {len(merged_df)} rows of aligned data.")
        return merged_df
    return None

if __name__ == "__main__":
    load_data()