import pandas as pd
import requests
import io

# 1. Official Government Source for Naples CAF (Latest 2026)
URL = "https://static-www.comune.napoli.it/wp-content/uploads/2025/05/ELENCO_SPORTELLI_CAF_AGG.TO_AL_14.05.25.25"

def get_napoli_caf_data():
    try:
        response = requests.get(URL)
        # Using ODS/Excel engine since government files are often .ods
        df = pd.read_excel(io.BytesIO(response.content))
        
        # 2. Cleanup: Rename columns to match your Hub & Spoke DB
        # Mapping based on official "Quartiere, Indirizzo, CAF, Orari" structure
        df.columns = ['district', 'address', 'name', 'days_1', 'hours_1', 'days_2', 'hours_2', 'phone', 'email']
        
        # 3. Filter: Remove rows that are empty or are "Caffè" (the noise)
        df = df[df['name'].str.contains('CAF|CAAF|PATRONATO|CISL|CGIL|UIL|ACLI', case=False, na=False)]
        
        # 4. JSON Format for your FastAPI Backend
        clean_data = df.to_dict(orient='records')
        
        print(f"Successfully extracted {len(clean_data)} verified offices.")
        return clean_data

    except Exception as e:
        return f"Error: {e}. Check if URL is still public."

# Run it
data = get_napoli_caf_data()
if isinstance(data, list):
    print(data[0]) # Show first record as proof