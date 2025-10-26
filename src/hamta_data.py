import os
import pandas as pd

def hamta_data(ligakod: str, säsong: str, namn: str = None, save_dir: str = "data") -> pd.DataFrame:
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://www.football-data.co.uk/mmz4281/{säsong}/{ligakod}.csv"
    try:
        df = pd.read_csv(url)
        if df.empty:
            return None
        if not namn:
            namn = ligakod
        filnamn = os.path.join(save_dir, f"{namn}_{säsong}.csv")
        df.to_csv(filnamn, index=False, encoding="utf-8")
        return df
    except Exception:
        return None

