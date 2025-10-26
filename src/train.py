import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# Lägg till projektroten i sys.path så att config.py hittas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from träna_liga import träna_liga
from predict import förutsäg_match
from config import DATA_DIR, PARAM_DIR


def expected_calibration_error(df_eval, val_liga, n_bins=10):
    """
    Beräkna Expected Calibration Error (ECE) för 1X2-sannolikheter.
    Kräver att df_eval innehåller kolumner: HomeTeam, AwayTeam, FTHG, FTAG.
    """
    preds = []
    for _, rad in df_eval.iterrows():
        try:
            _, sann, _, _, _ = förutsäg_match(df_eval, rad["HomeTeam"], rad["AwayTeam"], val_liga)
            preds.append(sann)
        except Exception:
            preds.append({"H": np.nan, "D": np.nan, "A": np.nan})

    df_eval[["pH", "pD", "pA"]] = pd.DataFrame(preds)
    df_eval["outcome"] = np.where(df_eval["FTHG"] > df_eval["FTAG"], "H",
                           np.where(df_eval["FTHG"] == df_eval["FTAG"], "D", "A"))

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for col in ["H", "D", "A"]:
        true_bin = (df_eval["outcome"] == col).astype(int)
        prob_col = df_eval[f"p{col}"].values
        bin_ids = np.digitize(prob_col, bins) - 1
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() > 0:
                avg_conf = prob_col[mask].mean()
                avg_acc = true_bin[mask].mean()
                ece += (mask.sum() / len(df_eval)) * abs(avg_conf - avg_acc)
    return round(ece, 4)


def main():
    # --- Hitta alla CSV-filer i data/ ---
    alla_filer = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    # --- Gruppera per liga (prefix före första "_") ---
    ligor = defaultdict(list)
    for fil in alla_filer:
        if "_" in fil:
            prefix = fil.split("_")[0]
            ligor[prefix].append(fil)

    if not ligor:
        print("❌ Inga giltiga ligafiler hittades i 'data/'")
        return

    # --- Träna varje liga ---
    sammanfattning = []
    for liga, csv_filer in ligor.items():
        print(f"\n🔄 Tränar {liga} baserat på {len(csv_filer)} fil(er)...")
        resultat = träna_liga(liga, csv_filer)
        if resultat:
            try:
                df = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in csv_filer])
                df = df.dropna(subset=["FTHG", "FTAG"])
                senaste_säsong = df["Season"].iloc[-1] if "Season" in df.columns else None
                if senaste_säsong:
                    df_eval = df[df["Season"] == senaste_säsong].copy()
                    ece = expected_calibration_error(df_eval, liga)
                else:
                    ece = None
            except Exception:
                ece = None

            resultat["ECE"] = ece
            sammanfattning.append(resultat)

    # --- Visa sammanfattning ---
    print("\n📋 Träningssammanfattning:")
    for r in sammanfattning:
        print(f"- {r['liga']}: Brier score {r['brier_score']}, "
              f"formfaktor {r['form_factor']}, zero inflation {r['zero_inflation']}, "
              f"ECE {r['ECE']}")

    # --- Logga till CSV ---
    loggfil = os.path.join(PARAM_DIR, "brier_log.csv")
    df_new = pd.DataFrame(sammanfattning)
    df_new["run"] = pd.Timestamp.now()

    if os.path.exists(loggfil):
        df_logg = pd.read_csv(loggfil)
        df_logg = pd.concat([df_logg, df_new], ignore_index=True)
    else:
        df_logg = df_new

    df_logg.to_csv(loggfil, index=False)

    print(f"\n🗂 Logg uppdaterad: {loggfil}")
    print(df_logg.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
