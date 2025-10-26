from hamta_data import hamta_data

# 📅 Ange säsonger (format: "2324" = 2023–2024)
säsonger = ["2526", "2425", "2324"]

# ⚽ Ange ligor att hämta (kod: namn)
ligor = {
    "E0": "england_premier",       # Premier League
    "E1": "england_championship",  # Championship
    "E2": "england_league1",       # League One
    "D1": "tyskland",              # Bundesliga
    "I1": "italien",               # Serie A
    "SP1": "spanien",              # La Liga
    "F1": "frankrike"              # Ligue 1
}

# 🔄 Hämta data för varje liga och säsong
for kod, namn in ligor.items():
    for säsong in säsonger:
        print(f"🔽 Hämtar {namn} {säsong}...")
        df = hamta_data(kod, säsong, namn=namn)
        if df is None:
            print(f"❌ Misslyckades: {namn} {säsong}")
        else:
            print(f"✅ Klar: {namn} {säsong} ({len(df)} rader)")

