"""
src/bet365_api.py
~~~~~~~~~~~~~~~~~
Hämtar live odds från Bet365 via The Odds API.
Integreras i din app för automatiska odds.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

class Bet365API:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.region = "eu"  # EU-odds (inkl. Bet365)
        self.market = "h2h"  # Head-to-head (1X2)

    def hämta_odds(
        self,
        sport: str = "soccer_epl",  # t.ex. soccer_epl, soccer_championship
        date_format: str = "iso",
        api_key: Optional[str] = None
    ) -> Dict:
        """
        Hämtar aktuella odds för en sport.

        Args:
            sport: t.ex. "soccer_epl" (Premier League)
            date_format: "iso" för ISO-datum

        Returns:
            Dict med matcher och odds från Bet365
        """
        api_key = api_key or self.api_key
        params = {
            "apiKey": api_key,
            "regions": self.region,
            "markets": self.market,
            "oddsFormat": "decimal",
            "dateFormat": date_format,
            "sport": sport
        }

        try:
            response = requests.get(f"{self.base_url}/sports/{sport}/odds", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.RequestException as e:
            print(f"Fel vid hämtning: {e}")
            return {}

    def hämta_match_odds(self, match_id: str) -> Dict:
        """Hämtar odds för specifik match."""
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": self.market,
            "oddsFormat": "decimal"
        }
        url = f"{self.base_url}/sports/soccer_epl/events/{match_id}/odds"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Fel vid match-hämtning: {e}")
            return {}

    def parse_odds_to_df(self, data: Dict) -> pd.DataFrame:
        """Parserar API-data till DataFrame för din app."""
        if "data" not in data:
            return pd.DataFrame()

        rows = []
        for event in data["data"]:
            home = event["home_team"]
            away = event["away_team"]
            commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
            for bookmaker in event["bookmakers"]:
                if bookmaker["key"] == "bet365":
                    for market in bookmaker["markets"]:
                        if market["key"] == "h2h":
                            outcomes = market["outcomes"]
                            home_odds = next((o["price"] for o in outcomes if o["name"] == home), None)
                            away_odds = next((o["price"] for o in outcomes if o["name"] == away), None)
                            draw_odds = next((o["price"] for o in outcomes if o["name"] == "Draw"), None)
                            if home_odds and away_odds and draw_odds:
                                rows.append({
                                    "match": f"{home} vs {away}",
                                    "home_team": home,
                                    "away_team": away,
                                    "start_time": commence,
                                    "home_odds": home_odds,
                                    "draw_odds": draw_odds,
                                    "away_odds": away_odds
                                })

        return pd.DataFrame(rows)

# EXEMPEL: Testa API
if __name__ == "__main__":
    api = Bet365API("DIN_API_NYCKEL_HÄR")
    data = api.hämta_odds("soccer_epl")
    df = api.parse_odds_to_df(data)
    print(df.head())