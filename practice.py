from nba_api.stats.static import teams
from nba_api.stats.static import players
import pandas as pd

df = pd.read_parquet('player_game_logs_2024-25_TEST.parquet')

print(df.head())