import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv3 as boxscore
import features as ft
import time

def fetch_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    game_ids = df['GAME_ID'].unique()[1107:]
    print(len(game_ids))
    for row, game_id in enumerate(game_ids):
        time.sleep(0.4)
        game_stats = boxscore.BoxScoreAdvancedV3(game_id).get_data_frames()[1]
        results.append(game_stats)
        print(f"{(row+1) / len(game_ids) * 100:.2f}% Successfully retrieved stats for Game: {game_id}")
    output = pd.concat(results, ignore_index=True)
    return output
    
def main():
    df = pd.read_parquet('player_game_logs_with_features_2024-25_TEST.parquet')
    game_df = pd.read_parquet('game_stats_2024-25_TEST.parquet')
    game_stats = fetch_game_stats(df)
    print(game_stats.head())
    game_df = pd.concat([game_df, game_stats], ignore_index=True)
    game_df.to_parquet('game_stats_2024-25_TEST.parquet')
    
if __name__ == "__main__":
    main()