import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import commonplayerinfo
import time
from requests.exceptions import ReadTimeout, ConnectionError
from concurrent.futures import ThreadPoolExecutor, as_completed

#grab all active players
nba_players = players.get_active_players()


def fetch_player_stats(player_id: int, season: str, retries: int = 3, timeoue: int = 10, backoff: float = 4):
    '''Add game log stats for a player for a specific season to the DataFrame.'''
    player_name = [player['full_name'] for player in nba_players if player['id'] == player_id][0]
    print(f"Fetching stats for {player_name}, ID {player_id}")
    attempt = 0
    while attempt < retries:
        try:
            #grab selected player stats for season and add to dataframe
            player_stats = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            stats_df = player_stats.get_data_frames()[0]
            if stats_df.empty:
                return None #no games this season
            desired = ['GAME_DATE', 'MATCHUP', 'PTS', 'AST', 'REB']
            selected = stats_df[desired].copy()
            selected.insert(0, 'PLAYER_ID', player_id)
            selected.insert(1, 'PLAYER_NAME', player_name)
            print(f"Successfully retrieved stats for player ID {player_id}, player Name {player_name}")
            return selected
        except (ReadTimeout, ConnectionError) as e:
            attempt += 1
            wait = backoff ** attempt
            print(f"Timeout/connection error for player ID {player_id}. Retrying in {wait:.1f} seconds...")
            time.sleep(wait)
        except Exception as e:
            print(f"Failed to retrieve stats for player ID {player_id}: {e}")
    print(f"All retries failed for player ID {player_id}, player Name {player_name}. Skipping.")
    return None

table = pd.DataFrame()
for player in nba_players:
    player_stats = fetch_player_stats(player['id'], season='2024-25')
    if player_stats is not None:
        table = pd.concat([table, player_stats], ignore_index=True)
    time.sleep(0.35) #pause to avoid rate limiting
    
table.to_parquet('player_game_logs_2024-25_TEST.parquet', index=False)
print(table)