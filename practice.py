from nba_api.stats.static import teams

nba_teams = teams.get_teams()
print(f"Number of NBA teams: {len(nba_teams)}")
for team in nba_teams:
    print(team['id'], team['full_name'])