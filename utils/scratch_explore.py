import os
import pandas as pd
import re

# # Load csv and try to learn how position classification works

# fp = "/Users/andydelworth/Downloads/games_and_players.csv"
# df = pd.read_csv(fp)
# # Compare the rebounds of guards, forwards, and centers
# # Take the mean of the dataframes down the columns
# avgs = df.mean()
# rgx = ".*_reb.*"
# # Get the columns that match the regex
# reb_cols = [col for col in df.columns if re.match(rgx, col)]
# # Display the average rebounds for each position
# print(avgs[reb_cols])

# Explore nba_api

from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import boxscoretraditionalv2

games = leaguegamelog.LeagueGameLog(
                                    direction='DESC',
                                    league_id='00',
                                    season_type_all_star='Regular Season',
                                    player_or_team_abbreviation='P',
                                    season='2019',
                                    sorter='DATE',
                                    ).get_data_frames()[0]
print(games.columns)