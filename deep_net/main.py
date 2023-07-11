
from IPython.display import HTML, display
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.endpoints import playerdashboardbylastngames
from nba_api.stats.static import teams
from database import Database
import pandas as pd

db = Database("../data/nbastats.db")
db.update_games_table()
db.update_boxscores_table()

games = leaguegamelog.LeagueGameLog(
                                    direction='DESC',
                                    league_id='00',
                                    season_type_all_star='Regular Season',
                                    player_or_team_abbreviation='P',
                                    season='2019',
                                    sorter='DATE',
                                    ).get_data_frames()[0]

games.drop(['SEASON_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 'FANTASY_PTS', 'VIDEO_AVAILABLE'], inplace=True, axis=1)
games.sort_values(['GAME_ID', 'PLAYER_ID'], inplace=True)
print(list(games.columns))



box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id='0021901316').get_data_frames()
print(box_score[0])
lebron_last_10 = playerdashboardbylastngames.PlayerDashboardByLastNGames(player_id='2544',
                                                                         last_n_games='10',
                                                                         measure_type_detailed='Base',
                                                                         ).get_data_frames()
print(lebron_last_10[0].columns.values.tolist())

print(lebron_last_10[0][['PTS', 'REB', 'AST']])

