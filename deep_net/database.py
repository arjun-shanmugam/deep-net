"""
File: database.py
Purpose: Defines a Database class to allow easy storage and pulls in SQL database
"""
import datetime
import sqlite3

import numpy as np
import pandas as pd
import time
from datetime import date
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import boxscoretraditionalv2

from deep_net.utilities import get_corresponding_nba_season, get_stat_categories


class Database:
    def __init__(self, database, sleep_time=0.6):
        self._conn = sqlite3.connect(database)  # creates if it does not exist
        self._sleep_time = sleep_time

    """
    Adds any games which have not been added since the last API request to the database.
    """

    def update_games_table(self):
        c = self._conn.cursor()
        c.execute('PRAGMA foreign_keys = ON')

        # season is the first year in each season's full title (2019-2020 season --> 2019)
        create_games_table = '''
        CREATE TABLE IF NOT EXISTS Games(
        GAME_ID TEXT NOT NULL,
        GAME_DATE TEXT NOT NULL, 
        SEASON TEXT NOT NULL,
        SEASON_TYPE TEXT NOT NULL,
        HOME_TEAM_ID TEXT NOT NULL,
        VISITING_TEAM_ID TEXT NOT NULL,
        POINT_DIFFERENTIAL INT NOT NULL,
        
        PRIMARY KEY (GAME_ID)
        );
        '''
        c.execute(create_games_table)
        c.close()

        # query database for most recent game
        most_recent_game = self.select_n_most_recent_games("Games", 1, ["SEASON"])
        season_types = ['Regular Season', 'Playoffs']
        all_game_data = []
        if len(most_recent_game) == 0:
            # if there are currently no games in the database, pull all games since 2011
            seasons = range(2011, date.today().year)  # use datetime module for extensibility (requesting from '2022' season returns empty DataFrame)
            for season in seasons:
                print("Pulling game data from season: " + str(season))
                for season_type in season_types:
                    time.sleep(self._sleep_time)  # avoid sending requests with excessive frequency; min sleep time listed as 600ms in nba_api GitHub repo
                    game_data = leaguegamelog.LeagueGameLog(
                        league_id='00',  # '00' indicates NBA
                        season=str(season),  # only the first number in the season's entire title (i.e., '2019-2020' --> '2019')
                        season_type_all_star=season_type,  # indicate 'Regular Season' or 'Playoffs'
                        sorter='DATE',  # sort by date
                        direction='ASC',  # ascending, so that all games from all seasons begin in October 2000 and extend to today
                    ).get_data_frames()[0]  # LeagueGameLog endpoint only returns a single DataFrame
                    all_game_data.append(game_data)
        else:
            # otherwise, we pull all games after the date we most recently pulled

            # first, pull any un-pulled games from current season
            season = most_recent_game['SEASON'].iloc[0]
            day = most_recent_game['GAME_DATE'].iloc[0]

            for season_type in season_types:
                game_data = leaguegamelog.LeagueGameLog(
                    league_id='00',  # '00' indicates NBA
                    season=str(season),  # only the first number in the season's full title (i.e., '2019-2020' --> '2019')
                    season_type_all_star=season_type,  # indicate 'Regular Season' or 'Playoffs'
                    sorter='DATE',  # sort by date
                    date_from_nullable=day + datetime.timedelta(days=1),
                    direction='ASC',
                ).get_data_frames()[0]
                all_game_data.append(game_data)
            # for the case in which the model is not run for >=1 season, we must pull games from un-pulled seasons as well
            seasons = range(int(season) + 1, date.today().year)
            for season in seasons:
                for season_type in season_types:
                    game_data = leaguegamelog.LeagueGameLog(
                        league_id='00',  # '00' indicates NBA
                        season=str(season),  # only the first number in the season's full title (i.e., '2019-2020' --> '2019')
                        season_type_all_star=season_type,  # indicate 'Regular Season' or 'Playoffs'
                        sorter='DATE',  # sort by date
                        direction='ASC',
                    ).get_data_frames()[0]
                    all_game_data.append(game_data)

        if len(all_game_data) != 0:
            games = pd.concat(all_game_data)  # combine all of our pulls from the API into a single DataFrame
            if len(games) == 0:
                print("The Games table is up-to-date.")
            else:
                game_data = self.clean_game_data(games)
                game_data.to_sql('Games', self._conn, if_exists='append', index=False)  # put in database
                print("Most recent games added to Games table.")


    """
    Adds any player box scores which have not been added since the last API request to the database.
    """
    def update_boxscores_table(self):

        # we create the table manually because SQLite does not allow the addition of foreign keys after instantiation
        c = self._conn.cursor()
        c.execute('PRAGMA foreign_keys = ON')
        create_boxscores_table =  '''
                CREATE TABLE IF NOT EXISTS BoxScores(
                PLAYER_ID TEXT NOT NULL,
                PLAYER_NAME TEXT NOT NULL, 
                GAME_ID TEXT NOT NULL,
                GAME_DATE TEXT NOT NULL,
                {} REAL NOT NULL,

                PRIMARY KEY (PLAYER_ID, GAME_ID),
                FOREIGN KEY(GAME_ID) REFERENCES Games(GAME_ID) ON DELETE CASCADE ON UPDATE CASCADE
                );
                '''.format(' REAL,\n'.join(get_stat_categories('BOX SCORE')))
        print(create_boxscores_table)
        c.execute(create_boxscores_table)
        c.close()


        # query database for most recent game
        most_recent_game = self.select_n_most_recent_games('BoxScores', 1)
        print(most_recent_game)
        season_types = ['Regular Season', 'Playoffs']
        all_player_data = []
        if len(most_recent_game) == 0:
            # if there are currently no games in the database, pull all games since 2011
            seasons = range(2011, date.today().year)  # use datetime module for extensibility (requesting from '2022' season returns empty DataFrame)
            for season in seasons:
                print("Pulling player box scores from season: " + str(season))
                for season_type in season_types:
                    time.sleep(self._sleep_time)  # avoid sending requests with excessive frequency; min sleep time listed as 600ms in nba_api GitHub repo
                    player_data = leaguegamelog.LeagueGameLog(
                        league_id='00',  # '00' indicates NBA
                        season=str(season),  # only the first number in the season's entire title (i.e., '2019-2020' --> '2019')
                        season_type_all_star=season_type,  # indicate 'Regular Season' or 'Playoffs'
                        player_or_team_abbreviation='P',  # we are requesting player data, not overall game data
                        sorter='DATE',  # sort by date
                        direction='ASC',  # ascending, so that all games from all seasons begin in October 2000 and extend to today
                    ).get_data_frames()[0]  # LeagueGameLog endpoint only returns a single DataFrame
                    all_player_data.append(player_data)
        else:
            # otherwise, we pull all games after the date we most recently pulled

            # first, pull any un-pulled games from current season

            day = most_recent_game['GAME_DATE'].iloc[0]
            season = get_corresponding_nba_season(day)
            for season_type in season_types:
                player_data = leaguegamelog.LeagueGameLog(
                    league_id='00',  # '00' indicates NBA
                    season=str(season),  # only the first number in the season's full title (i.e., '2019-2020' --> '2019')
                    season_type_all_star=season_type,  # indicate 'Regular Season' or 'Playoffs'
                    sorter='DATE',  # sort by date
                    player_or_team_abbreviation='P',
                    date_from_nullable=day + datetime.timedelta(days=1),
                    direction='ASC',
                ).get_data_frames()[0]
                all_player_data.append(player_data)
            # for the case in which the model is not run for >=1 season, we must pull games from un-pulled seasons as well
            seasons = range(int(season) + 1, date.today().year)
            for season in seasons:
                for season_type in season_types:
                    player_data = leaguegamelog.LeagueGameLog(
                        league_id='00',  # '00' indicates NBA
                        season=str(season),  # only the first number in the season's full title (i.e., '2019-2020' --> '2019')
                        season_type_all_star=season_type,  # indicate 'Regular Season' or 'Playoffs'
                        sorter='DATE',  # sort by date
                        player_or_team_abbreviation='P',
                        direction='ASC',
                    ).get_data_frames()[0]
                    all_player_data.append(player_data)


        if len(all_player_data) != 0:
            boxscores = pd.concat(all_player_data)  # combine all of our pulls from the API into a single DataFrame
            boxscores.to_csv("~/Desktop/players.csv")
            if len(boxscores) == 0:
                print("The BoxScore table is up-to-date.")
            else:
                player_data = self.clean_boxscore_data(boxscores)
                player_data.to_sql('BoxScores', self._conn, if_exists='append', index=False)  # put in database
                print("Most recent data added to BoxScore table.")


    """
    Selects data from the most recent game in a table; returns an empty DataFrame if the table is empty.
    """
    def select_n_most_recent_games(self, table, n, variables_besides_date=None):
        # query database for the most recent game it contains
        if variables_besides_date is None:
            select_most_recent_game = '''
                                SELECT GAME_DATE 
                                FROM {}
                                ORDER BY GAME_DATE DESC
                                LIMIT {};
                                '''.format(table, str(n))
        else:
            select_most_recent_game = '''
                    SELECT GAME_DATE, {}
                    FROM {}
                    ORDER BY GAME_DATE DESC
                    LIMIT {};
                    '''.format(", ".join(variables_besides_date), table, str(n))
        return pd.read_sql_query(sql=select_most_recent_game, con=self._conn, parse_dates=['GAME_DATE'])


    """
    Cleans a DataFrame of game data received from the LeagueGameLog endpoint.
    """
    @staticmethod
    def clean_game_data(games):
        games.reset_index(inplace=True)  # reset indices to work with combined dataset
        game_data = games.loc[:, ['SEASON_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', "MATCHUP", 'PTS']]  # select only the rows we want

        # SEASON_ID is a variable that concatenates season type and year: 22019 --> regular season 2019-20; 42020 --> playoffs, 2020-21
        game_data['SEASON_TYPE'] = game_data['SEASON_ID'].str[0]  # create SEASON_TYPE column equal to the first character of SEASON_ID
        game_data.loc[:, 'SEASON_TYPE'] = np.where(game_data['SEASON_TYPE'] == '2', 'Regular Season', 'Playoffs')
        game_data['SEASON'] = game_data['SEASON_ID'].str[1:]
        game_data = game_data.loc[:, ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'MATCHUP', 'PTS', 'SEASON', 'SEASON_TYPE']]  # delete original SEASON_ID col

        # API delivers two rows per game: one from perspective of home team, one from perspective of visiting team
        # we need to combine each pair of rows into a single row
        # rows corresponding to visiting teams are styled as: VIS @ HOM (i.e., NYK @ CLE)
        # rows corresponding to home teams are styled as: HOM vs. VIS (i.e., CLE vs. NYK)
        game_data['HOME_TEAM_ID'] = np.where(
            game_data['MATCHUP'].str.contains('vs.'),  # if the row corresponds to a home team...
            game_data['MATCHUP'].str[:3],  # ...HOME_TEAM_ID is the first three characters of MATCHUP
            game_data['MATCHUP'].str[-3:]  # otherwise, HOME_TEAM_ID is the last three characters
        )
        game_data['VISITING_TEAM_ID'] = np.where(
            game_data['MATCHUP'].str.contains('@'),  # if the row corresponds to an away team...
            game_data['MATCHUP'].str[:3],  # ... VISITING_TEAM_ID is the first three characters of matchup
            game_data['MATCHUP'].str[-3:]  # otherwise, VISITING_TEAM_ID is the last three characters
        )
        game_data['HOME_OR_AWAY'] = np.where(game_data['MATCHUP'].str.contains('@'), 1, 0)  # now, mark rows as home row (0) or away row (1)
        game_data.sort_values(['GAME_ID', 'HOME_OR_AWAY'],
                              inplace=True)  # each pair of game rows is now adjacent, with the home row first and the visiting row second
        game_data['POINT_DIFFERENTIAL'] = game_data.groupby('GAME_ID')['PTS'].diff(periods=-1)  # for each game_id, calculate home's PTS - away's PTS
        # now, all even rows contain all the data we want
        # odd rows have all the data we want but have a missing value for POINT DIFFERENTIAL
        # thus, we drop all the odd rows (which are visiting rows)
        game_data.drop(game_data.loc[game_data['MATCHUP'].str.contains("@")].index, inplace=True)
        """TODO: put games into the SQL database"""
        game_data = game_data.loc[:, [
                                         'GAME_ID',  # columns which will be stored in the database
                                         'GAME_DATE',
                                         'SEASON',
                                         'SEASON_TYPE',
                                         'HOME_TEAM_ID',
                                         'VISITING_TEAM_ID',
                                         'POINT_DIFFERENTIAL'
                                     ]]
        return game_data

    """
    Cleans a DataFrame of player data received from the LeagueGameLog endpoint.
    """
    @staticmethod
    def clean_boxscore_data(boxscores):
        # note that each row corresponds to one player's performance in one game
        boxscores.reset_index(inplace=True)  # reset indices to work with combined dataset
        cols_to_keep = ['MATCHUP',  # will be eventually dropped
                        'PLAYER_ID',
                        'PLAYER_NAME',
                        'GAME_ID',
                        'GAME_DATE',
                        'MIN',
                        'FGM',
                        'FGA',
                        'FG_PCT',
                        'FG3M',
                        'FG3A',
                        'FG3_PCT',
                        'FTM',
                        'FTA',
                        'FT_PCT',
                        'OREB',
                        'DREB',
                        'REB',
                        'AST',
                        'STL',
                        'BLK',
                        'TOV',
                        'PF',
                        'PTS',
                        'PLUS_MINUS']
        boxscore_data = boxscores.loc[:, cols_to_keep]

        # we want to mark each row as home away
        boxscore_data['HOME_OR_AWAY'] = np.where(boxscore_data['MATCHUP'].str.contains('@'), 'HOME', 'AWAY')
        boxscore_data.drop(['MATCHUP'], axis=1, inplace=True)
        return boxscore_data
