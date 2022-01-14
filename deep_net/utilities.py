import pandas as pd, time, numpy as np, datetime as dt
from tqdm import tqdm
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.endpoints import playernextngames
from nba_api.stats.endpoints import leaguegamefinder
"""
Given a Timestamp object (YYYY-MM-DD), return the first year of the corresponding NBA season
"""
def get_corresponding_nba_season(date):

    if date.month > 7:
        return date.year
    else:
        return date.year - 1

"""
Returns a list of the requested statistical categories
"""
def get_stat_categories(request):
    box_score = [
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
        'PLUS_MINUS',
        'HOME_OR_AWAY']
    if request == 'BOX SCORE':
        return box_score
"""
Returns the string id to be used to identify a team.
Ex: 1610612748 -> "MIA"
"""
def team_number_id_to_string_id(number_id):
    mapping = {
    1610612737: 'ATL',
    1610612738: 'BOS',
    1610612739: 'CLE', 
    1610612740: 'NOP', 
    1610612741: 'CHI', 
    1610612742: 'DAL', 
    1610612743: 'DEN', 
    1610612744: 'GSW', 
    1610612745: 'HOU', 
    1610612746: 'LAC', 
    1610612747: 'LAL', 
    1610612748: 'MIA', 
    1610612749: 'MIL', 
    1610612750: 'MIN',
    1610612751: 'BKN', 
    1610612752: 'NYK', 
    1610612753: 'ORL', 
    1610612754: 'IND', 
    1610612755: 'PHI', 
    1610612756: 'PHX', 
    1610612757: 'POR', 
    1610612758: 'SAC', 
    1610612759: 'SAS', 
    1610612760: 'OKC',
    1610612761: 'TOR', 
    1610612762: 'UTA', 
    1610612763: 'MEM', 
    1610612764: 'WAS', 
    1610612765: 'DET', 
    1610612766: 'CHA'}
    return mapping[number_id]

"""
Gets the nba games scheduled for later on the same day (most of code copied and
pasted directly from a slack channel with some additions) 
"""
def get_upcoming_nba_games():
    SEASON = '2021-22'
    SEASON_TYPE = 'Regular Season'
    TODAY_DATE = dt.date.today().strftime('%b %d, %Y').upper()

    nba_teams = pd.DataFrame(teams.get_teams()).id.to_list()
    
    df = []
    for team in tqdm(nba_teams,desc='nba_players'):
        df.append(commonteamroster.CommonTeamRoster(team_id=team).get_data_frames()[0])
        time.sleep(1)
    nba_players = pd.concat(df)
    nba_players['abbreviation'] = nba_players.TeamID.map(pd.DataFrame(teams.get_teams()).set_index('id')['abbreviation'].to_dict())

    player_of_each_team = []
    for i in tqdm(nba_teams,desc='player_of_each_team'):
        player_of_each_team.append(nba_players[nba_players.TeamID==i].iloc[0]['PLAYER_ID'])
        time.sleep(1)

    df = []
    for i in tqdm(player_of_each_team,desc='team_next_games'):
        df.append(playernextngames.PlayerNextNGames(player_id=i,number_of_games=5).get_data_frames()[0])
        time.sleep(1)
    team_next_games = pd.concat(df)

    today_games = team_next_games[team_next_games.GAME_DATE==TODAY_DATE]
    games_set = set()
    for index, row in today_games.iterrows():
        games_set.add((team_number_id_to_string_id(row['HOME_TEAM_ID']), team_number_id_to_string_id(row['VISITOR_TEAM_ID'])))
    print(list(games_set))
    return list(games_set)

get_upcoming_nba_games()
