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
