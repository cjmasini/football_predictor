import pandas as pd
import numpy as np
from glicko2 import Player

def make_row(team_stats, id0, id1, glicko, team_results):
    row = {'team_0': id0, 'team_1': id1}

    #glicko
    rating_0, rating_1 = glicko[id1].getRating(), glicko[id0].getRating()
    rd_0, rd_1 = glicko[id1].getRd(), glicko[id0].getRd()
    row['glicko_0'] = rating_0
    row['glicko_1'] = rating_1

    #RPI and win percentage
    #win percentage
    temp = 0
    for game in team_results[id0]:
        temp += game['result']
    if len(team_results[id0]) == 0:
        wp0 = 0
    else:
        wp0 = float(temp)/len(team_results[id0])
    row['win_rate_0'] = wp0

    temp = 0
    for game in team_results[id1]:
        temp += game['result']
    if len(team_results[id1]) == 0:
        wp1 = 0
    else:
        wp1 = float(temp)/len(team_results[id1])
    row['win_rate_1'] = wp1

    #weighted win percentage
    temp = 0.0
    for game in team_results[id0]:
        if game['loc'] == 'N':
            temp += 1*game['result']
        elif game['loc'] == 'H':
            temp += 1.4*game['result']
        else:
            temp += .6*game['result']
    if len(team_results[id0]) == 0:
        wwp0 = 0
    else:
        wwp0 = temp/len(team_results[id0])
    row['weighted_win_rate_0'] = wwp0

    temp = 0.0
    for game in team_results[id1]:
        if game['loc'] == 'N':
            temp += 1*game['result']
        elif game['loc'] == 'A':
            temp += 1.4*game['result']
        else:
            temp += .6*game['result']
    if len(team_results[id1]) == 0:
        wwp1 = 0
    else:
        wwp1 = temp/len(team_results[id1])
    row['weighted_win_rate_1'] = wwp1

    #Opponents win percentage
    temp = 0
    denom = 0.0
    for game in team_results[id0]:
        for game2 in team_results[game['opponent']]:
            if game2['opponent'] != id0:
                temp += game2['result']
                denom += 1
    if denom == 0:
        owp0 = 0
    else:
        owp0 = temp/denom
    row['opponents_win_rate_0'] = owp0

    temp = 0
    denom = 0.0
    for game in team_results[id1]:
        for game2 in team_results[game['opponent']]:
            if game2['opponent'] != id1:
                temp += game2['result']
                denom += 1
    if denom == 0:
        owp1 = 0
    else:
        owp1 = temp/denom
    row['opponents_win_rate_1'] = owp1

    #Opponents opponents win percentage
    temp = 0
    denom = 0.0
    for game in team_results[id0]:
        for game2 in team_results[game['opponent']]:
            for game3 in team_results[game2['opponent']]:
                if game3['opponent'] != id0:
                    temp += game3['result']
                    denom += 1
    if denom == 0:
        oowp0 = 0
    else:
        oowp0 = temp/denom
    row['opponents_opponents_win_rate_0'] = oowp0

    temp = 0
    denom = 0.0
    for game in team_results[id1]:
        for game2 in team_results[game['opponent']]:
            for game3 in team_results[game2['opponent']]:
                if game3['opponent'] != id1:
                    temp += game3['result']
                    denom += 1
    if denom == 0:
        oowp1 = 0
    else:
        oowp1 = temp/denom
    row['opponents_opponents_win_rate_1'] = oowp1

    row['rpi_0'] = .25*wwp0 + .5*owp0 + .25*oowp0
    row['rpi_1'] = .25*wwp1 + .5*owp1 + .25*oowp1

    #Pythagorean Expectation
    row['pyth_exp_0'] = 1.0/(1 + (team_stats[id0]['points_against']*1.0/team_stats[id0]['points'])**8) if team_stats[id0]['points'] else 0
    row['pyth_exp_1'] = 1.0/(1 + (team_stats[id1]['points_against']*1.0/team_stats[id1]['points'])**8) if team_stats[id1]['points'] else 0

    #Basic Statistics
    row['points_0'] = team_stats[id0]['points']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['num_passes_0'] = team_stats[id0]['num_passes']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['pass_yards_0'] = team_stats[id0]['pass_yards']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['yards_per_pass_0'] = team_stats[id0]['pass_yards']/team_stats[id0]['num_passes'] if team_stats[id0]['num_passes'] else 0
    row['num_rushes_0'] = team_stats[id0]['num_rushes']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['rush_yards_0'] = team_stats[id0]['rush_yards']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['yards_per_rush_0'] = team_stats[id0]['rush_yards']/team_stats[id0]['num_rushes'] if team_stats[id0]['num_rushes'] else 0
    row['num_plays_0'] = team_stats[id0]['num_plays']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['total_yards_0'] = team_stats[id0]['total_yards']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['yards_per_play_0'] = team_stats[id0]['total_yards']/team_stats[id0]['num_plays'] if team_stats[id0]['num_plays'] else 0
    row['num_turnovers_0'] = team_stats[id0]['num_turnovers']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['penalty_yards_0'] = team_stats[id0]['penalty_yards']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['time_of_possession_0'] = team_stats[id0]['time_of_possession']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['points_against_0'] = team_stats[id0]['points_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['num_passes_against_0'] = team_stats[id0]['num_passes_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['pass_yards_against_0'] = team_stats[id0]['pass_yards_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['yards_per_pass_against_0'] = team_stats[id0]['pass_yards_against']/team_stats[id0]['num_passes_against'] if team_stats[id0]['num_passes_against'] else 0
    row['num_rushes_against_0'] = team_stats[id0]['num_rushes_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['rush_yards_against_0'] = team_stats[id0]['rush_yards_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['yards_per_rush_against_0'] = team_stats[id0]['rush_yards_against']/team_stats[id0]['num_rushes_against'] if team_stats[id0]['num_rushes_against'] else 0
    row['num_plays_against_0'] = team_stats[id0]['num_plays_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['total_yards_against_0'] = team_stats[id0]['total_yards_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['yards_per_play_against_0'] = team_stats[id0]['total_yards_against']/team_stats[id0]['num_plays_against'] if team_stats[id0]['num_plays_against'] else 0
    row['num_turnovers_against_0'] = team_stats[id0]['num_turnovers_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['penalty_yards_against_0'] = team_stats[id0]['penalty_yards_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['time_of_possession_against_0'] = team_stats[id0]['time_of_possession_against']/team_stats[id0]['games'] if team_stats[id0]['games'] else 0
    row['points_1'] = team_stats[id1]['points']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['num_passes_1'] = team_stats[id1]['num_passes']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['pass_yards_1'] = team_stats[id1]['pass_yards']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['yards_per_pass_1'] = team_stats[id1]['pass_yards']/team_stats[id1]['num_passes'] if team_stats[id1]['num_passes'] else 0
    row['num_rushes_1'] = team_stats[id1]['num_rushes']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['rush_yards_1'] = team_stats[id1]['rush_yards']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['yards_per_rush_1'] = team_stats[id1]['rush_yards']/team_stats[id1]['num_rushes'] if team_stats[id1]['num_rushes'] else 0
    row['num_plays_1'] = team_stats[id1]['num_plays']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['total_yards_1'] = team_stats[id1]['total_yards']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['yards_per_play_1'] = team_stats[id1]['total_yards']/team_stats[id1]['num_plays'] if team_stats[id1]['num_plays'] else 0
    row['num_turnovers_1'] = team_stats[id1]['num_turnovers']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['penalty_yards_1'] = team_stats[id1]['penalty_yards']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['time_of_possession_1'] = team_stats[id1]['time_of_possession']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['points_against_1'] = team_stats[id1]['points_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['num_passes_against_1'] = team_stats[id1]['num_passes_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['pass_yards_against_1'] = team_stats[id1]['pass_yards_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['yards_per_pass_against_1'] = team_stats[id1]['pass_yards_against']/team_stats[id1]['num_passes_against'] if team_stats[id1]['num_passes_against'] else 0
    row['num_rushes_against_1'] = team_stats[id1]['num_rushes_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['rush_yards_against_1'] = team_stats[id1]['rush_yards_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['yards_per_rush_against_1'] = team_stats[id1]['rush_yards_against']/team_stats[id1]['num_rushes_against'] if team_stats[id1]['num_rushes_against'] else 0
    row['num_plays_against_1'] = team_stats[id1]['num_plays_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['total_yards_against_1'] = team_stats[id1]['total_yards_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['yards_per_play_against_1'] = team_stats[id1]['total_yards_against']/team_stats[id1]['num_plays_against'] if team_stats[id1]['num_plays_against'] else 0
    row['num_turnovers_against_1'] = team_stats[id1]['num_turnovers_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['penalty_yards_against_1'] = team_stats[id1]['penalty_yards_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0
    row['time_of_possession_against_1'] = team_stats[id1]['time_of_possession_against']/team_stats[id1]['games'] if team_stats[id1]['games'] else 0

    row = pd.DataFrame(row, index=[0])
    return row

def update_stats(team_stats, row, glicko, team_results):
    team_stats[row.Winning]['games'] += 1
    team_stats[row.Losing]['games'] += 1

    #Update glicko scores
    w_rating, l_rating = glicko[row.Winning].getRating(), glicko[row.Losing].getRating()
    w_rd, l_rd = glicko[row.Winning].getRd(), glicko[row.Losing].getRd()
    glicko[row.Winning].update_player([l_rating], [l_rd], [1])
    glicko[row.Losing].update_player([w_rating], [w_rd], [0])

    #Update team results
    team_results[row.Winning].append({'opponent': row.Losing, 'score': row.Winning_Points, 'opponent_score': row.Losing_Points, 'result': 1, 'loc': row.Winning_loc})
    team_results[row.Losing].append({'opponent': row.Winning, 'score': row.Losing_Points, 'opponent_score': row.Winning_Points, 'result': 0, 'loc': row.Losing_loc})

    #Update basic statistics
    team_stats[row.Winning]['points'] += int(row.Winning_Points)
    team_stats[row.Winning]['num_passes'] += int(row.Winning_Passes)
    team_stats[row.Winning]['pass_yards'] += int(row.Winning_Pass_Yards)
    team_stats[row.Winning]['num_rushes'] += int(row.Winning_Rushes)
    team_stats[row.Winning]['rush_yards'] += int(row.Winning_Rush_Yards)
    team_stats[row.Winning]['num_plays'] += int(row.Winning_Plays)
    team_stats[row.Winning]['total_yards'] += int(row.Winning_Total_Yards)
    team_stats[row.Winning]['num_turnovers'] += int(row.Winning_TO)
    team_stats[row.Winning]['penalty_yards'] += int(row.Winning_Pen_Yards)
    team_stats[row.Winning]['time_of_possession'] += int(row.Winning_TOP)
    team_stats[row.Winning]['points_against'] += int(row.Losing_Points)
    team_stats[row.Winning]['num_passes_against'] += int(row.Losing_Passes)
    team_stats[row.Winning]['pass_yards_against'] += int(row.Losing_Pass_Yards)
    team_stats[row.Winning]['num_rushes_against'] += int(row.Losing_Rush_Attempts)
    team_stats[row.Winning]['rush_yards_against'] += int(row.Losing_Rush_Yards)
    team_stats[row.Winning]['num_plays_against'] += int(row.Losing_Total_Plays)
    team_stats[row.Winning]['total_yards_against'] += int(row.Losing_Total_Yards)
    team_stats[row.Winning]['num_turnovers_against'] += int(row.Losing_TO)
    team_stats[row.Winning]['penalty_yards_against'] += int(row.Losing_Pen_Yards)
    team_stats[row.Winning]['time_of_possession_against'] += int(row.Losing_TOP)
    team_stats[row.Losing]['points'] += int(row.Losing_Points)
    team_stats[row.Losing]['num_passes'] += int(row.Losing_Passes)
    team_stats[row.Losing]['pass_yards'] += int(row.Losing_Pass_Yards)
    team_stats[row.Losing]['num_rushes'] += int(row.Losing_Rush_Attempts)
    team_stats[row.Losing]['rush_yards'] += int(row.Losing_Rush_Yards)
    team_stats[row.Losing]['num_plays'] += int(row.Losing_Total_Plays)
    team_stats[row.Losing]['total_yards'] += int(row.Losing_Total_Yards)
    team_stats[row.Losing]['num_turnovers'] += int(row.Losing_TO)
    team_stats[row.Losing]['penalty_yards'] += int(row.Losing_Pen_Yards)
    team_stats[row.Losing]['time_of_possession'] += int(row.Losing_TOP)
    team_stats[row.Losing]['points_against'] += int(row.Winning_Points)
    team_stats[row.Losing]['num_passes_against'] += int(row.Winning_Passes)
    team_stats[row.Losing]['pass_yards_against'] += int(row.Winning_Pass_Yards)
    team_stats[row.Losing]['num_rushes_against'] += int(row.Winning_Rushes)
    team_stats[row.Losing]['rush_yards_against'] += int(row.Winning_Rush_Yards)
    team_stats[row.Losing]['num_plays_against'] += int(row.Winning_Plays)
    team_stats[row.Losing]['total_yards_against'] += int(row.Winning_Total_Yards)
    team_stats[row.Losing]['num_turnovers_against'] += int(row.Winning_TO)
    team_stats[row.Losing]['penalty_yards_against'] += int(row.Winning_Pen_Yards)
    team_stats[row.Losing]['time_of_possession_against'] += int(row.Winning_TOP)
    return team_stats, glicko, team_results


def get_data_matrix(year):
    df = pd.read_csv('data.csv')
    df = df[df['Year'].isin([year-1,year])]
    team_ids = set(df['Winning']).union(df['Losing'])
    team_results = {}
    team_stats = {}
    glicko = dict(zip(list(team_ids), [Player() for _ in range(len(team_ids))]))
    for id in team_ids:
        team_results[id] = []
        team_stats[id] = {'games':0,'points':0,'num_passes':0,'pass_yards':0,'num_rushes':0,'rush_yards':0,'num_plays':0,'total_yards':0,'num_turnovers':0,'penalty_yards':0,'time_of_possession':0,'points_against':0,'num_passes_against':0,'pass_yards_against':0,'num_rushes_against':0,'rush_yards_against':0,'num_plays_against':0,'total_yards_against':0,'num_turnovers_against':0,'penalty_yards_against':0,'time_of_possession_against':0}
    data_matrix = pd.DataFrame()
    results_matrix = pd.DataFrame()
    for row in df.itertuples():
        id0 = row.Winning if row.Winning < row.Losing else row.Losing
        id1 = row.Winning if row.Winning > row.Losing else row.Losing
        if row.Year == year:
            data_matrix = data_matrix.append(make_row(team_stats,id0,id1,glicko,team_results))
        results_matrix = results_matrix.append(pd.DataFrame({'id0':id0,'id1':id1,'points_0':(row.Winning_Points if row.Winning < row.Losing else row.Losing_Points), 'points_1':(row.Winning_Points if row.Winning > row.Losing else row.Losing_Points)}, index=[0]))
        team_stats, glicko, team_results = update_stats(team_stats, row, glicko, team_results)
    data_matrix.to_csv(str(year) + "dataMatrix.csv")
    results_matrix.to_csv(str(year) + "resultsMatrix.csv")
    print(str(year) + ": Done")
    return data_matrix, results_matrix

combined_data_matrix = pd.DataFrame()
combined_results_matrix = pd.DataFrame()
for i in range(2012, 2019):
    dm, rm = get_data_matrix(i)
    combined_data_matrix = combined_data_matrix.append(dm)
    combined_results_matrix = combined_results_matrix.append(rm)
combined_data_matrix.to_csv("2012-2018_data_matrix.csv")
combined_results_matrix.to_csv("2012-2018_results_matrix.csv")


