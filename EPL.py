#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dan hislop |  https://github.com/danhislop | hislopdan@gmail.com

Created on Fri Apr  12 19:41:50 2019
pd.set_option('display.width', 700)
pd.set_option('display.max_columns',21)
"""
# Goal: summarize English Premier Leagues's 17-18 season.  Separate home v away record to look for correlations. 
# Data source contains game results, but does not contain team win/loss records, need to calculate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Use the pandas read_csv method to read the file into a dataframe
filepath = '/Users/danhi/Dropbox/bwork/python/datasets/season-1718_premier_league.csv'
epl = pd.read_csv(filepath, index_col=0)


def calc_record(team_df, home_or_away):

    ''' 
    input: 
        team_df = dataframe containing individual game results for this team's home (or away) games
        home_or_away = string, denoting whether team_df contains games played at home, or away
    purpose: calculate (Wins, Draws, Loss, Points) record for the games passed in
        note: Because dataset designates winner as 'home' or 'away' team, we need to separate in order to calculate
    output: returns a list of (Wins, Draws, Loss, Points) for this team's home or away game
    '''
    
    if home_or_away == 'home':
        record = [
                team_df.loc[(team_df['FTR'] == 'H')].shape[0],
                team_df.loc[(team_df['FTR'] == 'D')].shape[0],
                team_df.loc[(team_df['FTR'] == 'A')].shape[0],
                team_df.loc[(team_df['FTR'] == 'H')].shape[0]*3 + team_df.loc[(team_df['FTR'] == 'D')].shape[0]*1,]

    elif home_or_away == 'away':
        record = [
                team_df.loc[(team_df['FTR'] == 'A')].shape[0],
                team_df.loc[(team_df['FTR'] == 'D')].shape[0],
                team_df.loc[(team_df['FTR'] == 'H')].shape[0],
                team_df.loc[(team_df['FTR'] == 'A')].shape[0]*3 + team_df.loc[(team_df['FTR'] == 'D')].shape[0]*1,]
    else:
        print("please specify home or away")
        return()
    print(record)    
    return(record)

def team_df(teamname, year, df):
    '''
    input: 
        teamname = string, e.g. "Tottenham"
        year = string, e.g. 2017-18
        df = dataframe containing individual game results for all teams over whole season
    purpose: drop all games $teamname wasn't in,then calculate home, away, and overall record
    output: a list containing teamname, year, season record, home record, away record 
    '''
    info = [teamname,year]
    
    # create dataframes for each team: home games, away games
    #all = epl.loc[(epl['HomeTeam'] == teamname) | (epl['AwayTeam'] == teamname), :]
    home = epl.loc[(epl['HomeTeam'] == teamname)]
    away = epl.loc[(epl['AwayTeam'] == teamname)]


    # homerecord looks like this: (13, 4, 2, 43) (homewins, homedraws, homeloss, homepoints)
    homerecord = calc_record(home, 'home')    
    awayrecord = calc_record(away, 'away')   
    
    # add home and away records to create season record
    seasonrecord = []
    for i in range(0,4):
        seasonrecord.append(homerecord[i] + awayrecord[i])
    
    # return all above info in a list     
    output = []
    output.extend(info + seasonrecord + homerecord + awayrecord)    
    return(output)


year = '2017-18'
teamnames = epl.HomeTeam.unique()
teamnames = np.sort(teamnames, axis=0)

# Create dataframe to store all team records
col_names = ['Team', 'Season', 'Win', 'Draw', 'Loss', 'Points', 'HomeWin', 'HomeDraw', 'HomeLoss', 'HomePoints', 'AwayWin', 'AwayDraw', 'AwayLoss', 'AwayPoints']
team_records = pd.DataFrame(columns = col_names)

  
for name in teamnames:
    out = team_df(name, year, game_results)
    # append each list into a new DF.  Using len(team_records) to determine next empty row
    team_records.loc[len(team_records)] = out

team_records.set_index("Team", inplace=True)
print(team_records.sort_values(by='Points', ascending=False, inplace=True))


# More exploration to do here; looking for interesting data to correlate, such as comparing home/away record
team_records['Home_Away_Points_Ratio'] = team_records['HomePoints'] / team_records['AwayPoints']




# Now Visualize 
# Setup a bar chart showing TotalPoints

ax = team_records['Points'].plot(kind='bar', 
                 figsize=(20,8),
                 width=0.6,
                 fontsize=14,
                 #color=['#5cb85c', '#5bc0de', '#d9534f']
                 )

ax.set_title("Total Points earned by Team in EPL Season 2017-18", fontsize=16)

# Remove top, left, right borders, and set on white background
ax.patch.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_ylabel('Points earned (win=3, draw=1, loss=0)')
ax.yaxis.set_major_locator(plt.NullLocator())


# to prepare for labels, bring all the values from dataframe into one list; convert to string and add % symbol
values = team_records.iloc[:,4].tolist()
values = [i for i in values]
valstring = [str(i) for i in values]
rects = ax.patches   # creates a list of each patch (bar) in the chart

# for each patch, set location and value of label
for rect, value in zip(rects, valstring):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, value,
            ha='center', va='bottom')
   
    
