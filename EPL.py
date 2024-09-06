#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dan hislop |  https://github.com/danhislop | hislopdan@gmail.com

Created on Fri Apr  12 19:41:50 2019
pd.set_option('display.width', 700)
pd.set_option('display.max_columns',21)
"""
# Goal: summarize English Premier Leagues's statistis.  Separate home v away record to look for correlations.

### next: mid-adding season column into dataframe and now need to see it all the way through

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
pd.set_option('display.width', 700)
pd.set_option('display.max_columns',21)


def get_csv(scope):
    ''' input 'scope' is a string indicating 'single' or 'all' seasons
    purpose: use pandas read_csv to pull data from local files into dataframe
    output: dataframe with specified seasons for all teams
    '''

    if scope == 'single':
        # SINGLE SEASON Use the pandas read_csv method to read the file into a dataframe
        filepath = '/Users/danhi/Dropbox/bwork/python/datasets/season-1718_premier_league.csv'
        epl = pd.read_csv(filepath, index_col=0)
        epl['Season'] = '2017-18'

    elif scope == 'all':
        ## ALL SEASONS Use the pandas read_csv method to read the file into a dataframe
        for i in range(10,20):
            if i == 10:
                filepath = '/Users/danhi/Dropbox/bwork/python/datasets/epl_data/season-09{}_csv.csv'.format(i)
                epl = pd.read_csv(filepath, index_col=0)
                epl['Season'] = '2009-10'
            else:
                filepath = '/Users/danhi/Dropbox/bwork/python/datasets/epl_data/season-{}{}_csv.csv'.format(i-1,i)
                this_epl = pd.read_csv(filepath, index_col=0)
                this_epl['Season'] = '20{}-{}'.format(i-1,i)
                epl = epl.append(this_epl)
    else:
        print("please specify single or all")
        return()

    return(epl)

def calc_record(df, home_or_away):

    '''
    input:
        df = dataframe containing individual game results for this team's home (or away) games
        home_or_away = string, denoting whether df contains games played at home, or away
    purpose: calculate (Wins, Draws, Loss, Points) record for the games passed in
        note: Because dataset designates winner as 'home' or 'away' team, we need to separate in order to calculate
    output: returns a list of (Wins, Draws, Loss, Points) for this team's home or away game
    '''

    if home_or_away == 'home':
        record = [
                df.loc[(df['FTR'] == 'H')].shape[0],
                df.loc[(df['FTR'] == 'D')].shape[0],
                df.loc[(df['FTR'] == 'A')].shape[0],
                df.loc[(df['FTR'] == 'H')].shape[0]*3 + df.loc[(df['FTR'] == 'D')].shape[0]*1,]

    elif home_or_away == 'away':
        record = [
                df.loc[(df['FTR'] == 'A')].shape[0],
                df.loc[(df['FTR'] == 'D')].shape[0],
                df.loc[(df['FTR'] == 'H')].shape[0],
                df.loc[(df['FTR'] == 'A')].shape[0]*3 + df.loc[(df['FTR'] == 'D')].shape[0]*1,]
    else:
        print("please specify home or away")
        return()

    return(record)


def calc_stats(df, home_or_away):

    '''
    input:
        df = dataframe containing individual game results for this team's home (or away) games
        home_or_away = string, denoting whether df contains games played at home or away
    purpose: tabulate stats (Corner Kicks, Fouls, etc) for the games passed in
        note: Because dataset designates stats as 'home' or 'away' team, we need to separate in order to calculate
    output: returns a dataframe of (Shots, Shots on Target, Fouls Committed, Fouls Awarded (based on other team's fouls committed), Corners, Yellows, and Red Cards) for this team's home or away game
    '''

    if home_or_away == 'home':
        stats = df[['Season','HomeTeam','AwayTeam','FTR','HS','HST','HF','AF','HC','HY','HR','FTHG']]
        stats.columns = ['Season','Team', 'Opponent', 'FTR', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards', 'GoalsScored']
        stats = stats.assign(Location = 'Home')
        # Keep track of who won via points earned
        stats.loc[stats.FTR=='H', 'FTR'] = 3
        stats.loc[stats.FTR=='D', 'FTR'] = 1
        stats.loc[stats.FTR=='A', 'FTR'] = 0


    elif home_or_away == 'away':
        stats = df[['Season','AwayTeam','HomeTeam','FTR','AS','AST','AF','HF','AC','AY','AR', 'FTAG']]
        stats.columns = ['Season','Team', 'Opponent', 'FTR', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards', 'GoalsScored']
        stats = stats.assign(Location = 'Away')
        # Keep track of who won via points earned
        stats.loc[stats.FTR=='H', 'FTR'] = 0
        stats.loc[stats.FTR=='D', 'FTR'] = 1
        stats.loc[stats.FTR=='A', 'FTR'] = 3

    else:
        print("please specify home or away")
        return()

    stats.rename(columns={'FTR':'PointsEarned'}, inplace=True)
    return(stats)

def team_df(teamname, year, df):
    '''
    input:
        teamname = string, e.g. "Tottenham"
        year = string, e.g. 2017-18
        df = dataframe containing individual game results for all teams over whole season
    purpose: drop all games $teamname wasn't in,then calculate home, away, and overall record
    output: record_output is a list containing teamname, year, season record, home record, away record
    output: allstats is a dataframe with all stats for that season
    '''
    info = [teamname,year]

    # create dataframes for each team: home games, away games
    home = df.loc[(df['HomeTeam'] == teamname)]
    away = df.loc[(df['AwayTeam'] == teamname)]


    # homerecord looks like this: (13, 4, 2, 43) (homewins, homedraws, homeloss, homepoints)
    # No longer needed now that stats can easily calculate
    homerecord = calc_record(home, 'home')
    awayrecord = calc_record(away, 'away')

    # calculate the stats
    homestats = calc_stats(home, 'home')
    awaystats = calc_stats(away, 'away')
    #combine stats home,away dataframes into one allstats dataframe
    frames = [homestats, awaystats]
    allstats = pd.concat(frames) # now will allstats need the 'season' info?

    # add home and away records to create season record
    seasonrecord = []
    for i in range(0,4):
        seasonrecord.append(homerecord[i] + awayrecord[i])

    # return all above info in a list
    record_output = []
    record_output.extend(info + seasonrecord + homerecord + awayrecord)
    #return(output)
    return(allstats, record_output)

def clean(seasonstats_here):
    '''input: seasonstats dataframe from any season
    purpose: clean up by performing the steps below
    output: seasonstats dataframe
    '''

    # SeasonStats: reset the column order; through the append process it gets changed
    seasonstats_here = seasonstats_here[stats_cols]

    # Move gamedate to column, then reindex
    seasonstats_here.reset_index(level=0, inplace=True)
    #seasonstats_here['GameDate'] = seasonstats_here.index
    seasonstats_here.rename(columns={'index':'GameDate'}, inplace=True)

    # Records: index by team and sort by most points
    #team_records.set_index('Team', inplace=True)
    #team_records.sort_values(by='Points', ascending=False, inplace=True)

    # Change numerical columns to float, in preparation for finding correlation
    for i in range(5,len(seasonstats_here.columns)):
        seasonstats_here[seasonstats_here.columns[i]] = seasonstats_here[seasonstats_here.columns[i]].astype('float64')

    return(seasonstats_here)

def normalize(seasonstats_df):
    ''' input: seasonstats dataframe.  this function makes assumptions about index values
    purpose: Normalize the numeric values of the dataframe
    output: dataframe with all numeric values replaced by normalized values
    '''

    #### DAN TO TEST OHE here
    # one hot encoding for location, a potentially important factor in correlation
#    seasonstats_df = pd.concat([seasonstats_df, pd.get_dummies(seasonstats_df['Location'], prefix='Location')], axis=1)
#    seasonstats_df.drop(['Location'], axis = 1, inplace=True)

    # separate out numeric columns to be normalized (makes assumptions about shape of seasonstats df)
    numeric_columns = seasonstats_df.iloc[:,5:15]

    # normalize them
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(numeric_columns)
    df_normalized = pd.DataFrame(np_scaled)

    #put back together with first 3 columns of seasonstats
    normstats = pd.concat([seasonstats_df.iloc[:,0:5], df_normalized], axis=1, sort=False)
    normstats_cols = ['GameDate', 'Team', 'Opponent', 'PointsEarned', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards', 'GoalsScored', 'Location_Away' , 'Location_Home']
    normstats.columns = normstats_cols
    return(normstats)


# Create empty dataframe to store all team records
record_cols = ['Team', 'Season', 'Win', 'Draw', 'Loss', 'Points', 'HomeWin', 'HomeDraw', 'HomeLoss', 'HomePoints', 'AwayWin', 'AwayDraw', 'AwayLoss', 'AwayPoints']
team_records = pd.DataFrame(columns = record_cols)

stats_cols = ['Team', 'Season', 'Opponent', 'Location','PointsEarned', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards', 'GoalsScored']
seasonstats = pd.DataFrame(columns = stats_cols)

# Read in the csv file inputs.  specify 'single' or 'all' for how many seasons to get
epl = get_csv('all')

seasons = {} # creates a dict to store each season's dataframe
#campaigns = ['2017-18','2018-19']
campaigns = epl.Season.unique().tolist()

for c in campaigns:
        # df with all of this season's games as input
        this_df = epl.loc[(epl['Season'] == c)]

        # dict: key is this season, later to add value of this season's stats
        seasons[c] = pd.DataFrame(columns = stats_cols)
        print('processing Season: ', c)

        # Process each team's record:
        teamnames = this_df.HomeTeam.unique()
        #teamnames = ['Tottenham', 'Chelsea']
        teamnames = np.sort(teamnames, axis=0)
        print('teamnames for season',c,' :',teamnames)
        for name in teamnames:
            print('processing team:',name, 'for Season: ', c)
            stat, record = team_df(name, c, this_df)

            # append each list into a new DF.  Using len(team_records) to determine next empty row
            team_records.loc[len(team_records)] = record

            # for this season, combine all team stats dataframes into one season dataframe.
            stat = clean(stat)
            seasons[c] = seasons[c].append(stat)[stat.columns.tolist()] # keeps column order

# Now we can refer to any season's data set as e.g. allseasons['2017-18']
# We also want to make a single dataset for all time:

allseasons = pd.DataFrame(columns = stat.columns.tolist())
for key, value in seasons.items():
    allseasons = allseasons.append(value)
print(allseasons.shape)

# Normalize the allseasons  - troubleshoot this next
#allseasons = normalize(allseasons)

#''' At this point we have SeasonStats for all teams for one season.  Now to normalize, sort, and reindex '''
#raw_seasonstats = seasonstats # saving this for future use with raw numbers
#seasonstats = clean(seasonstats)
#seasonstats = normalize(seasonstats)



''' Now seasonstats is normalized, sorted, reindexed, and ready to process correlations '''
seasonstats.to_csv('seasonstats_normalized.csv', index = True) #export to evaluate in R or Jupyter

correlations = allseasons.corr(method='pearson')
print(correlations['PointsEarned'])

pearson_coef, p_value = stats.pearsonr(allseasons['PointsEarned'], allseasons['Shots_on_Target'])
print("The Pearson Correlation Coefficient for PointsEarned to Shots_on_Target is",
      pearson_coef, " with a P-value of P =", p_value)
