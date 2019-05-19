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
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
pd.set_option('display.width', 700)
pd.set_option('display.max_columns',21)

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
   
    return(record)
    
    
def calc_stats(team_df, home_or_away):

    ''' 
    input: 
        team_df = dataframe containing individual game results for this team's home (or away) games
        home_or_away = string, denoting whether team_df contains games played at home or away
    purpose: tabulate stats (Corner Kicks, Fouls, etc) for the games passed in
        note: Because dataset designates stats as 'home' or 'away' team, we need to separate in order to calculate
    output: returns a dataframe of (Shots, Shots on Target, Fouls Committed, Fouls Awarded (based on other team's fouls committed), Corners, Yellows, and Red Cards) for this team's home or away game
    '''
    
    if home_or_away == 'home':
        stats = team_df[['HomeTeam','AwayTeam','FTR','HS','HST','HF','AF','HC','HY','HR']]
        stats.columns = ['Team', 'Opponent', 'FTR', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards']
        stats = stats.assign(Location = 'Home')
        # Keep track of who won via points earned
        stats.loc[stats.FTR=='H', 'FTR'] = 3
        stats.loc[stats.FTR=='D', 'FTR'] = 1
        stats.loc[stats.FTR=='A', 'FTR'] = 0


    elif home_or_away == 'away':
        stats = team_df[['AwayTeam','HomeTeam','FTR','AS','AST','AF','HF','AC','AY','AR']]
        stats.columns = ['Team', 'Opponent', 'FTR', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards']
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
    output: a list containing teamname, year, season record, home record, away record 
    '''
    info = [teamname,year]
    
    # create dataframes for each team: home games, away games
    #all = epl.loc[(epl['HomeTeam'] == teamname) | (epl['AwayTeam'] == teamname), :]
    home = epl.loc[(epl['HomeTeam'] == teamname)]
    away = epl.loc[(epl['AwayTeam'] == teamname)]


    # homerecord looks like this: (13, 4, 2, 43) (homewins, homedraws, homeloss, homepoints)
    # No longer needed now that stats can easily calculate
    homerecord = calc_record(home, 'home')    
    awayrecord = calc_record(away, 'away')   
    
    # calculate the stats now
    homestats = calc_stats(home, 'home')
    awaystats = calc_stats(away, 'away')
    #combine stats home,away dataframes into one allstats dataframe
    frames = [homestats, awaystats]
    allstats = pd.concat(frames)
    
    # add home and away records to create season record
    seasonrecord = []
    for i in range(0,4):
        seasonrecord.append(homerecord[i] + awayrecord[i])
    
    # return all above info in a list     
    output = []
    output.extend(info + seasonrecord + homerecord + awayrecord)    
    #return(output)
    return(allstats, output)

year = '2017-18'
teamnames = epl.HomeTeam.unique()
teamnames = np.sort(teamnames, axis=0)

# Create dataframe to store all team records
record_cols = ['Team', 'Season', 'Win', 'Draw', 'Loss', 'Points', 'HomeWin', 'HomeDraw', 'HomeLoss', 'HomePoints', 'AwayWin', 'AwayDraw', 'AwayLoss', 'AwayPoints']
team_records = pd.DataFrame(columns = record_cols)

stats_cols = ['Team', 'Opponent', 'Location','PointsEarned', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards']
seasonstats = pd.DataFrame(columns = stats_cols)

#for name in ['Tottenham', 'Chelsea']:
for name in teamnames:
    stat, out = team_df(name, year, epl)

    # append each list into a new DF.  Using len(team_records) to determine next empty row
    team_records.loc[len(team_records)] = out

    #combine all team stats dataframes into one all season dataframe
    seasonstats = seasonstats.append(stat)


# SeasonStats: reset the column order; through the append process it gets changed
seasonstats = seasonstats[stats_cols]

''' At this point we have SeasonStats for all teams for one season.  Now to normalize, sort, and reindex '''

# Move gamedate to column, then reindex
seasonstats.reset_index(level=0, inplace=True)
#seasonstats['GameDate'] = seasonstats.index
seasonstats.rename(columns={'index':'GameDate'}, inplace=True)

# Records: index by team and sort by most points
team_records.set_index("Team", inplace=True)
team_records.sort_values(by='Points', ascending=False, inplace=True)

# Change numerical columns to float, in preparation for finding correlation 
for i in range(4,len(seasonstats.columns)):
    seasonstats[seasonstats.columns[i]] = seasonstats[seasonstats.columns[i]].astype('float64')

# one hot encoding for location, a potentially important factor in correlation
seasonstats = pd.concat([seasonstats, pd.get_dummies(seasonstats['Location'], prefix='Location')], axis=1)
seasonstats.drop(['Location'], axis = 1, inplace=True)

def normalize(seasonstats_df):
    ''' input: seasonstats dataframe.  this function makes assumptions about index values
    purpose: Normalize the numeric values of the dataframe
    output: dataframe with all numeric values replaced by normalized values
    '''
    # separate out numeric columns to be normalized (makes assumptions about shape of seasonstats df)
    numeric_columns = seasonstats_df.iloc[:,3:14]
    
    # normalize them
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(numeric_columns)
    df_normalized = pd.DataFrame(np_scaled)
    
    #put back together with first 3 columns of seasonstats
    normstats = pd.concat([seasonstats_df.iloc[:,0:3], df_normalized], axis=1, sort=False)
    normstats_cols = ['GameDate', 'Team', 'Opponent', 'PointsEarned', 'Shots', 'Shots_on_Target', 'FoulsCommitted', 'FoulsAwarded', 'Corners', 'YellowCards', 'RedCards', 'Location_Away' , 'Location_Home']
    normstats.columns = normstats_cols
    return(normstats)
    
# Normalize the Values in prep for correlations
raw_seasonstats = seasonstats # saving this for future use with raw numbers
seasonstats = normalize(seasonstats)

''' At this point seasonstats is normalized, sorted, reindexed, and ready to process correlations '''
# Taking an export here for JUPYTER NOTEBOOKS or R to evaluate:
seasonstats.to_csv('seasonstats_normalized.csv', index = True)
# Continue on to evaluate in this script

correlations = seasonstats.corr(method='pearson')
print(correlations)

pearson_coef, p_value = stats.pearsonr(seasonstats['PointsEarned'], seasonstats['Shots_on_Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

#####  DAN: maybe style doesn't work in spyder
''' Steps:
    0. normalize data?!
1. show overall correlation matrix.  must be formated correctly
2. drill in on important ones: SOT, shots, location
3. Find and explain pearson_coef and P-value
4. expand to more seasons.  track the winner from each season to see if different correlations
'''
    
    

'''    PLOTTING the correlations '''
simple_correlations = seasonstats.corr()['PointsEarned'].sort_values(ascending=True)
simple_correlations.drop(['PointsEarned'], axis=0, inplace=True)  # Drop the redundant self-correlation
simple_correlations.plot(kind='barh', 
                         figsize=(10,5),
                         color= 'green',
                         fontsize = 12,
                         xlim=(-1,1))
plt.title('Correlation with Points Earned', fontsize=15)
plt.show()

sot_correlations = seasonstats.corr()['Shots_on_Target'].sort_values(ascending=True)
sot_correlations.drop(['Shots_on_Target','PointsEarned'], axis=0, inplace=True)  # Drop the redundant self-correlation
sot_correlations.plot(kind='barh', 
                         figsize=(10,5),
                         color= 'green',
                         fontsize = 12,
                         xlim=(-1,1))

plt.title('Correlation with Shots_on_Target', fontsize=15)
plt.xticks=scale
plt.show()


# scatter plotting correlations
#plt.scatter(epl.HC, epl.HS, color='red')
plt.scatter(raw_seasonstats.Shots_on_Target, raw_seasonstats.Shots, 
            color='red')
plt.xlabel("Shots_on_Target", fontsize=15)
plt.ylabel("Shots", fontsize=15)
plt.title('Correlation between Shots and Shots_on_Target', fontsize=15)
plt.rcParams["figure.figsize"] = (5,5)
#print('\n\n----size',plt.rcParams["figure.figsize"])
plt.show()




# Question:  why does python show corr of SOT as .472 while R shows its coefficient as .474
#Trying LinearRegression
print('\n start linear regression section \n')
sot = seasonstats[['PointsEarned','Shots_on_Target']]
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm

X = seasonstats[['Shots_on_Target']]
Y = seasonstats[['PointsEarned']]

# fit the model
lm.fit(X,Y)

#output a prediction
yhat = lm.predict(X)
print('intercept is ', lm.intercept_, 'slope is ', lm.coef_)
print('our equation is Y-hat = ',lm.intercept_, '+', lm.coef_, '* Shots on Target')

# scatter matrix
#from pandas.plotting import scatter_matrix
#pd.plotting.scatter_matrix(correlations, figsize=(12,12))
#plt.show()
# shows too much detail

#plt.matshow(correlations.corr())
#plt.colorbar(correlations, fraction=0.046, pad=0.04)
#plt.xticks(range(len(correlations.columns)), correlations.columns)
#plt.yticks(range(len(correlations.columns)), correlations.columns)




# find correlation for each stat after PointsEarned  / unneeded since corr() can do all at once
#def find_correlations(seasonstats):
#    correlations = {}    
#    for i in range(5,len(seasonstats.columns)):
#        c = seasonstats['PointsEarned'].corr(seasonstats[seasonstats.columns[i]])
#        correlations[seasonstats.columns[i]] = c
#    return(correlations)
#
#print(find_correlations(seasonstats))




# More exploration to do here; looking for interesting data to correlate, such as comparing home/away record
#team_records['Home_Away_Points_Ratio'] = team_records['HomePoints'] / team_records['AwayPoints']




# Now Visualize 
# Setup a bar chart showing TotalPoints
'''
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
   
    
# scatter plotting correlations
plt.scatter(epl.HC, epl.HS, color='red')
plt.xlabel("home corner kicks")
plt.ylabel("Home shots")
plt.show()
'''