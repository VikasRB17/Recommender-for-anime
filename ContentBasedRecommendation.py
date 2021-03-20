#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:45:14 2020

@author: vikrb
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
        
#Read the data
anime = pd.read_csv('/Users/vikrb/Downloads/anime-recommendations-database/anime.csv')
ratings = pd.read_csv('/Users/vikrb/Downloads/anime-recommendations-database/rating.csv')

#Function for Dataset Exploration
def explore(df):
    print('\nShape of the data:')
    print('Num_Rows: ',df.shape[0],'\nNum_Cols: ',df.shape[1])
    print('\nColumn names:')
    print(df.columns)
    print('\nMissing Values:')
    count = df.isnull().sum()
    print(count[count>0])

#Anime dataset exploration
explore(anime)

#Ratings dataset exploration
explore(ratings)

#Data Cleaning & Preprocessing
anime['episodes'] = anime['episodes'].replace('Unknown',np.nan)
anime['episodes'] = anime['episodes'].astype(float)
common_ids = anime[anime['anime_id'].isin(ratings['anime_id'])]
common_ids['rating'].isnull().sum()

#If rating are missing for common AnimeIDs for both tables, replace with median value
for i,j in zip(common_ids[common_ids['rating'].isnull()].index,common_ids[common_ids['rating'].isnull()]['anime_id'].values):
    median_value = ratings[ratings['anime_id']==j]['rating'].median()
    print('median value: ',median_value)
    anime.loc[i,'rating'] = median_value
    print('index {} done!'.format(str(i)))

#Dropping rows that have no ratings, and also removing duplicates
anime.dropna(subset=['rating'],axis=0,inplace=True)
anime['genre']=anime['genre'].str.replace(', ',',')
anime=anime.drop_duplicates('name')

#EDA: Pie chart depicting distribution of the different anime types
plotcolors = {'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink'}
anime['type'].value_counts().plot.pie(shadow=True,figsize=(6,6),colors=plotcolors, labeldistance = 0.6)
plt.title('Anime Types')
plt.ylabel('')
plt.show()

#Outliers using box-plot
plt.figure(figsize=(8,4))
sns.boxplot(x='type',y='rating',data=anime, palette = "deep")
plt.title('Anime type VS Rating')
plt.show()

#Mean rating per type
for i in anime['type'].unique().tolist():
    print('\nMean of '+str(i)+' ratings:')
    print(anime[anime['type']==i]['rating'].mean())

#We will focus on TV shows as that is the most popular type    
#Top 15 genres in TV animes
TV_animes = anime[anime['type']=='TV']
TV_animes['genre'].value_counts().sort_values(ascending=True).tail(15).plot.barh(figsize=(6,6))
plt.title('TV Animes: Top 15 Genres')
plt.xlabel('Frequency')
plt.ylabel('Genres')
plt.show()

#Show metrics for episodes,ratings and members across all animes
TV_animes.drop('anime_id',axis=1).describe()

#TV anime that has the maximum and minimum number of episodes
TV_animes[TV_animes['episodes']==TV_animes['episodes'].max()]
TV_animes[TV_animes['episodes']==TV_animes['episodes'].min()]

#Distribution plots of rating, members
fig=plt.figure(figsize=(13,5))
for i,j in zip(TV_animes[['rating','members']].columns,range(3)):
    ax=fig.add_subplot(1,2,j+1)
    sns.distplot(TV_animes[i],ax=ax)
    plt.axvline(TV_animes[i].mean(),label='mean',color='black')
    plt.axvline(TV_animes[i].median(),label='median',color='blue')
    plt.title('{} distribtion'.format(i))
    plt.legend()
plt.show()

#Box plot for ratings and members
fig=plt.figure(figsize=(13,5))
for i,j in zip(TV_animes[['rating','members']].columns,range(3)):
    ax=fig.add_subplot(1,2,j+1)
    sns.boxplot(i,data=TV_animes,ax=ax)
    plt.title('{} distribtion'.format(i))
plt.show()

#Content-based recommendation:
#-----------------------------------------------------------------------------------------------------------------------------------------------------

#Create dummy variables for Genres
TV_animes_1=TV_animes.copy()
TV_animes_1['genre']=TV_animes_1['genre'].str.split(',')
TV_animes_1.head()
for index, glist in zip(TV_animes_1.index,TV_animes_1['genre'].values):
    for genre in glist:
        TV_animes_1.at[index, genre] = 1

#Filling in the NaN values with 0 (indicates that TV show doesn't belong to that genre)
TV_animes_1 = TV_animes_1.fillna(0)
TV_animes_1.head()

#Remove unnecessary columns to create a genre matrix
genre_matrix = TV_animes_1.set_index(TV_animes_1['anime_id'])
genre_matrix = genre_matrix.drop('anime_id', 1).drop('name', 1).drop('genre', 1).drop('episodes', 1).drop('members',1).drop('rating',1).drop('type',1)
genre_matrix.head()

#Create a user profile
user_profile=pd.DataFrame([{'name':'Shingeki no Kyojin','user_rating':8.5},
                        {'name':'Bleach','user_rating':8.3}])
user_profile

#Add other details pertaining to that Anime from the datasets
userdf = TV_animes[TV_animes['name'].isin(user_profile['name'].tolist())]
userdf = pd.merge(userdf, user_profile)
userdf = userdf.drop('genre', 1).drop('rating', 1).drop('episodes',1).drop('type',1).drop('members',1)
userdf

#Update with the genre dummy variables
user_animedf = TV_animes_1[TV_animes_1['name'].isin(userdf['name'].tolist())]
user_animedf = user_animedf.drop('rating',1)
user_animedf = user_animedf.reset_index(drop=True)
user_animedf

#Dropping unnecessary columns due to save memory
user_genre_matrix = user_animedf.drop('anime_id', 1).drop('name', 1).drop('genre', 1).drop('type', 1).drop('episodes',1).drop('members',1)
user_genre_matrix

#Perform dot product on user_genre_matrix and user_ratings. This provides a weighted preference
#list which depicts how much a use likes each genre. 0 indicates no preference.
userPref = user_genre_matrix.transpose().dot(userdf['user_rating'])
userPref

#Here we multiply each row of the genre matrix with corresponding elements in the userPref vector
#Then we calculate sum of each row of the resultant matrix. This results in a weighted sum
#across genres for every anime. We then normalize this as per use preference.
recommendation_df = ((genre_matrix*userPref).sum(axis=1))/(userPref.sum())
recommendation_df.head()

#Sort the values and show top N recommendations
recommendation_df = recommendation_df.sort_values(ascending=False)
TV_animes.loc[TV_animes['anime_id'].isin(recommendation_df.head(10).keys())]