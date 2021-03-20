#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:40:41 2020

@author: vikrb
"""

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import operator

anime = pd.read_csv('/Users/vikrb/Downloads/anime-recommendations-database/anime.csv')
ratings = pd.read_csv('/Users/vikrb/Downloads/anime-recommendations-database/rating.csv')

# The dataset represents missing ratings with -1. Replacing them with nan so
# as to take an unbiased mean
ratings.rating.replace({-1: np.nan}, regex=True, inplace = True)
ratings.head()
# We will focus on TV shows as that is the most popular type    
TVanime = anime[anime['type']=='TV']
TVanime.head()

# Merge the datasets on anime_id
merged_df = ratings.merge(TVanime, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
merged_df.rename(columns = {'rating_user':'user_rating'}, inplace = True)

# For computing reasons I'm limiting the dataframe length to 15,000 users
merged_df=merged_df[['user_id', 'name', 'user_rating']]
merged_subdf= merged_df[merged_df.user_id <= 15000]
merged_subdf.head()

#Create a matrix of Users vs Animes wih User Ratings as the values
piv_table = merged_subdf.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
print(piv_table.shape)
piv_table.head()

# Standardization is being done here.
# All users with only one rating or who had rated everything the same will be dropped

# Normalize the values
norm_piv_table = piv_table.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

# Drop all columns containing only zeros. These represent users who did not rate
norm_piv_table.fillna(0, inplace=True)
norm_piv_table = norm_piv_table.T
norm_piv_table = norm_piv_table.loc[:, (norm_piv_table != 0).any(axis=0)]

# Our data needs to be in a sparse matrix format to be read by the following functions
sparse_matrix = sp.sparse.csr_matrix(norm_piv_table.values)

#Calculate item-item similarity and user-user similarity
item_similarity = cosine_similarity(sparse_matrix)
user_similarity = cosine_similarity(sparse_matrix.T)

# Convert the similarity matrices into dataframes
item_similarity_df = pd.DataFrame(item_similarity, index = norm_piv_table.index, columns = norm_piv_table.index)
user_similarity_df = pd.DataFrame(user_similarity, index = norm_piv_table.columns, columns = norm_piv_table.columns)

# Function to return the top 10 shows based on cosine similarity values
def top_animes(anime):
    i = 1
    print('Similar shows to {} include:\n'.format(anime))
    for item in item_similarity_df.sort_values(by = anime, ascending = False).index[1:11]:
        print('No. {}: {}'.format(i, item))
        i +=1  

# Function to return the top 5 users 
def top_users(user): 
    if user not in norm_piv_table.columns:
        return('No data available on user {}'.format(user))
    
    print('Most Similar Users:\n')
    cos_sim = user_similarity_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:11]
    sim_users = user_similarity_df.sort_values(by=user, ascending=False).index[1:11]
    pairData = zip(sim_users, cos_sim)
    for u, s in pairData:
        print('User #{0}, Similarity value: {1:.2f}'.format(u, s)) 

# Function to construct a list of lists containing the highest rated shows per similar user
# and return the Anime title along with its frequency of appearance
def similar_user_recs(user):
    if user not in norm_piv_table.columns:
        return('No data available on user {}'.format(user))
    
    sim_users = user_similarity_df.sort_values(by=user, ascending=False).index[1:11]
    best = []
    most_frequent = {}
    
    for i in sim_users:
        max_cos_sim = norm_piv_table.loc[:, i].max()
        best.append(norm_piv_table[norm_piv_table.loc[:, i]==max_cos_sim].index.tolist())
    for i in range(len(best)):
        for j in best[i]:
            if j in most_frequent:
                most_frequent[j] += 1
            else:
                most_frequent[j] = 1
    sorted_list = sorted(most_frequent.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list[:5]    

# Function to calculate the weighted average of similar users
# Determines a potential rating for an input user and show
def predicted_rating(anime, user):
    sim_users = user_similarity_df.sort_values(by=user, ascending=False).index[1:1000]
    user_cos_sim = user_similarity_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:1000]
    ratings = []
    weights = []
    for j, i in enumerate(sim_users):
        rating = piv_table.loc[i, anime]
        similarity = user_cos_sim[j]
        if np.isnan(rating):
            continue
        else:
            ratings.append(rating*similarity)
            weights.append(similarity)
    return sum(ratings)/sum(weights)

# Check the functions
top_animes('Naruto')

top_users(2)

# List of every show watched by user 2
watched = piv_table.T[piv_table.loc[2,:]>0].index.tolist()

# Make a list of the squared errors between actual and predicted value
errors = []
for i in watched:
    actual=piv_table.loc[2, i]
    predicted = predicted_rating(i, 2)
    errors.append((actual-predicted)**2)

# This is the average squared error for user 2
np.mean(errors)