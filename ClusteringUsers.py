#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:01:31 2020

@author: vikrb
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud


#Read the anime data and display head
anime = pd.read_csv('/Users/vikrb/Downloads/anime-recommendations-database/anime.csv')
anime.head()
print('\nAnime data shape: ' + str(anime.shape))

#Read the ratings data and display head
ratings = pd.read_csv('/Users/vikrb/Downloads/anime-recommendations-database/rating.csv')
ratings.head()
print('\nRatings data shape: ' + str(ratings.shape))

#Let us look at what the mean rating of each anime is per user    
PerUserMeanRatings = ratings.groupby(['user_id']).mean().reset_index()
PerUserMeanRatings['mean_rating'] = PerUserMeanRatings['rating']
PerUserMeanRatings.drop(['anime_id','rating'],axis=1, inplace=True)
PerUserMeanRatings.head(10)

#Merging the mean rating with the user rating data
user_ratings = pd.merge(ratings,PerUserMeanRatings,on=['user_id','user_id'])
user_ratings.head(5)
print(user_ratings.shape)

#Since we are building a recommender, we are only interested in animes that the user likes
#So we will drop all entries where user_rating < user_mean_rating
user_ratings = user_ratings.drop(user_ratings[user_ratings.rating < user_ratings.mean_rating].index)
print(user_ratings.shape)

#Renaming rating to UserRating
user_ratings = user_ratings.rename({'rating':'userRating'}, axis='columns')

#------------------------------------------------------------------------------------------------------------

#Merging the 2 datasets on anime_id. Will also limit it to 20000 users due to lack of processing power
data = pd.merge(anime,user_ratings,on=['anime_id','anime_id'])
data= data[data.user_id <= 20000]
data.head(10)

len(data['anime_id'].unique())
len(anime['anime_id'].unique())
#We have already dropped 35% of the animes in this process

#Let us now create a matrix of users vs animes to see which users like which animes
userAnime = pd.crosstab(data['user_id'], data['name'])
userAnime.head(10)
userAnime.shape

#PCA is the process of transforming possibly correlated variables into linearly uncorrelated variables
pca = PCA(n_components=3)
pca.fit(userAnime)
pca_points = pca.transform(userAnime)
PrincipalComponents = pd.DataFrame(pca_points)
PrincipalComponents.head()

#Let us view the principal components in 3D
plt.rcParams['figure.figsize'] = (9, 5)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(PrincipalComponents[0], PrincipalComponents[2], PrincipalComponents[1], c = 'Grey')
plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()

#Let us now break the above into clusters of userAnimes

'''Silhouette refers to a method of interpretation and validation of consistency within clusters
of data. The technique provides a succinct graphical representation of how well each object
has been classified.'''

'''Inertia is a measure of how internally coherent the clusters are'''

Sil_scores = []
inertia_values = np.empty(8)

#Calculate Silhouette scores and inertia values for num_clusters in range (2,10)
for i in range(2,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(PrincipalComponents)
    inertia_values[i-2] = kmeans.inertia_
    Sil_scores.append(silhouette_score(PrincipalComponents, kmeans.labels_))
    
#Plotting Inertia vs num_clusters.
plt.plot(range(2,10),inertia_values,'-x')
plt.title('Inertia vs Num_Clusters')
plt.xlabel('Number of clusters')
plt.axvline(x=4, color='blue', linestyle='--')
plt.ylabel('Inertia')
plt.show()

#Plotting Sil_scores vs num_clusters. Notice the peak at num_clusters = 4
plt.plot(range(2,10), Sil_scores);
plt.title('Silhouette scores vs Num_Clusters')
plt.xlabel('Number of clusters');
plt.axvline(x=4, color='blue', linestyle='--')
plt.ylabel('Silhouette Score');
plt.show()

#Partitioning into 4 clusters.
UAClusters = KMeans(n_clusters=4,random_state=30).fit(PrincipalComponents)
centers = UAClusters.cluster_centers_
cluster_preds = UAClusters.predict(PrincipalComponents)
print(centers)

#Visualize the clusters
fig = plt.figure(figsize=(8,5))
ax = Axes3D(fig)
ax.scatter(PrincipalComponents[0], PrincipalComponents[2], PrincipalComponents[1], c = cluster_preds)
plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()

fig = plt.figure(figsize=(8,5))
plt.scatter(PrincipalComponents[1], PrincipalComponents[0],c = cluster_preds)
for ci,c in enumerate(centers):
    plt.plot(c[1], c[0], 'o', markersize=8, color='red', alpha=1)
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('Data points in 2D PCA axis', fontsize=20)
plt.show()

#Adding column to recognize which cluster the user belongs to
userAnime['cluster'] = cluster_preds
userAnime.head(10)

#Creating separate dataframes for each cluster
Cluster1 = userAnime[userAnime['cluster']==0].drop('cluster',axis=1).mean()
Cluster2 = userAnime[userAnime['cluster']==1].drop('cluster',axis=1).mean()
Cluster3 = userAnime[userAnime['cluster']==2].drop('cluster',axis=1).mean()
Cluster4 = userAnime[userAnime['cluster']==3].drop('cluster',axis=1).mean()

#Custom function for getting related Anime info
def generateAnimeInfo(animes):
    members = list()
    ratingl = list()
    episodes = list()
    genres = list()
    for ani in anime['name']:
        if ani in animes:
            members.append(anime[anime['name']==ani].members.values.astype(int))
            ratingl.append(anime[anime['name']==ani].rating.values.astype(int))
            episodes.append(anime[anime['name']==ani].episodes.values.astype(int))
            for gen in anime[anime['name']==ani].genre.values:
                 genres.append(gen)
    return members,ratingl,episodes,genres
         
#Function to calculate frequency of occurence of genres
def count_genres(df, colname, wordlist):
    word_counter = dict()
    for word in wordlist:
        word_counter[word] = 0
    for list_words in df[colname].str.split(','):        
        if type(list_words) == float and pd.isnull(list_words): continue        
        for s in [s for s in list_words if s in wordlist]: 
            if pd.notnull(s): word_counter[s] += 1
    
    # convert the dictionary to a list and sort by frequency
    word_occurences = []
    for word,freq in word_counter.items():
        word_occurences.append([word,freq])
    word_occurences.sort(key = lambda x:x[1], reverse = True)
    return word_occurences

#Function to draw the WordCloud
def generateWordCloud(Dict,title,color):
    words = dict()
    for word in Dict:
        words[word[0]] = word[1]

        wordcloud = WordCloud(
                      width=1000,
                      height=350, 
                      background_color=color, 
                      max_words=20,
                      max_font_size=350, 
                      normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)

    plt.title(title)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

#Cluster 1
animelist = list(Cluster1.index)
df = pd.DataFrame()
df['member'],df['rating'],df['episode'],df['genre'] =  generateAnimeInfo(animelist)
set_words = set()
for genre_list in data['genre'].str.split(',').values:
    if isinstance(genre_list, float): continue  # only occurs if genre_list = NaN
    set_words = set_words.union(genre_list)

Cluster1_animes = list(Cluster1.sort_values(ascending=False)[0:15].index)
c1_data = pd.DataFrame()
c1_data['member'],c1_data['rating'],c1_data['episode'],c1_data['genre'] =  generateAnimeInfo(Cluster1_animes)
c1_data.iloc[:,0:3] = c1_data.iloc[:,0:3].astype(int) # change numeric to integer
word_occurences = count_genres(c1_data, 'genre', set_words)
generateWordCloud(word_occurences[0:15],"Cluster 1","lemonchiffon")

#Cluster 2
animelist = list(Cluster2.index)
df = pd.DataFrame()
df['member'],df['rating'],df['episode'],df['genre'] =  generateAnimeInfo(animelist)
set_words = set()
for genre_list in data['genre'].str.split(',').values:
    if isinstance(genre_list, float): continue
    set_words = set_words.union(genre_list)

Cluster2_animes = list(Cluster2.sort_values(ascending=False)[0:15].index)
c2_data = pd.DataFrame()
c2_data['member'],c2_data['rating'],c2_data['episode'],c2_data['genre'] =  generateAnimeInfo(Cluster2_animes)
c2_data.iloc[:,0:3] = c2_data.iloc[:,0:3].astype(int) 
word_occurences = count_genres(c2_data, 'genre', set_words)
generateWordCloud(word_occurences[0:15],"Cluster 2","black")