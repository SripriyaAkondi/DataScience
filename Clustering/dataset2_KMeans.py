# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:52:57 2018

@author: akond
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('dataset2.csv')
X=dataset.iloc[:,0:5].values
y=dataset.iloc[:,4].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


#from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X=pca.fit_transform(X)

explained_variance=pca.explained_variance_ratio_
print(explained_variance)

variance=[0.50161595,0.33103104,0.1141432,0.05320981,0.01]
n=[1,2,3,4,5]
plt.plot(n,variance)
plt.scatter(n,variance)
plt.xlabel('Number of principle components')
plt.ylabel('Amount of maximum variance')


#Using Elbow method to determine the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.scatter(range(1,15),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares')


#Applying Kmeans to the dataset
kmeans=KMeans(n_clusters=9, init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

datapoints1=0
datapoints2=0
datapoints3=0



for i in range(len(y_kmeans)):
    if(y_kmeans[i]==0):
        datapoints1+=1
    else:
        datapoints2+=1
        
        
print(datapoints1)  
print(datapoints2)  
 



#Visulasing clusters
plt.scatter(X[y_kmeans==0, 0],X[y_kmeans==0, 1],c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1, 0],X[y_kmeans==1, 1],c='green',label='Cluster2')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',label='Centriods')
plt.title('Kmeans Clustering')
plt.legend()



#DB Index
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
davies_bouldin_score(X, y_kmeans) 

#silhouette index 
metrics.silhouette_score(X, y_kmeans, metric='euclidean')



from sklearn.metrics.cluster import homogeneity_score
homogeneity_score(y, y_kmeans)  

from sklearn.metrics.cluster import completeness_score
completeness_score(y, y_kmeans)  

from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score(y,y_kmeans)




