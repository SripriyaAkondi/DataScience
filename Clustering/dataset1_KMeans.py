# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:09:04 2018

@author: akond
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('HTRU_2.csv')
X=dataset.iloc[:,0:9].values
y=dataset.iloc[:,8].values
#preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

#pca
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X=pca.fit_transform(X)

explained_variance=pca.explained_variance_ratio_
print(explained_variance)

variance=[0.52569923, 0.24414578, 0.09087955, 0.05605115, 0.03459552, 0.02865671,0.01625574, 0.00193562 ,0.0017807]
n=[1,2,3,4,5,6,7,8,9]
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
kmeans=KMeans(n_clusters=3, init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

datapoints1=0
datapoints2=0
datapoints3=0

for i in range(len(y_kmeans)):
    if(y_kmeans[i]==0):
        datapoints1+=1
    elif(y_kmeans[i]==1):
        datapoints2+=1
    else:
        datapoints9+=1
        
        
print(datapoints1)  
print(datapoints2)  
print(datapoints3)  

#Visulasing clusters
plt.scatter(X[y_kmeans==0, 0],X[y_kmeans==0, 1],c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1, 0],X[y_kmeans==1, 1],c='green',label='Cluster2')
plt.scatter(X[y_kmeans==2, 0],X[y_kmeans==2, 1],c='blue',label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',label='Centriods')
plt.title('Kmeans Clustering')
plt.legend()

#Calculate DB index ,silhouette index , Homogenity score and completeness score for the clusters
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


















