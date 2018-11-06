# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:28:32 2018

@author: akond
"""

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

#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('DataPoints')
plt.ylabel('Euclidean Distance')


#Applying Hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

datapoints1=0
datapoints2=0
datapoints3=0



for i in range(len(y_hc)):
    if(y_hc[i]==0):
        datapoints1+=1
    elif(y_hc[i]==1):
        datapoints2+=1
    else:
        datapoints3+=1
        
        
print(datapoints1)  
print(datapoints2)  
print(datapoints3)  


#Visulasing clusters
plt.scatter(X[y_hc==0, 0],X[y_hc==0, 1],c='red',label='Cluster1')
plt.scatter(X[y_hc==1, 0],X[y_hc==1, 1],c='green',label='Cluster2')
plt.scatter(X[y_hc==2, 0],X[y_hc==2, 1],c='blue',label='Cluster3')
plt.title('Hierarchial Clustering')
plt.legend()

#DB Index
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
davies_bouldin_score(X, y_hc)


#silhouette index 
metrics.silhouette_score(X, y_hc, metric='euclidean')



from sklearn.metrics.cluster import homogeneity_score
homogeneity_score(y, y_hc)  #0.64(n=3)

from sklearn.metrics.cluster import completeness_score
completeness_score(y, y_hc)   #0.32(n=3)


from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score(y,y_hc) #0.45(n=3)

























