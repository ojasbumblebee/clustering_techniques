#!/usr/bin/env python
# -*- coding: utf-8 -*-
# importing libraries
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math
import sys
import os
import scipy.cluster.hierarchy as sch
import plotly
import plotly.graph_objs as go
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

# loading the dataset
dataset = pd.read_csv('ovbarve.csv')
X = dataset.iloc[:, [0, 1, 2]].values


#Hierarchial clustering Dendrogram
os.chdir("results")
Z = sch.linkage(X, method='ward')
den = sch.dendrogram(Z)
plt.title('Dendrogram for the clustering of the dataset')
plt.xlabel('datapoints')
plt.ylabel('Euclidean distance in the space with 3 dimensions');
plt.savefig("dendogram.png")
plt.show()

def getTrace(x, y, z, c, label, s=2):
    trace_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return trace_points


def showGraph(title, x_colname, x_range, y_colname, y_range, z_colname, z_range, traces, clustering_techinique):
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=x_colname, range=x_range),
            yaxis=dict(title=y_colname, range=y_range),
            zaxis=dict(title=z_colname, range=z_range)
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.plot(fig, filename=clustering_techinique+'.html')

#Agglomerative Clustering with number of clusters obtained from dendrogram
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# visualising the clusters
t1 = getTrace(X[y_hc == 0, 0], X[y_hc == 0, 1], X[y_hc == 0, 2], s=4, c='red', label='1')  #
t2 = getTrace(X[y_hc == 1, 0], X[y_hc == 1, 1], X[y_hc == 1, 2], s=4, c='green', label='2')  #
t3 = getTrace(X[y_hc == 2, 0], X[y_hc == 2, 1], X[y_hc == 2, 2], s=4, c='blue', label='3')  #

x = X[:, 0]
y = X[:, 1]
z = X[:, 2]
showGraph("Clustering results of the dataset represented in the 3D space with 3 dimensions for X, Y and Z", "X",
 [min(x),max(x)], "Y", [min(y),max(y)], "Z", [min(z)-1,max(z)], [t1,t2,t3], clustering_techinique="hierarchical")

# k-means Algorithm
# determine k using the elbow method by plotting the Sum of Square Errors (SSE) v/s the no. of clusters
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('The Elbow Method showing the optimal k')
plt.savefig("K-means_elbow.png")
plt.show()

#Clustering data based on the value of k obtained using the elbow method
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=100, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# visualising the clusters
centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=8,
                     c='yellow', label='Centroids')
t1 = getTrace(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s=4, c='red',
              label='1')  # match with red=1 initial class
t2 = getTrace(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s=4, c='black',
              label='2')  # match with black=3 initial class
t3 = getTrace(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s=4, c='blue',
              label='3')  # match with blue=2 initial class

x = X[:, 0]
y = X[:, 1]
z = X[:, 2]

showGraph("CLustering based on K means represented in the 3D space", "X",
 [min(x),max(x)], "Y", [min(y),max(y)], "Z", [min(z)-1,max(z)], [t1,t2,t3,centroids], clustering_techinique="k-means")


#DBSCAN algorithm for clustering
#Below we Show the elbow for different values of Min points = 3,4,5,6 
#we obtain the eps value for each of the minpoints values defined above 
def best_eps_for_DBSCAN(X, min_pts):
    data = np.array(X)
    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(data)
    distances, indices = nbrs.kneighbors(data)
    distanceDec = sorted(distances[:, min_pts - 1], reverse=True)
    plt.plot(indices[:, 0], distanceDec)
    plt.title('DBSCAN Elbow graph for Minpoints:'+str(min_pts))
    plt.savefig("DBSCAN_elbow_minpoints"+str(min_pts)+".png")
    plt.show()

for min_point in [3,4,5,6]:
    best_eps_for_DBSCAN(X, min_pts=min_point)

#Cluster using dbscan algorithm with min points = 3 and eps = 22
dbscan = DBSCAN(eps=18, metric='euclidean', min_samples=5)
dbscan.fit(np.array(X))

fig = pyplot.figure()
ax = Axes3D(fig)
_x = dataset.iloc[:, [0]].values
_y = dataset.iloc[:, [1]].values
_z = dataset.iloc[:, [2]].values
print(dbscan.labels_)
ax.scatter(_x, _y, _z, c=dbscan.labels_)
plt.title('DBSCAN clustering min_points=5 eps=18')  
plt.savefig("DBSCAN_min_points_5_eps_18.png")
pyplot.show()


#Gaussian Mixture Models
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]

plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.title('AIC and BIC curves for GMM')  
plt.savefig("AIC_and_BIC_curves_GMM.png")
plt.show()

gmm = GaussianMixture(3, covariance_type='full', random_state=0).fit(X)
labels = gmm.predict(X)
fig = pyplot.figure()
ax = Axes3D(fig)
_x = dataset.iloc[:, [0]].values
_y = dataset.iloc[:, [1]].values
_z = dataset.iloc[:, [2]].values
ax.scatter(_x, _y, _z, c=labels)
plt.title('GMM clustering for 3 clusters')  
plt.savefig("Gaussian_Mixture_models_k_3.png")
pyplot.show()

color = []
for label in labels:
    if label == 0:
        color.append("red")
    elif label == 1:
        color.append("blue")
    elif label == 2:
        color.append("green")

plt.scatter(_z, _y, c=color)
plt.title('GMM clustering for 3 clusters represented in 2 dimension z v/s y')  
plt.savefig("GAussian_Mixture_models_(z vs y).png")
plt.show()

plt.scatter(_x, _y, c=color)
plt.title('GMM clustering for 3 clusters represented in 2 dimension x v/s y')  
plt.savefig("GAussian_Mixture_models_(x vs y).png")
plt.show()

plt.scatter(_x, _z, c=color)
plt.title('GMM clustering for 3 clusters represented in 2 dimension x v/s z')  
plt.savefig("GAussian_Mixture_models_(x vs z).png")
plt.show()

