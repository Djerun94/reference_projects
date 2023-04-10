# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:18:57 2023

@author: tobias
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


#read dataset
data = pd.read_csv("Wholesale customers data.csv")

#check for missing values
print(data.isnull().sum())

#initial exploration of data
print(data.describe())


# Visualize distribution of variables
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

sns.histplot(data['Fresh'], ax=axes[0], kde=False)
sns.histplot(data['Milk'], ax=axes[1], kde=False)
sns.histplot(data['Grocery'], ax=axes[2], kde=False)
sns.histplot(data['Frozen'], ax=axes[3], kde=False)
sns.histplot(data['Detergents_Paper'], ax=axes[4], kde=False)
sns.histplot(data['Delicassen'], ax=axes[5], kde=False)
plt.show()


# show outliers and extreme values
sns.boxplot(data=data)
plt.show()


# normalize data (z-Score norm.)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)


# find opt nr of clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_normalized)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# perform Kmeans clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(data_normalized)

data_clustered = pd.concat([data.reset_index(drop=True), pd.DataFrame({'Cluster': clusters})], axis=1)
sns.pairplot(data_clustered, hue='Cluster')

# cluster labels to initial data
data_clustered = data.copy()
data_clustered['Cluster'] = clusters

# mean values for each cluster
cluster_means = data_clustered.groupby('Cluster').mean()

print(cluster_means)