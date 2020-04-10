import numpy as np
import config as cfg
from tqdm import tqdm
import os
# from sklearn.cluster import KMeans, MeanShift
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


vectorDir = cfg.vectorDir
dataList = os.listdir(vectorDir)
len(dataList)

vectors = np.array([np.load(os.path.join(vectorDir, file)) for file in tqdm(dataList)])

wcss = []
for i in tqdm(range(1, 20)):
    kmeans = KMeans(n_clusters=i).fit(vectors)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=6).fit(vectors)
clustSenters = kmeans.cluster_centers_
ykmeans = kmeans.labels_
