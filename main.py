import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("final.csv")

X = []

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()

mass.sort()
radius.sort()
gravity.sort()
plt.plot(radius,mass)

plt.title("Radius & Mass of the Star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.plot(mass,gravity)

plt.title("Mass vs Gravity")
plt.xlabel("Mass")
plt.ylabel("Gravity")
plt.show()

plt.scatter(radius,mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

cluster_1_x = []
cluster_1_y = []
cluster_2_x = []
cluster_2_y = []
cluster_3_x = []
cluster_3_y = []
cluster_4_x = []
cluster_4_y = []

for index, data in enumerate(X):
  if y_kmeans[index] == 0:
    cluster_1_x.append(data[0])
    cluster_1_y.append(data[1])
  elif y_kmeans[index] == 1:
    cluster_2_x.append(data[0])
    cluster_2_y.append(data[1])
  elif y_kmeans[index] == 2:
    cluster_3_x.append(data[0])
    cluster_3_y.append(data[1])
  elif y_kmeans[index] == 3:
    cluster_4_x.append(data[0])
    cluster_4_y.append(data[1])


plt.figure(figsize=(15,7))
sns.scatterplot(cluster_1_x, cluster_1_y, color = 'yellow', label = 'Cluster 1')
sns.scatterplot(cluster_2_x, cluster_2_y, color = 'blue', label = 'Cluster 2')
sns.scatterplot(cluster_3_x, cluster_3_y, color = 'green', label = 'Cluster 3')
sns.scatterplot(cluster_4_x, cluster_4_y, color = 'red', label = 'Cluster 4')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'black', label = 'Centroids',s=100,marker=',')
plt.title('Clusters of Planets')
plt.xlabel('Planet Radius')
plt.ylabel('Planet Mass')
plt.legend()
plt.gca().invert_yaxis()
plt.show()