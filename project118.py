import pandas as pd
import matplotlib.pyplot as plp
import seaborn as sns
from  sklearn.cluster import KMeans
import plotly.express as px
import numpy as np

data = pd.read_csv('stars.csv')
#px.scatter(data,x='Size',y='Light').show()

x = data.iloc[:,[0,1]].values
#print(x)
sum = []
for i in range(1,11):
    k = KMeans(n_clusters=i,init='k-means++',random_state=42)
    k.fit(x)
    sum.append(k.inertia_)

plp.figure(figsize=(7,6))
sns.lineplot(sum,marker='o',color='red')
plp.xlabel('Clusters')
plp.ylabel('Inertia')
plp.title('Elbow Method')
plp.show()

d = KMeans(n_clusters=3,init='k-means++',random_state=42)
f = d.fit_predict(x)

plp.figure(figsize=(7,6))
sns.scatterplot(x=x[f==0,0],y=x[f==0,1],color='red',label='cluster1')
sns.scatterplot(x=x[f==1,0],y=x[f==1,1],color='blue',label='cluster2')
sns.scatterplot(x=x[f==2,0],y=x[f==2,1],color='green',label='cluster3')
sns.scatterplot(x=k.cluster_centers_[:,0],y=k.cluster_centers_[:,1],color='yellow',label='centroids',marker='D',s=100)
plp.xlabel('Size')
plp.ylabel('Light')
plp.title('Clusters')
plp.show()