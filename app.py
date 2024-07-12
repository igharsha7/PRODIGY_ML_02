import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Loading the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# Sidebar for user input
st.sidebar.header('Number of Clusters')

def user_input_features():
    clusters = st.sidebar.slider('Use the slider', 2, 10, 5)
    return clusters

n_clusters = user_input_features()

# Display the dataset
st.write("## Mall Customers Dataset")
st.write(df.head())

# Data Preprocessing and Visualization
st.write("## Data Visualization")
fig, ax = plt.subplots()
sns.histplot(df['Age'], bins=20, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.histplot(df['Annual Income (k$)'], bins=20, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.histplot(df['Spending Score (1-100)'], bins=20, ax=ax)
st.pyplot(fig)

# K-means Clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting the clusters
fig, ax = plt.subplots()
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=labels, palette='deep', ax=ax)
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='black', marker='+', s=200, ax=ax)
plt.title("Annual Income vs Spending Score")
st.pyplot(fig)

# Displaying the Silhouette Score
silhouette_avg = silhouette_score(X, labels)
st.write(f"## Silhouette Score: {silhouette_avg}")

# Displaying the centroids
st.write("## Cluster Centroids")
st.write(pd.DataFrame(centroids, columns=['Annual Income (k$)', 'Spending Score (1-100)']))
