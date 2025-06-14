import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()
st.write("### Raw Data", df.head())

# Data Preprocessing
df_proc = df.drop("CustomerID", axis=1)
le = LabelEncoder()
df_proc["Gender"] = le.fit_transform(df_proc["Gender"])

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_proc)

# Elbow Method
def elbow_plot(data):
    inertia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        inertia.append(km.inertia_)
    return inertia

inertia = elbow_plot(scaled_features)

# Elbow Plot Visualization
st.write("### Elbow Method to Determine Optimal Clusters")
fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), inertia, 'bo-')
ax1.set_title('Elbow Method')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
st.pyplot(fig1)

# Cluster Slider
k = st.slider("Select number of clusters (k)", 2, 10, 5)
kmeans = KMeans(n_clusters=k, random_state=42)
df_proc["Cluster"] = kmeans.fit_predict(scaled_features)
df["Cluster"] = df_proc["Cluster"]

# Cluster Summary
st.write("### Cluster-wise Averages")
st.dataframe(df.groupby('Cluster').mean(numeric_only=True))

# Matplotlib 2D Scatter Plot
st.write("### 2D Cluster Visualization")
fig2, ax2 = plt.subplots()
sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["Cluster"],
    palette="Set2",
    ax=ax2
)
ax2.set_title("Customer Segments (2D)")
st.pyplot(fig2)

# Plotly 3D Cluster Visualization
st.write("### 3D Cluster Visualization with Plotly")
fig3d = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Cluster",
    symbol="Gender",
    opacity=0.7,
    title="Customer Segments in 3D",
    color_continuous_scale='Viridis'
)
fig3d.update_layout(scene=dict(
    xaxis_title='Age',
    yaxis_title='Annual Income',
    zaxis_title='Spending Score'
))
st.plotly_chart(fig3d, use_container_width=True)
