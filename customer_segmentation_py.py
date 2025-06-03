import io
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

@st.cache_data
def load_data(file):
  try:
    if file.name.endswith('.csv'):
      return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
      return pd.read_excel(file)
    elif file.name.endswith('.json'):
      return pd.read_json(file)
    else:
      st.error("Unsupported file format. Please upload a CSV, Excel (.xlsx), or JSON file.")
    return None
  except Exception as e:
    st.error(f"Error loading file: {e}")
    return None

st.set_page_config(page_title="Customer Segmentation App", page_icon=":bar_chart:", layout="wide")
st.title("Customer Segmentation App")
st.write("Upload the Document")

uploaded_file=st.file_uploader("Upload File",type=['csv'])

def show_data_summary(df):
  st.subheader("Dataset Preview")
  st.dataframe(df)
  st.write("Shape:", df.shape)
  st.write("Columns:", df.columns.tolist())
  st.write("Missing Values:")
  st.dataframe(df.isnull().sum())

X_scaled = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    show_data_summary(df)

    mode = st.radio("Choose a mode for feature selection:", ["Auto", "Manual", "Use PCA"])

    if mode == "Auto":
        st.markdown("Automatically selecting all numeric features.")
        numeric_features = df.select_dtypes(include=["number"])
        features_used = numeric_features.columns.tolist()
        X = numeric_features

    elif mode == "Manual":
        st.markdown("Manually select numeric features for clustering.")
        all_numeric = df.select_dtypes(include=["number"]).columns.tolist()

        features_used = st.multiselect("Select at least 2 numeric features:", all_numeric, default=all_numeric[:2])

        if len(features_used) < 2:
            st.warning("Please select at least 2 features.")
            X = None
        else:
            X = df[features_used]

    elif mode == "Use PCA":
        st.markdown("Using Principal Component Analysis (PCA) to reduce features.")
        numeric_features = df.select_dtypes(include=["number"])
        features_used = numeric_features.columns.tolist()
        X_raw = numeric_features

        # Scale before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        pca = PCA(n_components=2)
        X = pca.fit_transform(X_scaled)
        st.info("PCA applied. Clustering will be based on first two components.")

    else:
        st.warning("Please select a mode.")

    # Scaling (if needed)
    if mode in ["Auto", "Manual"] and X is not None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif mode == "Use PCA":
        X_scaled = X  # already scaled by PCA
    else:
        X_scaled = None

    if X_scaled is not None:
        st.markdown("Features used for clustering:")
        st.write(features_used)
        st.markdown("Sample data (after scaling or PCA):")
        st.dataframe(X_scaled[:5])

st.subheader("Elbow Method to select optimal k")

max_k = 10
inertias = []

if X_scaled is not None:
    for k in range(1, max_k + 1):
        kmeans_tmp = KMeans(n_clusters=k, random_state=42)
        kmeans_tmp.fit(X_scaled)
        inertias.append(kmeans_tmp.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, max_k + 1), inertias, marker='o')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)
else:
    st.info("Upload data and select features to see Elbow plot.")

if X_scaled is not None:
    st.subheader("🔗 Clustering")
    n_clusters = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = cluster_labels
    st.success("Clustering complete.")
    st.markdown("### 📋 Sample of clustered data:")
    st.dataframe(df.head())
    inertia = kmeans.inertia_
    st.write(f"Inertia (Sum of squared distances): {inertia:.2f}")

    if n_clusters > 1 and len(set(cluster_labels)) > 1:
        score = silhouette_score(X_scaled, cluster_labels)
        st.write(f"Silhouette Score: {score:.2f}")
    else:
        st.warning("Silhouette Score requires at least 2 clusters with more than one point.")

    # Optional visualization
    if X_scaled.shape[1] == 2:
        st.subheader("📊 Cluster Visualization (2D)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap="viridis", s=50)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("KMeans Clusters")
        st.pyplot(fig)
else:
    st.info("ℹClustering will be enabled once valid features are selected.")


if 'df' in st.session_state:
    df = st.session_state['df']
    csv = df.to_csv(index=False)
    st.download_button(
      label="Download clustered data as CSV",
      data=csv,
      file_name='clustered_data.csv',
      mime='text/csv'
    )

if X_scaled is not None and 'Cluster' in df.columns:
    st.subheader("Cluster Profile Summary")
    cluster_summary = df.groupby("Cluster")[features_used].mean()
    st.dataframe(cluster_summary.style.background_gradient(cmap='viridis'))

