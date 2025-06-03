# Customer Segmentation App

This is a Streamlit web app for customer segmentation using KMeans clustering.  
Users can upload a dataset, select features (or use PCA), and visualize clusters interactively.

## Features

- Upload CSV dataset for clustering
- Automatic or manual feature selection
- Dimensionality reduction with PCA option
- Elbow method visualization to choose optimal number of clusters
- Interactive cluster visualization (2D scatter plot)
- Cluster summary statistics
- Download clustered data as CSV

## How to Use

1. Upload your customer dataset in CSV format.
2. Choose feature selection mode:
   - **Auto**: Use all numeric features automatically.
   - **Manual**: Select numeric features manually.
   - **Use PCA**: Apply Principal Component Analysis to reduce dimensions.
3. View the Elbow plot to decide the number of clusters.
4. Select number of clusters (k) using the slider.
5. Explore cluster assignments and summary.
6. Download the clustered data.

## Tech Stack

- Python 3.x
- Streamlit
- Pandas
- scikit-learn
- matplotlib & seaborn

## Running Locally

```bash
pip install -r requirements.txt
streamlit run customer_segmentation.py
