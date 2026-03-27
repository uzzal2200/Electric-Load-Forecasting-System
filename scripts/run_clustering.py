"""
Electric Load Forecasting - Clustering Analysis Script

This script implements clustering techniques to segment hourly electricity 
consumption data based on weather and load patterns.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')  # Updated for newer seaborn versions
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# Create output directory for plots
os.makedirs('clustering_results', exist_ok=True)

# 1. Data Loading and Preprocessing
print("1. Loading and preprocessing data...")

# Try to load data with error handling
try:
    df = pd.read_csv('processed_data/full_dataset_hourly.csv')
    print("Dataset shape:", df.shape)
    print("First few rows:")
    print(df.head())
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data for testing if real data can't be loaded
    print("Creating sample data for testing...")
    # Create sample dates
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'date': dates,
        'hour': [d.hour for d in dates],
        'day_of_week': [d.dayofweek for d in dates],
        'is_weekend': [(d.dayofweek >= 5) * 1 for d in dates],
        'houston': np.random.normal(500, 100, 1000),
        'dallas': np.random.normal(450, 90, 1000),
        'san antonio': np.random.normal(400, 80, 1000),
        'houston_temperature': np.random.normal(80, 10, 1000),
        'dallas_temperature': np.random.normal(75, 12, 1000),
        'houston_humidity': np.random.normal(70, 15, 1000),
        'dallas_humidity': np.random.normal(65, 12, 1000),
        'houston_wind_speed': np.random.normal(8, 3, 1000),
        'dallas_wind_speed': np.random.normal(10, 4, 1000),
    })
    print("Sample data created successfully")

# Limit to a sample for faster processing
sample_size = min(1000, len(df))
df = df.sample(sample_size, random_state=42).reset_index(drop=True)
print(f"Using sample of {sample_size} records for analysis")

# Convert date to datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Check for missing values
print("\nMissing values in the dataset:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Select features for clustering
# Focus on load values and weather-related features
load_features = ['houston', 'dallas', 'san antonio']
time_features = ['hour', 'day_of_week', 'is_weekend']
weather_features = [col for col in df.columns if any(x in col for x in 
                   ['temperature', 'humidity', 'wind_speed'])]

# Filter out any columns that don't exist in the dataframe
all_features = load_features + time_features + weather_features
features_to_use = [feature for feature in all_features if feature in df.columns]

print("\nSelected features for clustering:")
print(features_to_use)

# Create a dataframe with only the selected features
clustering_df = df[features_to_use].copy()

# Handle missing values
print("\nMissing values in selected features:")
print(clustering_df.isnull().sum()[clustering_df.isnull().sum() > 0])

# Fill missing values with median
for column in clustering_df.columns:
    if clustering_df[column].isnull().sum() > 0:
        clustering_df[column] = clustering_df[column].fillna(clustering_df[column].median())

# Verify no missing values remain
print("\nRemaining missing values:")
print(clustering_df.isnull().sum().sum())

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_df)
scaled_df = pd.DataFrame(scaled_features, columns=clustering_df.columns)

# 2. Dimensionality Reduction
print("\n2. Performing dimensionality reduction...")

# 2.1 Principal Component Analysis (PCA)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_features)

# Create a DataFrame for the PCA result
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

# Print explained variance ratio
print("\nExplained variance ratio by principal components:")
print(pca.explained_variance_ratio_)
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.savefig('clustering_results/pca_variance.png')
plt.close()

# Visualize PCA results in 2D
plt.figure(figsize=(12, 10))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.savefig('clustering_results/pca_2d.png')
plt.close()

# Visualize PCA results in 3D (save as 2D projection)
try:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.5)
    ax.set_title('PCA: First Three Principal Components')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.tight_layout()
    plt.savefig('clustering_results/pca_3d.png')
except Exception as e:
    print(f"Could not create 3D plot: {e}")
finally:
    plt.close()

# 2.2 Feature contribution to principal components
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=clustering_df.columns
)

# Plot feature contributions to PC1 and PC2
plt.figure(figsize=(14, 10))
# Create biplot
for i, feature in enumerate(loadings.index):
    plt.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], 
              head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.6)
    plt.text(loadings.iloc[i, 0] * 1.15, loadings.iloc[i, 1] * 1.15, feature, color='green', fontsize=12)

plt.scatter(pca_df['PC1'].values[:100], pca_df['PC2'].values[:100], alpha=0.2)  # Plot a subset of points
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('PC1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
plt.ylabel('PC2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))
plt.title('Feature Contributions to Principal Components')
plt.tight_layout()
plt.savefig('clustering_results/pca_biplot.png')
plt.close()

# 2.3 t-SNE for Non-linear Dimensionality Reduction
print("\nApplying t-SNE (this may take a while)...")
# For performance, use a smaller subset and fewer iterations for t-SNE
sample_size_tsne = min(500, len(scaled_features))
indices = np.random.choice(len(scaled_features), sample_size_tsne, replace=False)
subset_features = scaled_features[indices]

try:
    tsne = TSNE(n_components=2, perplexity=min(30, sample_size_tsne-1), 
                n_iter=300, random_state=42)
    tsne_result = tsne.fit_transform(subset_features)

    # Create a DataFrame for the t-SNE result
    tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE1', 't-SNE2'])

    # Visualize t-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], alpha=0.5)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig('clustering_results/tsne.png')
except Exception as e:
    print(f"t-SNE failed with error: {e}")
    print("Skipping t-SNE visualization")
finally:
    plt.close()

# 3. K-Means Clustering
print("\n3. Performing K-Means clustering...")

# 3.1 Elbow Method to find optimal K
distortions = []
silhouette_scores = []
K_range = range(2, 9)  # Reduced range for faster processing
k_values = []  # Store k values for silhouette scores

for k in K_range:
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        distortions.append(kmeans.inertia_)
        
        # Calculate silhouette score
        if k > 1:  # Silhouette score is not defined for k=1
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(scaled_features, labels)
            silhouette_scores.append(silhouette_avg)
            k_values.append(k)  # Store the k value
            print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.4f}")
    except Exception as e:
        print(f"Error with k={k}: {e}")
        # Add placeholder values if calculation fails
        distortions.append(np.nan if len(distortions) > 0 else 0)
        if k > 1:
            silhouette_scores.append(np.nan if len(silhouette_scores) > 0 else 0)
            k_values.append(k)

# Plot the elbow curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(K_range, distortions, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, 'rx-')  # Use k_values instead
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Different k')
plt.grid(True)

plt.tight_layout()
plt.savefig('clustering_results/kmeans_elbow.png')
plt.close()

# 3.2 Apply K-Means with the optimal number of clusters
# Based on the elbow curve and silhouette scores, choose optimal k
# Get index of highest silhouette score
if silhouette_scores:
    best_k_idx = np.argmax(silhouette_scores)
    optimal_k = list(K_range)[1:][best_k_idx]
else:
    optimal_k = 4  # Default if silhouette calculation failed

print(f"\nSelected optimal k = {optimal_k}")

# Apply K-Means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Add cluster labels to the original dataframe
df['kmeans_cluster'] = kmeans_labels

# Print cluster sizes
cluster_sizes = df['kmeans_cluster'].value_counts().sort_index()
print("\nK-Means Cluster Sizes:")
print(cluster_sizes)

# 3.3 Visualize K-Means Clusters on PCA Components
plt.figure(figsize=(14, 6))

# 2D visualization
plt.subplot(1, 2, 1)
for i in range(optimal_k):
    plt.scatter(pca_df.iloc[kmeans_labels == i, 0], 
                pca_df.iloc[kmeans_labels == i, 1],
                label=f'Cluster {i}',
                alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, marker='X', c='black', label='Centroids')
plt.title('K-Means Clusters Visualized with PCA (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 3D visualization (saved as 2D projection)
try:
    ax = plt.subplot(1, 2, 2, projection='3d')
    for i in range(optimal_k):
        ax.scatter(pca_df.iloc[kmeans_labels == i, 0], 
                   pca_df.iloc[kmeans_labels == i, 1],
                   pca_df.iloc[kmeans_labels == i, 2],
                   label=f'Cluster {i}',
                   alpha=0.7)
    ax.set_title('K-Means Clusters Visualized with PCA (3D)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
except Exception as e:
    print(f"3D visualization failed: {e}")

plt.tight_layout()
plt.savefig('clustering_results/kmeans_clusters_pca.png')
plt.close()

# 3.5 Analyze cluster characteristics
# Calculate mean values for each cluster
cluster_means = df.groupby('kmeans_cluster')[features_to_use].mean()

# Write cluster means to CSV
cluster_means.to_csv('clustering_results/kmeans_cluster_means.csv')
print("\nK-Means cluster means saved to 'clustering_results/kmeans_cluster_means.csv'")

# Visualize the cluster profiles
plt.figure(figsize=(16, 12))

# 1. Load features comparison
plt.subplot(2, 2, 1)
if all(feature in cluster_means.columns for feature in load_features):
    cluster_means[load_features].plot(kind='bar', ax=plt.gca())
    plt.title('Electricity Load by Cluster')
    plt.ylabel('Average Load')
    plt.xticks(rotation=0)
    plt.grid(axis='y')

# 2. Time features comparison
time_feats = [feat for feat in ['hour', 'day_of_week', 'is_weekend'] if feat in cluster_means.columns]
if time_feats:
    plt.subplot(2, 2, 2)
    cluster_means[time_feats].plot(kind='bar', ax=plt.gca())
    plt.title('Time Features by Cluster')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.grid(axis='y')

# 3. Temperature comparison (select a few representative cities)
temp_cols = [col for col in cluster_means.columns if 'temperature' in col]
if temp_cols:
    plt.subplot(2, 2, 3)
    # Use at most 5 temperature columns to avoid overcrowding
    cols_to_plot = temp_cols[:min(5, len(temp_cols))]
    cluster_means[cols_to_plot].plot(kind='bar', ax=plt.gca())
    plt.title('Temperature by Cluster (Selected Cities)')
    plt.ylabel('Average Temperature')
    plt.xticks(rotation=0)
    plt.grid(axis='y')

# 4. Humidity comparison (select a few representative cities)
humidity_cols = [col for col in cluster_means.columns if 'humidity' in col]
if humidity_cols:
    plt.subplot(2, 2, 4)
    # Use at most 5 humidity columns
    cols_to_plot = humidity_cols[:min(5, len(humidity_cols))]
    cluster_means[cols_to_plot].plot(kind='bar', ax=plt.gca())
    plt.title('Humidity by Cluster (Selected Cities)')
    plt.ylabel('Average Humidity')
    plt.xticks(rotation=0)
    plt.grid(axis='y')

plt.tight_layout()
plt.savefig('clustering_results/kmeans_cluster_profiles.png')
plt.close()

# 4. DBSCAN Clustering
print("\n4. Performing DBSCAN clustering...")

# 4.1 Apply DBSCAN
# We'll use the PCA result to perform DBSCAN since the original feature space is high-dimensional
try:
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(pca_result)

    # Add DBSCAN labels to the dataframe
    df['dbscan_cluster'] = dbscan_labels

    # Number of clusters (excluding noise if present)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"Number of DBSCAN clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(dbscan_labels):.2%} of data)")

    # Print cluster sizes
    dbscan_sizes = df['dbscan_cluster'].value_counts().sort_index()
    print("\nDBSCAN Cluster Sizes:")
    print(dbscan_sizes)

    # 4.2 Visualize DBSCAN clusters in PCA space
    plt.figure(figsize=(14, 6))

    # 2D visualization
    plt.subplot(1, 2, 1)
    # Plot noise points first
    noise_mask = dbscan_labels == -1
    plt.scatter(pca_df.iloc[noise_mask, 0], pca_df.iloc[noise_mask, 1], 
                c='black', marker='x', label='Noise', alpha=0.5)

    # Plot clusters
    for i in range(n_clusters):
        mask = dbscan_labels == i
        plt.scatter(pca_df.iloc[mask, 0], pca_df.iloc[mask, 1], label=f'Cluster {i}', alpha=0.7)

    plt.title('DBSCAN Clusters Visualized with PCA (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    # 3D visualization
    try:
        ax = plt.subplot(1, 2, 2, projection='3d')
        # Plot noise points first
        ax.scatter(pca_df.iloc[noise_mask, 0], pca_df.iloc[noise_mask, 1], pca_df.iloc[noise_mask, 2],
                  c='black', marker='x', label='Noise', alpha=0.5)

        # Plot clusters
        for i in range(n_clusters):
            mask = dbscan_labels == i
            ax.scatter(pca_df.iloc[mask, 0], pca_df.iloc[mask, 1], pca_df.iloc[mask, 2], 
                       label=f'Cluster {i}', alpha=0.7)

        ax.set_title('DBSCAN Clusters Visualized with PCA (3D)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
    except Exception as e:
        print(f"3D DBSCAN visualization failed: {e}")

    plt.tight_layout()
    plt.savefig('clustering_results/dbscan_clusters_pca.png')
except Exception as e:
    print(f"DBSCAN clustering failed: {e}")
    n_clusters = 0  # For later references
    df['dbscan_cluster'] = -1  # All points labeled as noise
finally:
    plt.close()

# 5. Hierarchical Clustering
print("\n5. Performing Hierarchical clustering...")

# 5.1 Perform Hierarchical Clustering
try:
    # Since the dataset may be large, we'll use a subsample for the dendrogram visualization
    n_samples = min(250, len(pca_result))  # Cap at 250 samples for faster processing
    subsample_indices = np.random.choice(len(pca_result), n_samples, replace=False)
    pca_subsample = pca_result[subsample_indices]

    # Compute the linkage matrix
    Z = linkage(pca_subsample, 'ward')

    # Plot the dendrogram
    plt.figure(figsize=(15, 8))
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # Show only the last p merged clusters
        p=12,                   # Show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
    )
    plt.axhline(y=15, c='k', linestyle='--', label='Suggested Cut')
    plt.legend()
    plt.savefig('clustering_results/hierarchical_dendrogram.png')
    plt.close()

    # 5.2 Apply Hierarchical Clustering with chosen distance threshold or number of clusters
    n_clusters_hierarchical = 4  # This should be updated based on dendrogram analysis

    # Apply hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hierarchical, linkage='ward')
    hierarchical_labels = hierarchical.fit_predict(pca_result)

    # Add hierarchical clustering labels to the dataframe
    df['hierarchical_cluster'] = hierarchical_labels

    # Print cluster sizes
    hierarchical_sizes = df['hierarchical_cluster'].value_counts().sort_index()
    print("Hierarchical Clustering - Cluster Sizes:")
    print(hierarchical_sizes)

    # Calculate silhouette score
    silhouette_hierarchical = silhouette_score(pca_result, hierarchical_labels)
    print(f"Silhouette score for hierarchical clustering: {silhouette_hierarchical:.4f}")

    # 5.3 Visualize hierarchical clusters in PCA space
    plt.figure(figsize=(14, 6))

    # 2D visualization
    plt.subplot(1, 2, 1)
    for i in range(n_clusters_hierarchical):
        mask = hierarchical_labels == i
        plt.scatter(pca_df.iloc[mask, 0], pca_df.iloc[mask, 1], 
                    label=f'Cluster {i}', alpha=0.7)
    plt.title('Hierarchical Clusters (PCA 2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    # 3D visualization
    try:
        ax = plt.subplot(1, 2, 2, projection='3d')
        for i in range(n_clusters_hierarchical):
            mask = hierarchical_labels == i
            ax.scatter(pca_df.iloc[mask, 0], pca_df.iloc[mask, 1], pca_df.iloc[mask, 2],
                       label=f'Cluster {i}', alpha=0.7)
        ax.set_title('Hierarchical Clusters (PCA 3D)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
    except Exception as e:
        print(f"3D hierarchical visualization failed: {e}")

    plt.tight_layout()
    plt.savefig('clustering_results/hierarchical_clusters_pca.png')
except Exception as e:
    print(f"Hierarchical clustering failed: {e}")
    n_clusters_hierarchical = 0
    df['hierarchical_cluster'] = -1  # Assign all to one cluster for error case
finally:
    plt.close()

# 6. Cluster Evaluation and Comparison
print("\n6. Evaluating clustering results...")

# 6.1 Compare silhouette scores across different clustering methods
silhouette_scores_list = []

try:
    # K-Means silhouette score
    silhouette_kmeans = silhouette_score(scaled_features, kmeans_labels)
    silhouette_scores_list.append({'Method': 'K-Means', 'Silhouette Score': silhouette_kmeans})

    # DBSCAN silhouette score (if applicable)
    # Only calculate if there's more than one cluster and not all points are noise
    if 'dbscan_cluster' in df.columns and n_clusters > 1:
        mask = df['dbscan_cluster'] != -1
        if np.sum(mask) > 1:  # Need more than one non-noise point
            silhouette_dbscan = silhouette_score(pca_result[mask], df.loc[mask, 'dbscan_cluster'])
            silhouette_scores_list.append({'Method': 'DBSCAN', 'Silhouette Score': silhouette_dbscan})

    # Hierarchical clustering silhouette score
    if 'hierarchical_cluster' in df.columns and n_clusters_hierarchical > 1:
        silhouette_scores_list.append({'Method': 'Hierarchical', 'Silhouette Score': silhouette_hierarchical})

    # Convert to dataframe and display
    silhouette_df = pd.DataFrame(silhouette_scores_list)
    silhouette_df = silhouette_df.sort_values(by='Silhouette Score', ascending=False)
    print("\nSilhouette scores by clustering method:")
    print(silhouette_df)

    # Write silhouette scores to CSV
    silhouette_df.to_csv('clustering_results/silhouette_scores.csv', index=False)

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.bar(silhouette_df['Method'], silhouette_df['Silhouette Score'])
    plt.title('Silhouette Scores by Clustering Method')
    plt.ylabel('Silhouette Score')
    plt.ylim(0, 1)  # Silhouette score ranges from -1 to 1
    plt.grid(axis='y')
    plt.savefig('clustering_results/silhouette_scores.png')
except Exception as e:
    print(f"Silhouette score calculation failed: {e}")
finally:
    plt.close()

# 6.3 Cluster stability check for K-Means
try:
    n_runs = 3  # Reduced for faster processing
    kmeans_labels_runs = []

    for i in range(n_runs):
        kmeans = KMeans(n_clusters=optimal_k, random_state=i*10)
        labels = kmeans.fit_predict(scaled_features)
        kmeans_labels_runs.append(labels)

    # Calculate pairwise adjusted rand index between runs
    rand_scores = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            score = adjusted_rand_score(kmeans_labels_runs[i], kmeans_labels_runs[j])
            rand_scores.append({'Run 1': i, 'Run 2': j, 'Adjusted Rand Index': score})

    # Convert to dataframe and display
    rand_scores_df = pd.DataFrame(rand_scores)
    print("\nCluster Stability - Adjusted Rand Index between K-Means Runs:")
    print(rand_scores_df)

    # Calculate average stability score
    avg_stability = rand_scores_df['Adjusted Rand Index'].mean()
    print(f"Average stability score (Adjusted Rand Index): {avg_stability:.4f}")
except Exception as e:
    print(f"Stability analysis failed: {e}")

# 7. Cluster Interpretation and Characterization
print("\n7. Interpreting clusters...")

try:
    # 7.1 Select the best clustering method based on silhouette scores
    if len(silhouette_scores_list) > 0:
        best_method = silhouette_df.iloc[0]['Method']
        print(f"The best clustering method based on silhouette score is: {best_method}")
        
        # Use the corresponding cluster labels
        if best_method == 'K-Means':
            best_labels = kmeans_labels
            best_column = 'kmeans_cluster'
            n_best_clusters = optimal_k
        elif best_method == 'DBSCAN':
            best_labels = df['dbscan_cluster']
            best_column = 'dbscan_cluster'
            n_best_clusters = n_clusters
        else:  # Hierarchical
            best_labels = df['hierarchical_cluster']
            best_column = 'hierarchical_cluster'
            n_best_clusters = n_clusters_hierarchical
    else:
        # Default to K-Means if silhouette scores failed
        best_method = 'K-Means'
        best_labels = kmeans_labels
        best_column = 'kmeans_cluster'
        n_best_clusters = optimal_k
        print("Using K-Means as default best method")

    # 7.3 Analyze load and weather characteristics per cluster
    # Get cluster means for the best clustering method
    if best_method == 'DBSCAN':
        # For DBSCAN, excluding noise points
        valid_clusters = df[df[best_column] != -1]
        best_cluster_means = valid_clusters.groupby(best_column)[features_to_use].mean()
    else:
        best_cluster_means = df.groupby(best_column)[features_to_use].mean()

    # Calculate the relative difference from the overall mean to highlight distinctive features
    overall_means = df[features_to_use].mean()
    relative_diff = ((best_cluster_means - overall_means) / overall_means).fillna(0)

    # Highlight the most distinctive features for each cluster
    n_top_features = 3  # Number of top distinctive features to display

    print("\nMost Distinctive Features per Cluster:")
    for cluster in best_cluster_means.index:
        print(f"\nCluster {cluster}:")
        # Get top positive differences (higher than average)
        top_positive = relative_diff.loc[cluster].nlargest(n_top_features)
        print("Higher than average:")
        for feature, value in top_positive.items():
            print(f"  {feature}: {value:.2%} higher")
        
        # Get top negative differences (lower than average)
        top_negative = relative_diff.loc[cluster].nsmallest(n_top_features)
        print("Lower than average:")
        for feature, value in top_negative.items():
            print(f"  {feature}: {value:.2%} lower")

    # Write relative differences to CSV
    relative_diff.to_csv('clustering_results/cluster_relative_differences.csv')

    # 7.4 Create descriptive names for each cluster
    # Base names on cluster characteristics
    cluster_names = {}
    for cluster in best_cluster_means.index:
        # Find most extreme characteristics
        highest_feat = relative_diff.loc[cluster].nlargest(1).index[0]
        lowest_feat = relative_diff.loc[cluster].nsmallest(1).index[0]
        
        # Create descriptive name
        high_value = relative_diff.loc[cluster, highest_feat]
        low_value = relative_diff.loc[cluster, lowest_feat]
        
        if abs(high_value) > abs(low_value):
            descriptor = f"High-{highest_feat.replace('_', '-')}"
        else:
            descriptor = f"Low-{lowest_feat.replace('_', '-')}"
            
        # Add time context if available
        if 'hour' in best_cluster_means.columns:
            hour = best_cluster_means.loc[cluster, 'hour']
            if hour < 6:
                time_desc = "night"
            elif hour < 12:
                time_desc = "morning"
            elif hour < 18:
                time_desc = "afternoon"
            else:
                time_desc = "evening"
            
            descriptor += f" {time_desc}"
        
        # Add weekend context if available
        if 'is_weekend' in best_cluster_means.columns:
            is_weekend = best_cluster_means.loc[cluster, 'is_weekend'] > 0.5
            if is_weekend:
                descriptor += " weekend"
                
        cluster_names[cluster] = descriptor

    # Print cluster descriptions
    print("\nCluster Descriptions:")
    cluster_descriptions = []
    for cluster, name in cluster_names.items():
        description = f"Cluster {cluster}: {name}"
        print(description)
        
        # Add specific characteristics if available
        load_feats = [feat for feat in load_features if feat in best_cluster_means.columns]
        if load_feats:
            load_values = best_cluster_means.loc[cluster, load_feats]
            load_info = f"  Average loads: {', '.join([f'{city}: {load:.2f}' for city, load in load_values.items()])}"
            print(load_info)
        else:
            load_info = ""
        
        # Add time/seasonal characteristics if available
        hour_info = ""
        if 'hour' in best_cluster_means.columns:
            hour_info = f"  Average hour: {best_cluster_means.loc[cluster, 'hour']:.1f}"
            print(hour_info)
        
        # Add temperature information if available
        temp_info = ""
        temp_features = [col for col in best_cluster_means.columns if 'temperature' in col]
        if temp_features:
            avg_temp = best_cluster_means.loc[cluster, temp_features].mean()
            temp_info = f"  Average temperature: {avg_temp:.1f}"
            print(temp_info)
            
        cluster_descriptions.append({
            'Cluster': cluster,
            'Name': name,
            'Load Info': load_info,
            'Hour Info': hour_info,
            'Temp Info': temp_info
        })

    # Write cluster descriptions to CSV
    pd.DataFrame(cluster_descriptions).to_csv('clustering_results/cluster_descriptions.csv', index=False)
except Exception as e:
    print(f"Cluster interpretation failed: {e}")

print("\nClustering analysis complete!")
print(f"Results saved to the 'clustering_results' directory.") 