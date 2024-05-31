import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist, pdist
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

selected = option_menu(     
    menu_title=None,
    options=["Home", "2D-Visualization", "Machine Learning", "Info"],
    icons=["house", "graph-up", "robot", "info"],
    orientation="horizontal",
)

def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    min_inter_cluster_distance = np.inf
    
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i = X[labels == unique_clusters[i]]
            cluster_j = X[labels == unique_clusters[j]]
            inter_cluster_distance = np.min(cdist(cluster_i, cluster_j))
            if inter_cluster_distance < min_inter_cluster_distance:
                min_inter_cluster_distance = inter_cluster_distance
    
    max_intra_cluster_distance = 0
    
    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        intra_cluster_distance = np.max(pdist(cluster_points))
        if intra_cluster_distance > max_intra_cluster_distance:
            max_intra_cluster_distance = intra_cluster_distance
    
    dunn_index_value = min_inter_cluster_distance / max_intra_cluster_distance
    return dunn_index_value

def preprocess_data(df):
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace('.', '').astype(float)
            except ValueError:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                except Exception as e:
                    df.drop(columns=[col], inplace=True)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            df.drop(columns=[col], inplace=True)
    
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def read(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

if "file_pd" not in st.session_state:
    st.session_state.file_pd = None
    
if "preprocessed" not in st.session_state: 
    st.session_state.preprocessed = False
    
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

if selected == "Home":
    uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df = read(uploaded_file)
            st.session_state.file_pd = df
            st.text(f"Σύνολο γραμμών (δείγματα): {df.shape[0]}")
            st.text(f"Σύνολο στηλών (χαρακτηριστικά + ετικέτα): {df.shape[1]}")
        except Exception as e:
            st.error(f"Σφάλμα κατά την φόρτωση των δεδομένων: {e}")
            
elif selected == "2D-Visualization":
    if st.session_state.file_pd is None:
        st.title("You haven't uploaded any data yet.")
        st.text("Please go back to the home page and upload a CSV or Excel file.")
    else:
        st.title("2D Visualization")
        data = st.session_state.file_pd
        ppd, pca, tsne, eda = st.tabs(["Preprocess", "PCA", "t-SNE", "Explore your data"])
        
        with ppd:
            if st.button("Preprocess Data"):
                st.session_state.preprocessed = True
                placeholder = st.empty()
                placeholder.text("Processing ...")
                data = preprocess_data(data)
                st.session_state.processed_data = data
                placeholder.text(data)
                
        with pca:
            if st.button("Generate PCA graph"):
                if not st.session_state.preprocessed:
                    st.error("Please preprocess the data first.")
                else:
                    placeholder = st.empty()
                    placeholder.text("Generating PCA graph...")
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(st.session_state.processed_data)
                    plt.figure(figsize=(10, 6))
                    plt.scatter(pca_data[:, 0], pca_data[:, 1])
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('PCA')
                    st.pyplot(plt.gcf())
                    placeholder.empty()
                
        with tsne:
            if st.button("Generate t-SNE graph"):
                if not st.session_state.preprocessed:
                    st.error("Please preprocess the data first.")
                else:
                    placeholder = st.empty()
                    placeholder.text("Generating t-SNE graph...")
                    n_samples = st.session_state.processed_data.shape[0]
                    perplexity = min(20, n_samples - 1)
                    try:
                        tsne = TSNE(n_components=2, perplexity=perplexity)
                        tsne_data = tsne.fit_transform(st.session_state.processed_data)
                        plt.figure(figsize=(10, 6))
                        plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.title('t-SNE')
                        st.pyplot(plt.gcf())
                        placeholder.empty()
                    except Exception as e:
                        placeholder.error(f"Error during t-SNE computation: {e}")

        with eda:
            if st.button("Explore your data"):
                if not st.session_state.preprocessed:
                    st.error("Please preprocess the data first.")
                else:
                    df = st.session_state.processed_data
                    placeholder = st.empty()
                    
                    placeholder.text("Generating a scatter plot ...")
                    if len(df.columns) >= 2:
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df)
                        plt.title(f'Scatter Plot of {df.columns[0]} and {df.columns[1]}')
                        plt.xlabel(df.columns[0])
                        plt.ylabel(df.columns[1])
                        st.pyplot(plt.gcf())
                    
                    placeholder.text("Generating a distribution plot ...")
                    if len(df.columns) >= 1:
                        plt.figure(figsize=(10, 6))
                        sns.histplot(df[df.columns[0]], kde=True, color='blue')
                        plt.title(f'Distribution of {df.columns[0]}')
                        plt.xlabel(df.columns[0])
                        plt.ylabel('Frequency')
                        st.pyplot(plt.gcf())

                    placeholder.text("Generating a pair plot for all features (might take some time) ...")
                    pairplot_fig = sns.pairplot(df)
                    pairplot_fig.figure.suptitle('Pair Plot of the Dataset')
                    st.pyplot(pairplot_fig)
                    
                    placeholder.empty()

elif selected == "Machine Learning":
    if st.session_state.file_pd is None:
        st.title("You haven't uploaded any data yet.")
        st.text("Please go back to the home page and upload a CSV or Excel file.")
    else:
        if not st.session_state.preprocessed:
            st.error("Please preprocess the data first.")
        else:
            st.title("Machine learning")
            classification, clustering = st.tabs(["Classification algorithms", "Clustering algorithms"])
            
            with classification:
                st.header("Classification Algorithms")
                n_neighbors = st.slider("Select number of neighbors for the K-Nearest Neighbors Classifier", min_value=1, max_value=20, value=5)
                
                if st.button("Compare Random Forest Classifier and K-Nearest Neighbors Classifier"):
                    data = st.session_state.processed_data
                    target_column = data.columns[-1]  # Automatically select the last column as target
                    X = data.drop(columns=target_column)
                    y = data[target_column]
                    
                    # Logging shape and unique values in target column for debugging
                    st.write(f"Shape of X: {X.shape}")
                    st.write(f"Shape of y: {y.shape}")
                    st.write(f"Unique values in target column: {y.unique()}")
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    rfc = RandomForestClassifier(random_state=42)
                    rfc.fit(X_train, y_train)
                    rfc_pred = rfc.predict(X_test)
                    rfc_acc = calculate_accuracy(y_test, rfc_pred)
                    rfc_f1 = calculate_f1_score(y_test, rfc_pred)
                    
                    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                    knn.fit(X_train, y_train)
                    knn_pred = knn.predict(X_test)
                    knn_acc = calculate_accuracy(y_test, knn_pred)
                    knn_f1 = calculate_f1_score(y_test, knn_pred)
                    
                    st.write("### Random Forest Classifier Results")
                    st.write(f"Accuracy: {rfc_acc}")
                    st.write(f"F1 Score: {rfc_f1}")
                    
                    st.write("### K-Nearest Neighbors Classifier Results")
                    st.write(f"Accuracy: {knn_acc}")
                    st.write(f"F1 Score: {knn_f1}")
                    
                    if rfc_acc > knn_acc:
                        st.success("Random Forest Classifier performed better in terms of accuracy.")
                    elif rfc_acc < knn_acc:
                        st.success("K-Nearest Neighbors Classifier performed better in terms of accuracy.")
                    else:
                        st.info("Both classifiers performed equally well in terms of accuracy.")
                        
            with clustering:
                st.header("Clustering Algorithms")
                n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
                
                if st.button("Compare K-Means and Gaussian Mixture Model"):
                    X = st.session_state.processed_data
                    
                    # Feature scaling
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # K-Means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans_labels = kmeans.fit_predict(X_scaled)
                    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
                    kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)
                    
                    # Gaussian Mixture Model
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                    gmm_labels = gmm.fit_predict(X_scaled)
                    gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
                    gmm_db = davies_bouldin_score(X_scaled, gmm_labels)

                    # Visualize clustering results
                    plt.figure(figsize=(10, 6))
                    plt.subplot(1, 2, 1)
                    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=kmeans_labels, palette="Set1", legend="full")
                    plt.title('K-Means Clustering')
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    
                    plt.subplot(1, 2, 2)
                    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=gmm_labels, palette="Set1", legend="full")
                    plt.title('Gaussian Mixture Model Clustering')
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    
                    st.pyplot(plt.gcf())
                    
                    st.write("### K-Means Clustering Results")
                    st.write(f"Silhouette Score: {kmeans_silhouette}")
                    st.write(f"Davies-Bouldin Score: {kmeans_db}")
                    
                    st.write("### Gaussian Mixture Model Results")
                    st.write(f"Silhouette Score: {gmm_silhouette}")
                    st.write(f"Davies-Bouldin Score: {gmm_db}")   
                
elif selected == "Info":
    st.title("Tool Info")
    st.write("This tool allows you to:")
    st.write("- Upload CSV or Excel data files.")
    st.write("- Preprocess data and handle missing values.")
    st.write("- Visualize data in 2D using PCA and t-SNE.")
    st.write("- Explore your data with scatter plots, distribution plots, and pair plots.")
    st.write("- Compare classification algorithms (Random Forest and K-Nearest Neighbors).")
    st.write("- Compare clustering algorithms (K-Means and Gaussian Mixture Model).")
    st.write("- Evaluate the performance of classification and clustering algorithms using various metrics.")
    st.write("Developed by [Your Name].")
    st.write("For more information, contact [Your Contact Information].")
