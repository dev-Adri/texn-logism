import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist, pdist
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from streamlit_option_menu import option_menu
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score

selected = option_menu(     
    menu_title=None,
    options=["Home", "2D-Visualization", "Machine Learning", "Info"],
    icons=["house", "graph-up", "robot", "info"],
    orientation="horizontal",
)

def dunn_index(X, labels):
    # Find all unique clusters
    unique_clusters = np.unique(labels)
    
    # Initialize the minimum inter-cluster distance to a large number
    min_inter_cluster_distance = np.inf
    
    # Calculate the minimum inter-cluster distance
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i = X[labels == unique_clusters[i]]
            cluster_j = X[labels == unique_clusters[j]]
            inter_cluster_distance = np.min(cdist(cluster_i, cluster_j))
            if inter_cluster_distance < min_inter_cluster_distance:
                min_inter_cluster_distance = inter_cluster_distance
    
    # Initialize the maximum intra-cluster distance to a small number
    max_intra_cluster_distance = 0
    
    # Calculate the maximum intra-cluster distance
    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        intra_cluster_distance = np.max(pdist(cluster_points))
        if intra_cluster_distance > max_intra_cluster_distance:
            max_intra_cluster_distance = intra_cluster_distance
    
    # Calculate the Dunn Index
    dunn_index_value = min_inter_cluster_distance / max_intra_cluster_distance
    return dunn_index_value

def preprocess_data(df):
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                except Exception as e:
                    df.drop(columns=[col], inplace=True)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            df.drop(columns=[col], inplace=True)
    
    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df

def calculate_accuracy(y_true, y_pred):
    correct_predictions = 0
    total_predictions = len(y_true)
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_f1_score(y_true, y_pred):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            true_positives += 1
        elif true_label == 0 and pred_label == 1:
            false_positives += 1
        elif true_label == 1 and pred_label == 0:
            false_negatives += 1

    # print("True positives:", true_positives)
    # print("False positives:", false_positives)
    # print("False negatives:", false_negatives)
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score


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
    # Φόρτωση αρχείου
    uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df = read(uploaded_file)

            # Save file to browser session, so even if page reloads -
            # - (which it does when you change tab) the file wont be lost
            st.session_state.file_pd = df

            # Εμφάνιση των γραμμών και στηλών
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
        
        pca_data, tsne_data = None, None

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
                if st.session_state.preprocessed == False:
                    st.error("Please preprocess the data first.")
                else:
                    placeholder = st.empty()
                    placeholder.text("Generating PCA graph...")
                    
                    pca = PCA(n_components=2)
                    pca.fit(st.session_state.processed_data)
                    pca_data = pca.transform(st.session_state.processed_data)
                    
                    plt.figure(figsize=(10, 6))
                    plt.scatter(pca_data[:, 0], pca_data[:, 1])
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('PCA')
                    
                    st.pyplot(plt.gcf())
                    placeholder.empty()
                
        with tsne:
            if st.button("Generate t-SNE graph"):
                if st.session_state.preprocessed == False:
                    st.error("Please preprocess the data first.")
                else:
                    placeholder = st.empty()
                    placeholder.text("Generating t-SNE graph...")
                    
                    # Check the number of samples and adjust perplexity
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
            # st.session_state.processed_data -> Proccesed data (used for PCA and t-SNE)
            # st.session_state.file_pd -> Unprocessed data
            if st.button("Explore your data"):
                if st.session_state.preprocessed == False:
                    st.error("Please preprocess the data first.")
                else:
                    if st.session_state.processed_data.shape[1] < 2:
                        st.write("Cannot run exploratory data analysis on a dataset with less than 2 features.")
                    else:
                        df = st.session_state.processed_data
                        placeholder = st.empty()
                        
                        placeholder.text("Generating a scatter plot ...")
                        # Scatter plot (separate figure)
                        if len(df.columns) >= 2:
                            plt.figure(figsize=(10, 6))
                            sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df)
                            plt.title(f'Scatter Plot of {df.columns[0]} and {df.columns[1]}')
                            plt.xlabel(df.columns[0])
                            plt.ylabel(df.columns[1])
                            st.pyplot(plt.gcf())  # Display the scatter plot
                        
                        placeholder.text("Generating a distribution plot ...")
                        # Distribution plot (separate figure)
                        if len(df.columns) >= 1:
                            plt.figure(figsize=(10, 6))
                            sns.histplot(df[df.columns[0]], kde=True, color='blue')
                            plt.title(f'Distribution of {df.columns[0]}')
                            plt.xlabel(df.columns[0])
                            plt.ylabel('Frequency')
                            st.pyplot(plt.gcf())  # Display the distribution plot

                        placeholder.text("Generating a pair plot for all features (might take some time) ...")
                        # Create a separate figure for the pair plot
                        pairplot_fig = sns.pairplot(df)
                        pairplot_fig.figure.suptitle('Pair Plot of the Dataset')
                        st.pyplot(pairplot_fig)
                        
                        placeholder.empty()

elif selected == "Machine Learning":
    if st.session_state.file_pd is None:
        st.title("You haven't uploaded any data yet.")
        st.text("Please go back to the home page and upload a CSV or Excel file.")
    else:
        if st.session_state.preprocessed == False:
            st.error("Please preprocess the data first.")
        else:
            st.title("Machine learning")

            classification, clustering = st.tabs(["Classification algorithms", "Clustering algorithms"])
            
            with classification:
                
                st.header("Classification Algorithms")
                target_column = st.selectbox("Select target column for the Random Forest Classifier", st.session_state.file_pd.columns)
                n_neighbors = st.slider("Select number of neighbors for the K-Nearest Neighbors Classifier", min_value=1, max_value=20, value=5)
                # Random Forest Classifier
                if st.button("Compare Random Forest Classifier and K-Nearest Neighbors Classifier"):
                    # Split the data into features and target variable
                    X = st.session_state.processed_data.drop(columns=target_column)
                    y = st.session_state.processed_data[target_column]

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Initialize and train the Random Forest Classifier
                    rf_classifier = RandomForestClassifier()
                    rf_classifier.fit(X_train, y_train)

                    # Initialize and train the KNN Classifier
                    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
                    knn_classifier.fit(X_train, y_train)

                    # Make predictions
                    rf_y_pred = rf_classifier.predict(X_test)
                    knn_y_pred = knn_classifier.predict(X_test)

                    # Extract actual values
                    actual_values = y_test.values

                    # Print the actual values
                    # print(actual_values)

                    # Calculate accuracy and F1 score for Random Forest Classifier
                    rf_accuracy = calculate_accuracy(actual_values, rf_y_pred)
                    rf_f1 = calculate_f1_score(actual_values, rf_y_pred)

                    # Calculate accuracy and F1 score for K-Nearest Neighbors Classifier
                    knn_accuracy = calculate_accuracy(actual_values, knn_y_pred)
                    knn_f1 = calculate_f1_score(actual_values, knn_y_pred)

                    # Display the results
                    st.subheader("Model Comparison")
                    st.write("Random Forest Classifier:")
                    st.write(f"Accuracy: {rf_accuracy:.3f}")
                    st.write(f"F1 Score: {rf_f1:.3f}")
                    st.write("K-Nearest Neighbors Classifier:")
                    st.write(f"Accuracy: {knn_accuracy:.3f}")
                    st.write(f"F1 Score: {knn_f1:.3f}")

            with clustering:
                st.header("Clustering Algorithms")

                num_clusters = st.slider("Select number of clusters for both algorithms", min_value=2, max_value=10, value=2)

                if st.button("Run clustering algorithms"):
                    X = st.session_state.processed_data

                    # K-Means Clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                    kmeans.fit(X)
                    kmeans_clusters = kmeans.labels_
                    kmeans_silhouette = silhouette_score(X, kmeans_clusters)
                    kmeans_dunn = dunn_index(X, kmeans_clusters)
                    kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_clusters)
                    st.subheader("K-Means cluster algorithm :")
                    st.write(f"")
                    st.write(f"K-Means Silhouette score: {kmeans_silhouette:.3f}")
                    st.write(f"")
                    st.write(f"K-Means Dunn Index score: {kmeans_dunn:.3f}")
                    st.write(f"")
                    st.write(f"K-Means Davies-Bouldin Index score: {kmeans_davies_bouldin:.3f}")
                    st.write(f"")

                    # Gaussian Mixture Models Clustering
                    gmm = GaussianMixture(n_components=num_clusters).fit(X)
                    gmm_clusters = gmm.predict(X)
                    gmm_silhouette = silhouette_score(X, gmm_clusters)
                    gmm_dunn = dunn_index(X, gmm_clusters)
                    gmm_davies_bouldin = davies_bouldin_score(X, gmm_clusters)
                    st.subheader("GMM cluster algorithm :")
                    st.write(f"")
                    st.write(f"GMM Silhouette score: {gmm_silhouette:.3f}")
                    st.write(f"")
                    st.write(f"GMM Dunn Index score: {gmm_dunn:.3f}")
                    st.write(f"")
                    st.write(f"GMM Davies-Bouldin Index score: {gmm_davies_bouldin:.3f}")
                    st.write(f"")

                    st.subheader("By comparing the algorithms :")
                    
                    kmeans_points = 0
                    gmm_points = 0

                    if kmeans_silhouette > gmm_silhouette:
                        kmeans_points += 1
                    elif kmeans_silhouette < gmm_silhouette:
                        gmm_points += 1

                    if kmeans_dunn > gmm_dunn:
                        kmeans_points += 1
                    elif kmeans_dunn < gmm_dunn:
                        gmm_points += 1

                    if kmeans_davies_bouldin < gmm_davies_bouldin:
                        kmeans_points += 1
                    elif kmeans_davies_bouldin > gmm_davies_bouldin:
                        gmm_points += 1

                    if kmeans_points > gmm_points:
                        st.write(f"We can see that K-Means algorithm is performing better in this dataset as it won in {kmeans_points} of the 3 categories")
                    elif kmeans_points < gmm_points:
                        st.write(f"We can see that GMM algorithm is performing better in this dataset as it won in {gmm_points} of the 3 categories")
                    elif kmeans_points == gmm_points:
                        st.write(f"The algorithms are performing equally in this dataset ")


#-------------------------------------------------------------------------------------------------------------#