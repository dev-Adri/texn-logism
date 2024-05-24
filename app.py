import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

from streamlit_option_menu import option_menu

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

selected = option_menu(     
    menu_title=None,
    options=["Home", "2D-Visualization", "Machine Learning"],
    icons=["house", "graph-up", "robot"],
    orientation="horizontal",
)

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
                
                target = st.selectbox("Select the target variable", st.session_state.processed_data.columns)
                if st.button("Run Random Forest Classifier"):
                    X = st.session_state.processed_data.drop(columns=[target])
                    y = st.session_state.processed_data[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred, zero_division=0))
                    
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                    
            with clustering:
                st.header("Clustering Algorithms")
                
                if st.button("Run K-Means Clustering"):
                    X = st.session_state.processed_data
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    kmeans.fit(X)
                    clusters = kmeans.labels_
                    
                    st.subheader("Silhouette Score")
                    score = silhouette_score(X, clusters)
                    st.text(f"Silhouette Score: {score}")
                    
                    st.subheader("Cluster Centers")
                    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=X.columns))
                    
                    st.subheader("Cluster Visualization")
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(X)
                    plt.figure(figsize=(10, 6))
                    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
                    plt.xlabel('PCA Component 1')
                    plt.ylabel('PCA Component 2')
                    plt.title('K-Means Clustering')
                    st.pyplot(plt.gcf())