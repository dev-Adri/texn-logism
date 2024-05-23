# Έχουμε κάνει μέχρι και το 3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

from streamlit_option_menu import option_menu

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
        st.title("You have not uploaded any data.")
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
                    
                    tsne = TSNE(n_components=2, perplexity=30)
                    tsne_data = tsne.fit_transform(st.session_state.processed_data)
                    
                    plt.figure(figsize=(10, 6))
                    plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('t-SNE')
                    
                    st.pyplot(plt.gcf())
                    placeholder.empty()
        
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
        st.title("You have not uploaded any data.")
        st.text("Please go back to the home page and upload a CSV or Excel file.")
    else:
        if st.session_state.preprocessed == False:
            st.error("Please preprocess the data first.")
        else:
            st.title("Machine learning")

            classification, clustering = st.tabs(["Classification algorithms", "Clustering algorithms"])
            
            with classification:
                st.write("Edw tha valoume ton xristi na epilegei kapies parametrous ...")
                
            with clustering:
                st.write("Edw tha valoume ton xristi na epilegei kapies parametrous 2 ...")