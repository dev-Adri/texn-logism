# Έχουμε κάνει μέχρι και το 3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Τίτλος
st.title('Εφαρμογή ...')

# Φόρτωση αρχείου
uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Ανάγνωση του αρχείου με βάση τον τύπο του
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        # Εμφάνιση των γραμμών και στηλών
        st.write("Σύνολο γραμμών (δείγματα):", df.shape[0])
        st.write("Σύνολο στηλών (χαρακτηριστικά + ετικέτα):", df.shape[1])
        
        # Έλεγχος ότι υπάρχει τουλάχιστον μία στήλη χαρακτηριστικών και μία στήλη ετικετών
        if df.shape[1] < 2:
            st.error("Ο πίνακας πρέπει να περιλαμβάνει τουλάχιστον μία στήλη χαρακτηριστικών και μία στήλη ετικετών.")
        else:
            # Διαχωρισμός των χαρακτηριστικών και των ετικετών
            features = df.iloc[:, :-1]
            labels = df.iloc[:, -1]

            # Χειρισμός τιμών NaN
            if features.isnull().values.any():
                st.warning("Τα δεδομένα περιέχουν τιμές NaN. Θα αντικατασταθούν με τη μέση τιμή της κάθε στήλης.")
                features = features.fillna(features.mean())

            # Χρήση tabs για τα διάγραμματα
            tab1, tab2 = st.tabs(["Δεδομένα", "2D Οπτικοποιήσεις & EDA"])

            with tab1:
                st.write("Χαρακτηριστικά (SxF):")
                st.dataframe(features)

                st.write("Ετικέτες (F+1):")
                st.dataframe(labels)

            with tab2: 
                # Σταθερή τιμή perplexity
                perplexity = min(30, features.shape[0] - 1)
                
                # Εφαρμογή PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(features)
                pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
                pca_df['label'] = labels.values

                st.write("PCA Διάγραμμα:")
                fig, ax = plt.subplots()
                sns.scatterplot(x='PCA1', y='PCA2', hue='label', data=pca_df, ax=ax)
                st.pyplot(fig)

                # Εφαρμογή t-SNE
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                tsne_result = tsne.fit_transform(features)
                tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
                tsne_df['label'] = labels.values

                st.write("t-SNE Διάγραμμα:")
                fig, ax = plt.subplots()
                sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='label', data=tsne_df, ax=ax)
                st.pyplot(fig)


                # Exploratory Data Analysis (EDA) Διαγράμματα
                st.write("EDA Διαγράμματα:")

                # Διάγραμμα κατανομής (Histogram)
                st.write("Διάγραμμα Κατανομής για την πρώτη στήλη χαρακτηριστικών:")
                fig, ax = plt.subplots()
                sns.histplot(features.iloc[:, 0], kde=True, ax=ax)
                st.pyplot(fig)

                # Boxplot για τον εντοπισμό outliers
                st.write("Boxplot για την πρώτη στήλη χαρακτηριστικών:")
                fig, ax = plt.subplots()
                sns.boxplot(y=features.iloc[:, 0], ax=ax)
                st.pyplot(fig)

                # Pairplot για να δούμε τη σχέση μεταξύ χαρακτηριστικών
                st.write("Pairplot για τα πρώτα 3 χαρακτηριστικά:")
                if features.shape[1] >= 3:
                    fig = sns.pairplot(features.iloc[:, :3])
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Σφάλμα κατά την φόρτωση των δεδομένων: {e}")

else:
    st.info("Παρακαλώ φορτώστε ένα αρχείο για να ξεκινήσετε.")
