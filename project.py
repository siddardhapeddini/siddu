import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Background Image CSS ---
page_bg = """
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
background-size: cover;
background-position: center;
background-attachment: fixed;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")

# Movie dataset
data = {
    "title": [
        "Pushpa",
        "Dhurandhar The Revenge",
        "The Raja Saab",
        "RRR",
        "Toxic",
        "Kantara: A Legend - Chapter 1",
        "Kalki 2898 AD",
        "Tere Ishk Mein",
        "Bahubali: The Beginning",
        "Mana SankaraVaraprasad Garu",
        "Arya",
        "Arya 2",
        "Orange",
        "Geethanjali",
        "Ye Maaya Chesave",
        "Adavarai Matalaku Ardhalu Verule",
        "Malli Malli Idi Rani Roju",
        "Anand",
        "Tholi Prema",
        "Nuvve...Nuvve..."
    ],

    "genre": [
        "action drama",
        "action thriller",
        "action comedy",
        "action epic",
        "action thriller",
        "action mythological",
        "sci-fi action",
        "romance drama",
        "action epic",
        "drama",
        "romance",
        "romance drama",
        "romance drama",
        "romance classic",
        "romance drama",
        "romance drama",
        "romance emotional",
        "romance feel-good",
        "romance classic",
        "romance drama"
    ]
}

df = pd.DataFrame(data)

# Convert genres into vectors
cv = CountVectorizer()
matrix = cv.fit_transform(df["genre"])

similarity = cosine_similarity(matrix)

# User input
movie_type = st.text_input("Enter movie type (action, romance, sci-fi etc):")

if st.button("Recommend Movies"):

    movie_type = movie_type.lower()

    results = df[df["genre"].str.lower().str.contains(movie_type)]

    if results.empty:
        st.error("No movies found for this type.")
    else:
        st.subheader("🎥 Recommended Movies")
        for movie in results["title"]:
            st.write("✅", movie)
