import streamlit as st
import joblib
import pandas as pd
import os

# =========================
# PATH HANDLING
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "mood_model.pkl")

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# UI
# =========================
st.set_page_config(page_title="Feel the Music", layout="centered")
st.title("🎵 Music Mood Classifier")

st.write("Enter the audio features of a song to predict its mood")

# =========================
# INPUT FEATURES
# =========================
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)

# =========================
# PREDICTION
# =========================
if st.button("Predict Mood"):
    input_df = pd.DataFrame([[
        danceability, energy, valence, tempo,
        loudness, acousticness, speechiness, instrumentalness
    ]], columns=[
        "danceability", "energy", "valence", "tempo",
        "loudness", "acousticness", "speechiness", "instrumentalness"
    ])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df).max()

    st.success(f"🎧 Predicted Mood: **{prediction}**")
    st.write(f"Confidence: **{proba:.2%}**")
