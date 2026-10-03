import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed/spotify_labeled.csv")
X = df.drop("mood", axis=1)
y = df["mood"]

model = joblib.load("model/mood_model.pkl")

y_pred = model.predict(X)

print("Classification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
