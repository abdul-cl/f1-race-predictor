import pandas as pd
import joblib

model = joblib.load("model/f1_model.pkl")

df = pd.read_csv("data/clean_f1_data.csv")

X = df[["Starting Grid", "Driver", "Team", "Track"]]
X_encoded = pd.get_dummies(X, columns=["Driver", "Team", "Track"])

race = df[df["Track"] == "Monaco"].copy()

race_features = race[["Starting Grid", "Driver", "Team", "Track"]]
race_encoded = pd.get_dummies(race_features)
race_encoded = race_encoded.reindex(columns=X_encoded.columns, fill_value=0)

race["win_probability"] = model.predict_proba(race_encoded)[:, 1]

race = race.sort_values("win_probability", ascending=False)

print(race[["Driver", "Team", "Starting Grid", "win_probability"]])