import pandas as pd
import joblib

model = joblib.load("model/f1_model.pkl")
model_columns = joblib.load("model/model_columns.pkl")

df_2026 = pd.read_csv("data/Formula1_2026season_raceResults.csv")

df_2026["Position"] = pd.to_numeric(df_2026["Position"], errors="coerce")
df_2026["Points"] = pd.to_numeric(df_2026["Points"], errors="coerce")

driver_stats = df_2026.groupby(["Driver", "Team"]).agg(
    driver_points_before_race=("Points", "sum"),
    driver_wins_before_race=("Position", lambda x: (x == 1).sum()),
    driver_avg_finish_before_race=("Position", "mean")
).reset_index()

next_race = driver_stats.copy()

next_race["Track"] = "China"

grid_order = {
    "Max Verstappen": 7,
    "Lando Norris": 5,
    "Oscar Piastri": 3,
    "Charles Leclerc": 4,
    "George Russell": 2,
    "Lewis Hamilton": 6,
    "Kimi Antonelli": 1,
    "Carlos Sainz": 8,
    "Alexander Albon": 9,
    "Fernando Alonso": 10,
    "Liam Lawson": 11,
    "Isack Hadjar": 12,
    "Esteban Ocon": 13,
    "Pierre Gasly": 14,
    "Gabriel Bortoleto": 15,
    "Nico Hulkenberg": 16,
    "Oliver Bearman": 17,
    "Franco Colapinto": 18,
    "Lance Stroll": 19,
    "Sergio Perez": 20,
    "Valtteri Bottas": 21,
    "Arvid Lindblad": 22
}
next_race["Starting Grid"] = next_race["Driver"].map(grid_order)

next_race = next_race.dropna(subset=["Starting Grid"])
X = next_race[
    [
        "Starting Grid",
        "driver_points_before_race",
        "driver_wins_before_race",
        "driver_avg_finish_before_race",
        "Driver",
        "Team",
        "Track",
    ]
]

X = pd.get_dummies(X)
X = X.reindex(columns=model_columns, fill_value=0)

next_race["win_probability"] = model.predict_proba(X)[:, 1]

next_race = next_race.sort_values("win_probability", ascending=False)

print(next_race[["Driver", "Team", "Starting Grid", "win_probability"]])