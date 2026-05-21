import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# LOAD CLEAN DATA
df = pd.read_csv("data/clean_f1_data.csv")

# FEATURES
X = df[
    [
        "Starting Grid",
        "driver_points_before_race",
        "driver_wins_before_race",
        "driver_avg_finish_before_race",
        "Driver",
        "Team",
        "Track",
        "team_points_before_race",
        "team_wins_before_race",
        "team_avg_finish_before_race"
    ]
]

# CONVERT TEXT COLUMNS INTO NUMBERS
X = pd.get_dummies(X, columns=["Driver", "Team", "Track"])

# TARGET
y = df["won_race"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# CREATE MODEL
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICT
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# RESULTS
print(classification_report(y_test, predictions))

results = X_test.copy()
results["actual_won"] = y_test.values
results["predicted_win_probability"] = probabilities

print("\nSample predictions:")
print(results[["actual_won", "predicted_win_probability"]].head(10))

# SAVE MODEL AND FEATURE COLUMNS
joblib.dump(model, "model/f1_model.pkl")
joblib.dump(X.columns.tolist(), "model/model_columns.pkl")

print("\nModel saved successfully!")