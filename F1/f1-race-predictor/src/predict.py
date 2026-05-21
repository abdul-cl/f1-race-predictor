import pandas as pd
import joblib

# load model
model = joblib.load("model/f1_model.pkl")

# load training data only to match columns
df = pd.read_csv("data/clean_f1_data.csv")

X = df[
    [
        "Starting Grid",
        "Driver",
        "Team",
        "Track",
    ]
]

X = pd.get_dummies(X, columns=["Driver", "Team", "Track"])

# example new driver prediction
new_driver = pd.DataFrame(
    [
        {
            "Starting Grid": 1,
            "Driver": "Max Verstappen",
            "Team": "Red Bull Racing Honda RBPT",
            "Track": "Bahrain",
        }
    ]
)

new_driver = pd.get_dummies(new_driver)

new_driver = new_driver.reindex(columns=X.columns, fill_value=0)

probability = model.predict_proba(new_driver)[0][1]

print(f"Predicted win probability: {probability:.2%}")