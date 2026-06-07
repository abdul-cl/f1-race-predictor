from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/f1_model.pkl")
model_columns = joblib.load("model/model_columns.pkl")

df_2026 = pd.read_csv("data/Formula1_2026season_raceResults.csv")

df_2026["Position"] = pd.to_numeric(df_2026["Position"], errors="coerce")
df_2026["Points"] = pd.to_numeric(df_2026["Points"], errors="coerce")

driver_stats = df_2026.groupby(["Driver", "Team"]).agg(
    driver_points_before_race=("Points", "sum"),
    driver_wins_before_race=("Position", lambda x: (x == 1).sum()),
    driver_avg_finish_before_race=("Position", "mean"),
    team_points_before_race=("Points", "sum"),
    team_wins_before_race=("Position", lambda x: (x == 1).sum()),
    team_avg_finish_before_race=("Position", "mean"),
).reset_index()

class GridEntry(BaseModel):
    driver: str
    grid: int


class PredictionRequest(BaseModel):
    track: str
    grid_order: List[GridEntry]


@app.get("/")
def home():
    return {"message": "F1 Predictor API is running"}


@app.get("/drivers")
def get_drivers():
    return driver_stats[["Driver", "Team"]].to_dict(orient="records")


@app.post("/predict")
def predict(request: PredictionRequest):
    next_race = driver_stats.copy()

    valid_tracks = [
        "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
        "Miami", "Emilia Romagna", "Monaco", "Spain", "Canada",
        "Austria", "Britain", "Belgium", "Hungary", "Dutch",
        "Italy", "Azerbaijan", "Singapore", "United States",
        "Mexico City", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
    ]

    if request.track not in valid_tracks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid track name. Valid tracks are: {valid_tracks}"
        )

    grid_values = [entry.grid for entry in request.grid_order]

    if len(grid_values) != len(set(grid_values)):
        raise HTTPException(
            status_code=400,
            detail="Two drivers cannot have the same grid position."
        )

    if any(grid < 1 for grid in grid_values):
        raise HTTPException(
            status_code=400,
            detail="Grid positions must be 1 or higher."
        )

    grid_order = {
        entry.driver: entry.grid for entry in request.grid_order
    }

    next_race["Track"] = request.track
    next_race["Starting Grid"] = next_race["Driver"].map(grid_order)

    if next_race["Starting Grid"].isna().any():
        missing_drivers = next_race[next_race["Starting Grid"].isna()]["Driver"].tolist()
        raise HTTPException(
            status_code=400,
            detail=f"Missing grid positions for: {missing_drivers}"
        )

    X = next_race[
        [
            "Starting Grid",
            "driver_points_before_race",
            "driver_wins_before_race",
            "driver_avg_finish_before_race",
            "team_points_before_race",
            "team_wins_before_race",
            "team_avg_finish_before_race",
            "Driver",
            "Team",
            "Track",
        ]
    ]

    X = pd.get_dummies(X)
    X = X.reindex(columns=model_columns, fill_value=0)

    next_race["win_probability"] = model.predict_proba(X)[:, 1]

    next_race = next_race.sort_values("win_probability", ascending=False)

    return next_race[
        [
            "Driver",
            "Team",
            "Starting Grid",
            "win_probability",
        ]
    ].to_dict(orient="records")