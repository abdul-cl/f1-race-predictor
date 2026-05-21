import pandas as pd

# LOAD DATA
df = pd.read_csv("data/Formula1_2024season_raceResults.csv")

# SORT BY RACE ORDER
df = df.sort_values(["Track"])

# CLEAN POSITION
df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

# CREATE TARGET
df["won_race"] = df["Position"].apply(
    lambda x: 1 if x == 1 else 0
)

# RUNNING DRIVER POINTS BEFORE EACH RACE
df["driver_points_before_race"] = (
    df.groupby("Driver")["Points"]
    .cumsum()
    - df["Points"]
)

# DRIVER WINS BEFORE EACH RACE
df["driver_wins_before_race"] = (
    df.groupby("Driver")["won_race"]
    .cumsum()
    - df["won_race"]
)

# DRIVER AVG FINISH BEFORE EACH RACE
df["driver_avg_finish_before_race"] = (
    df.groupby("Driver")["Position"]
    .expanding()
    .mean()
    .shift()
    .reset_index(level=0, drop=True)
)

# FILL NaN VALUES
df["driver_avg_finish_before_race"] = (
    df["driver_avg_finish_before_race"]
    .fillna(df["Position"].mean())
)

# KEEP IMPORTANT COLUMNS
clean_df = df[
    [
        "Track",
        "Driver",
        "Team",
        "Starting Grid",
        "driver_points_before_race",
        "driver_wins_before_race",
        "driver_avg_finish_before_race",
        "won_race",
    ]
]

# SAVE DATASET
clean_df.to_csv("data/clean_f1_data.csv", index=False)

print(clean_df.head())

print("\nDataset created successfully!")