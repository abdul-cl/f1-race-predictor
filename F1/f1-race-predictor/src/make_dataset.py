import pandas as pd

seasons = [2022, 2023, 2024, 2025]

all_seasons = []

for year in seasons:
    file_path = f"data/Formula1_{year}season_raceResults.csv"

    try:
        df = pd.read_csv(file_path)
        df["Year"] = year
        all_seasons.append(df)
        print(f"Loaded {year}")
    except FileNotFoundError:
        print(f"Missing file for {year}: {file_path}")

df = pd.concat(all_seasons, ignore_index=True)

df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
df["Starting Grid"] = pd.to_numeric(df["Starting Grid"], errors="coerce")

df = df.dropna(subset=["Position", "Points", "Starting Grid"])

df["won_race"] = df["Position"].apply(lambda x: 1 if x == 1 else 0)

df = df.sort_values(["Year", "Track"])

df["driver_points_before_race"] = (
    df.groupby(["Year", "Driver"])["Points"].cumsum() - df["Points"]
)

df["driver_wins_before_race"] = (
    df.groupby(["Year", "Driver"])["won_race"].cumsum() - df["won_race"]
)

df["driver_avg_finish_before_race"] = (
    df.groupby(["Year", "Driver"])["Position"]
    .expanding()
    .mean()
    .shift()
    .reset_index(level=[0, 1], drop=True)
)

df["driver_avg_finish_before_race"] = df["driver_avg_finish_before_race"].fillna(
    df["Position"].mean()
)

df["team_points_before_race"] = (
    df.groupby(["Year", "Team"])["Points"].cumsum() - df["Points"]
)

df["team_wins_before_race"] = (
    df.groupby(["Year", "Team"])["won_race"].cumsum() - df["won_race"]
)

df["team_avg_finish_before_race"] = (
    df.groupby(["Year", "Team"])["Position"]
    .expanding()
    .mean()
    .shift()
    .reset_index(level=[0, 1], drop=True)
)

df["team_avg_finish_before_race"] = df["team_avg_finish_before_race"].fillna(
    df["Position"].mean()
)

clean_df = df[
    [
        "Year",
        "Track",
        "Driver",
        "Team",
        "Starting Grid",
        "driver_points_before_race",
        "driver_wins_before_race",
        "driver_avg_finish_before_race",
        "won_race",
        "team_points_before_race",
        "team_wins_before_race",
        "team_avg_finish_before_race"
    ]
]

clean_df.to_csv("data/clean_f1_data.csv", index=False)

print(clean_df.head())
print("\nWin counts:")
print(clean_df["won_race"].value_counts())
print("\nSaved multi-season training data!")