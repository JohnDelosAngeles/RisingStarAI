import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

DATA_FILE = "nba_advanced_df.csv"


def weighted_avg(x, col):
    weights = x["MP"] * x["MPG"]
    return (x[col] * weights).sum() / weights.sum()


df = pd.read_csv(DATA_FILE)

numeric_cols = [
    "Age", "AST%", "STL%", "BLK%", "USG%", "TOV%",
    "Rel TS%", "BPM", "PTS per 100", "ORB%", "DRB%",
    "FTr", "MPG", "GS%", "MP", "Year"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

early_df = df[df["Age"] <= 22]
early_df = early_df.groupby("Player").filter(lambda x: len(x) >= 2)

grouped = early_df.groupby("Player")

knn_df = pd.DataFrame()
knn_df["weighted_AST%"] = grouped.apply(lambda x: weighted_avg(x, "AST%"))
knn_df["weighted_STL%"] = grouped.apply(lambda x: weighted_avg(x, "STL%"))
knn_df["weighted_BLK%"] = grouped.apply(lambda x: weighted_avg(x, "BLK%"))
knn_df["weighted_USG%"] = grouped.apply(lambda x: weighted_avg(x, "USG%"))
knn_df["weighted_TOV%"] = grouped.apply(lambda x: weighted_avg(x, "TOV%"))
knn_df["weighted_Rel_TS%"] = grouped.apply(lambda x: weighted_avg(x, "Rel TS%"))
knn_df["weighted_BPM"] = grouped.apply(lambda x: weighted_avg(x, "BPM"))
knn_df["weighted_PTS_per_100"] = grouped.apply(lambda x: weighted_avg(x, "PTS per 100"))
knn_df["weighted_ORB%"] = grouped.apply(lambda x: weighted_avg(x, "ORB%"))
knn_df["weighted_DRB%"] = grouped.apply(lambda x: weighted_avg(x, "DRB%"))
knn_df["weighted_FTr"] = grouped.apply(lambda x: weighted_avg(x, "FTr"))
knn_df["avg_MPG"] = grouped["MPG"].mean()
knn_df["weighted_GS%"] = grouped.apply(lambda x: weighted_avg(x, "GS%"))
knn_df["weighted_Age"] = grouped.apply(lambda x: weighted_avg(x, "Age"))

knn_df = knn_df.reset_index()

career_grouped = df.groupby("Player")

career_df = pd.DataFrame()
career_df["peak_BPM"] = career_grouped["BPM"].max()
career_df["career_MPG"] = career_grouped["MPG"].mean()
career_df["career_years"] = career_grouped["Year"].nunique()

career_df["career_score"] = (
    career_df["peak_BPM"] * 0.5 +
    career_df["career_MPG"] * 0.2 +
    career_df["career_years"] * 0.3
)

career_df = career_df.reset_index()

final_df = knn_df.merge(career_df, on="Player")

feature_cols = [
    "weighted_AST%",
    "weighted_STL%",
    "weighted_BLK%",
    "weighted_USG%",
    "weighted_TOV%",
    "weighted_Rel_TS%",
    "weighted_BPM",
    "weighted_PTS_per_100",
    "weighted_ORB%",
    "weighted_DRB%",
    "weighted_FTr",
    "avg_MPG",
    "weighted_GS%",
    "weighted_Age"
]

X = final_df[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

feature_weights = np.array([
    1.2,
    1.2,
    0.8,
    1.5,
    1.2,
    1.5,
    2.0,
    2.0,
    0.7,
    0.7,
    1.0,
    1.0,
    1.3,
    1.0,
])

X_scaled = X_scaled * feature_weights

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_scaled)


def get_career_projection(player_name):
    matches = final_df[final_df["Player"].str.lower() == player_name.lower()]

    if matches.empty:
        return None

    player_history = df[df["Player"].str.lower() == player_name.lower()].copy()
    player_history = player_history.sort_values("Age")

    player_index = matches.index[0]

    distances, indices = knn.kneighbors([X_scaled[player_index]])

    neighbor_idx = indices[0][1:]
    neighbor_dist = distances[0][1:]

    similar_players = final_df.iloc[neighbor_idx].copy()
    similar_names = similar_players["Player"].values

    weights = 1 / (neighbor_dist + 0.1)
    similar_players["weight"] = weights

    weighted_prediction = (
        np.sum(similar_players["career_score"] * weights) / np.sum(weights)
    )

    early_curve = player_history[player_history["Age"] <= 22][["Age", "BPM"]]

    similar_careers = df[df["Player"].isin(similar_names)].copy()

    weighted_curves = similar_careers.merge(
        similar_players[["Player", "weight"]],
        on="Player"
    )

    weighted_curves = weighted_curves[weighted_curves["Age"] > 22]

    avg_future_curve = (
        weighted_curves.groupby("Age")
        .apply(lambda x: np.average(x["BPM"], weights=x["weight"]))
        .reset_index(name="BPM")
    )

    similar_output = similar_players[
        ["Player", "peak_BPM", "career_years", "career_score"]
    ].head(5)

    return {
        "projected_score": round(float(weighted_prediction), 2),
        "similar_players": similar_output.to_dict(orient="records"),
        "actual_curve": early_curve.to_dict(orient="records"),
        "projected_curve": avg_future_curve.to_dict(orient="records")
    }