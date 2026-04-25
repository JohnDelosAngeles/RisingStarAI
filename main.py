import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from time import sleep

'''variable checklist
Advanced - assist %, steal %, block %, PER, Usage %, drb %, orb %, mins played, age, turnover %, relative ts %, FTr,
BPM
Per game - minutes per game, Games Started %
Per 100 possessions - points per 100 poss
'''


def clean_traded_players(df):
    team_col = "Team" if "Team" in df.columns else "Tm"
    df["is_total"] = df["Team"].str.contains("TM", na=False)
    df = df.sort_values(["Player", "is_total"])
    df = df.drop_duplicates(subset=["Player"], keep="last")
    return df.drop(columns=["is_total"])


BUILD_DATA = False

if BUILD_DATA:
    start_year = 1990
    end_year = 2026
    all_data = []
    for year in range(start_year, end_year):
        # -----ADVANCED-------
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find(name='table', id='advanced')
            if table:
                print("table found!")
                header = [th.text for th in table.find_all('tr', limit=1)[0].find_all('th')[1:]]
                '''= ["AST%", "STL%", "BLK%", "PER", "USG%", "DRB%", "ORB%", "MP", "Age", "TOV%", "TS%", "FRr", "BPM",
                           "Games Started%"]'''
                rows = [row for row in table.find_all('tr')[1:] if row.find_all('td')]
            else:
                print('table not found')

            league_ts = np.nan

            # Find the league average row
            league_row = table.find("td", {"data-stat": "name_display"}, string="League Average")

            if league_row:
                parent_row = league_row.find_parent("tr")

                ts_cell = parent_row.find("td", {"data-stat": "ts_pct"})

                if ts_cell and ts_cell.has_attr("csk"):
                    league_ts = float(ts_cell["csk"])
                else:
                    league_ts = float(ts_cell.text)  # fallback

            data = []
            for row in rows:
                cols = row.find_all('td')
                if not cols:
                    continue

                cols = [ele.text.strip() for ele in cols]

                if cols[-1] == '':
                    cols[-1] = np.nan

                data.append(cols)

            df = pd.DataFrame(data, columns=header)

            df = clean_traded_players(df)
            # Get relative True Shooting %
            df["TS%"] = pd.to_numeric(df["TS%"], errors="coerce")
            df["Rel TS%"] = ((df["TS%"] - league_ts) * 100).round(4)

            cols_to_keep = ["Player", "Age", "AST%", "STL%", "BLK%", "PER", "USG%", "DRB%", "ORB%", "MP", "TOV%",
                            "Rel TS%", "FTr", "BPM"]
            df_advanced = df[cols_to_keep].copy()
            df_advanced["Year"] = year

            time.sleep(3)

            # -----PER GAME-------
            url_pg = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
            df_pg = pd.read_html(url_pg)[0]

            df_pg = clean_traded_players(df_pg)

            # Convert to numeric
            df_pg["MPG"] = pd.to_numeric(df_pg["MP"], errors="coerce")
            df_pg["G"] = pd.to_numeric(df_pg["G"], errors="coerce")
            df_pg["GS"] = pd.to_numeric(df_pg["GS"], errors="coerce")

            # Features
            df_pg["GS%"] = (df_pg["GS"] / df_pg["G"]).round(3)

            df_pg["Year"] = year
            df_pg = df_pg[["Player", "Year", "MPG", "GS%"]]
            time.sleep(3)
            # -----PER 100 POSS-------
            per_100_url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html'
            df_per100 = pd.read_html(per_100_url)[0]

            df_per100 = clean_traded_players(df_per100)
            df_per100["PTS per 100"] = pd.to_numeric(df_per100["PTS"], errors="coerce")
            df_per100["Year"] = year
            df_per100 = df_per100[["Player", "Year", "PTS per 100"]]
            print(df_per100)

            # clean names
            df_advanced["Player"] = df_advanced["Player"].str.strip()
            df_pg["Player"] = df_pg["Player"].str.strip()
            df_per100["Player"] = df_per100["Player"].str.strip()

            # ---- MERGE ALL THREE -----
            df_merged = df_advanced.merge(df_pg, on=["Player", "Year"], how="inner")
            df_merged = df_merged.merge(df_per100, on=["Player", "Year"], how="inner")
            all_data.append(df_merged)

    final_df = pd.concat(all_data, ignore_index=True)
    if os.path.exists("nba_advanced_df.csv"):
        os.remove("nba_advanced_df.csv")
    final_df.to_csv("nba_advanced_df.csv", index=False)

# Filter to early career before condensing
df = pd.read_csv("nba_advanced_df.csv")

# Filter to early career
early_df = df[df["Age"] <= 22]

# 🔥 NEW: remove players with too little early data
early_df = early_df.groupby("Player").filter(lambda x: len(x) >= 2)

grouped = early_df.groupby("Player")

knn_df = pd.DataFrame()


def weighted_avg(x, col):
    return (x[col] * x["MP"] * x["MPG"]).sum() / (x["MP"] * x["MPG"]).sum()


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

# Build career outcome dataset (Full Career)
career_grouped = df.groupby("Player")

career_df = pd.DataFrame()

career_df["peak_BPM"] = career_grouped["BPM"].max()
career_df["career_MPG"] = career_grouped["MPG"].mean()
career_df["career_years"] = career_grouped["Year"].nunique()

# 🔥 NEW: combine into one clean target
career_df["career_score"] = (
    career_df["peak_BPM"] * 0.5 +
    career_df["career_MPG"] * 0.2 +
    career_df["career_years"] * 0.3
)

career_df = career_df.reset_index()

# Merge them
final_df = knn_df.merge(career_df, on="Player")

X = final_df.drop(columns=[
    "Player",
    "career_score",
    "peak_BPM",
    "career_MPG",
    "career_years"
])
y = final_df["career_score"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔥 Add feature weights (must match column order in X)
feature_weights = np.array([
    1.2,  # weighted_AST%
    1.2,  # weighted_STL%
    0.8,  # weighted_BLK%
    1.5,  # weighted_USG%
    1.2,  # weighted_TOV%
    1.5,  # weighted_Rel_TS%
    2.0,  # weighted_BPM  <-- VERY important
    2.0,  # weighted_PTS_per_100
    0.7,  # weighted_ORB%
    0.7,  # weighted_DRB%
    1.0,  # weighted_FTr
    1.0,  # avg_MPG
    1.3,  # weighted_GS%
    1.0,  # weighted_Age
])
print(f'Here are the x columns: {X.columns}')
# Apply weights
X_scaled = X_scaled * feature_weights

# Fit KNN
knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_scaled)

# Find similar players
player_name = input("Enter a player you want to analyze: ")

player_history = df[df["Player"].str.lower() == player_name.lower()].copy()
player_history = player_history.sort_values("Age")

early_curve = player_history[player_history["Age"] <= 22][["Age", "BPM"]]

matches = final_df[final_df["Player"].str.lower() == player_name.lower()]

if matches.empty:
    print("Player not found. Try one of these:")
    print(final_df["Player"].sample(10).values)
    exit()

player_index = matches.index[0]
# player_index = final_df[final_df["Player"] == player_name].index[0]

distances, indices = knn.kneighbors([X_scaled[player_index]])

# skip self (first result)
neighbor_idx = indices[0][1:]
neighbor_dist = distances[0][1:]

similar_players = final_df.iloc[neighbor_idx].copy()

similar_names = similar_players["Player"].values

similar_careers = df[df["Player"].isin(similar_names)].copy()
print(similar_players[["Player", "peak_BPM", "career_years"]])

# Predict career outcome
weights = 1 / (neighbor_dist + 0.1)  # avoid division by 0
similar_players["weight"] = weights

weighted_curves = similar_careers.merge(
    similar_players[["Player", "weight"]],
    on="Player"
)

# only future years
weighted_curves = weighted_curves[weighted_curves["Age"] > 22]

avg_future_curve = (
    weighted_curves.groupby("Age")
    .apply(lambda x: np.average(x["BPM"], weights=x["weight"]))
    .reset_index(name="BPM")
)

weighted_prediction = np.sum(similar_players["career_score"] * weights) / np.sum(weights)

print("Projected Career Score:", weighted_prediction)

plt.figure()

# Player actual (solid line)
plt.plot(early_curve["Age"], early_curve["BPM"], marker='o', label="Actual (<=22)")

# Predicted future (dashed)
plt.plot(avg_future_curve["Age"], avg_future_curve["BPM"], linestyle='--', marker='o', label="Projected (23+)")

plt.xlabel("Age")
plt.ylabel("BPM")
plt.title(f"{player_name} Career Projection (BPM)")
plt.legend()

plt.show()
final_df.to_csv("player_model_df.csv", index=False)