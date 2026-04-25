import os
import re
import time
import random
import unicodedata
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
DATA_FILE = "player_model_df.csv"
CACHE_FILE = "player_awards_cache.csv"

MIN_CAREER_YEARS_FOR_TRAINING = 7

LABEL_NAMES = {
    7: "All-time great",
    6: "Superstar",
    5: "Borderline superstar",
    4: "Quality All-Star",
    3: "Fringe All-Star",
    2: "Starter",
    1: "Role player",
    0: "Out of league"
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0 Safari/537.36"
    )
}


# =========================
# HELPERS
# =========================
def clean_name(name):
    name = str(name)

    # remove *
    name = name.replace("*", "")

    # normalize accents
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

    # remove punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # normalize spaces
    name = " ".join(name.split())

    return name.lower().strip()


def safe_get(url, retries=3):
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r
        except Exception:
            pass

        time.sleep(1)

    return None


def search_player_url(player_name):
    """
    Uses Basketball Reference search.
    """
    cleaned = player_name.replace("*", "").strip()

    url = (
        "https://www.basketball-reference.com/search/search.fcgi"
        f"?search={cleaned.replace(' ', '+')}"
    )

    r = safe_get(url)
    if r is None:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    first_link = soup.select_one("div.search-item-name a")

    if first_link:
        href = first_link["href"]
        return "https://www.basketball-reference.com" + href

    return None


def parse_player_awards(player_name):
    """
    Scrape one player page.
    Returns dict of award counts.
    """

    player_url = search_player_url(player_name)

    if player_url is None:
        return {
            "Player": player_name,
            "MVP": 0,
            "FMVP": 0,
            "AllStar": 0,
            "AllNBA": 0,
            "Top5MVP": 0
        }

    r = safe_get(player_url)
    if r is None:
        return {
            "Player": player_name,
            "MVP": 0,
            "FMVP": 0,
            "AllStar": 0,
            "AllNBA": 0,
            "Top5MVP": 0
        }

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)

    mvp = len(re.findall(r"Most Valuable Player", text))
    fmvp = len(re.findall(r"Finals MVP", text))
    allstar = len(re.findall(r"All-Star", text))
    allnba = len(re.findall(r"All-NBA", text))

    # MVP shares table -> top 5 finishes
    top5 = 0
    try:
        tables = pd.read_html(StringIO(r.text))

        for table in tables:
            cols = [str(c) for c in table.columns]

            if any("MVP" in c for c in cols):
                rank_col = None

                for c in cols:
                    if "Rank" in c or "Rk" in c:
                        rank_col = c
                        break

                if rank_col:
                    ranks = pd.to_numeric(table[rank_col], errors="coerce")
                    top5 = int((ranks <= 5).sum())
                    break
    except Exception:
        pass

    return {
        "Player": player_name,
        "MVP": mvp,
        "FMVP": fmvp,
        "AllStar": allstar,
        "AllNBA": allnba,
        "Top5MVP": top5
    }


def build_awards_cache(players):
    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE)

        cached_players = set(cache["Player"])
        missing = [p for p in players if p not in cached_players]

        rows = cache.to_dict("records")

    else:
        missing = players
        rows = []

    print(f"\nNeed to scrape {len(missing)} players\n")

    for i, player in enumerate(missing, start=1):
        print(f"[{i}/{len(missing)}] {player}")

        info = parse_player_awards(player)
        rows.append(info)

        # polite delay
        time.sleep(random.uniform(1.0, 2.0))

        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(CACHE_FILE, index=False)

    cache_df = pd.DataFrame(rows)
    cache_df.to_csv(CACHE_FILE, index=False)

    return cache_df


def assign_label_percentile(model_df):
    """
    Percentile-based labeling ensures all 8 classes exist.
    """

    scores = model_df["career_score"]

    p99 = np.percentile(scores, 99)
    p97 = np.percentile(scores, 97)
    p90 = np.percentile(scores, 90)
    p75 = np.percentile(scores, 75)
    p60 = np.percentile(scores, 60)
    p40 = np.percentile(scores, 40)
    p20 = np.percentile(scores, 20)

    def label_row(x):
        s = x["career_score"]

        if s >= p99:
            return 7
        elif s >= p97:
            return 6
        elif s >= p90:
            return 5
        elif s >= p75:
            return 4
        elif s >= p60:
            return 3
        elif s >= p40:
            return 2
        elif s >= p20:
            return 1
        else:
            return 0

    return model_df.apply(label_row, axis=1)


def plot_probabilities(probs, idx_to_label):
    labels = [LABEL_NAMES[idx_to_label[i]] for i in range(len(probs))]
    values = [probs[i] * 100 for i in range(len(probs))]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel("Probability (%)")
    plt.title("Career Outcome Probabilities")
    plt.tight_layout()
    plt.show()


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_FILE)

print("Loaded:", len(df), "players")

# =========================
# AWARDS CACHE
# =========================
awards = build_awards_cache(df["Player"].tolist())

# merge
model_df = df.merge(awards, on="Player", how="left")

# label
# =========================
# 🔥 PERCENTILE LABELING
# =========================
model_df["label"] = assign_label_percentile(model_df)

print("\nLabel counts:")
print(model_df["label"].value_counts().sort_index())

# =========================
# 🔥 FORCE CONTIGUOUS LABELS STARTING AT 0
# =========================

model_df["label"] = assign_label_percentile(model_df).astype(int)

print("\nLabel mapping (original → training):")

print("\nNew label counts:")
print(model_df["label"].value_counts().sort_index())


# =========================
# TRAIN SET
# =========================
train_df = model_df.copy()
train_df = train_df[train_df["career_years"] >= MIN_CAREER_YEARS_FOR_TRAINING].copy()

# ✅ Move this filter UP — before defining X and y
class_counts = train_df["label"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
train_df = train_df[train_df["label"].isin(valid_classes)].copy()

print("\nFinal training label distribution:")
print(train_df["label"].value_counts().sort_index())

assert train_df["label"].nunique() >= 2, "Not enough classes in training set!"

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

# After filtering valid_classes and before defining X and y:

# Remap labels to contiguous 0-based integers
unique_labels = sorted(train_df["label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

train_df["label_remapped"] = train_df["label"].map(label_to_idx)

X = train_df[feature_cols]
y = train_df["label_remapped"]  # ✅ use remapped labels for training

# Now train_test_split will work fine
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
# =========================
# TRAIN MODEL
# =========================
num_classes = len(unique_labels)

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, preds), 3))
print("\nClassification Report:")
print(classification_report(y_test, preds, zero_division=0))

# =========================
# PREDICT PLAYER
# =========================
while True:
    player_name = input("\nEnter player name (or quit): ").strip()

    if player_name.lower() == "quit":
        break

    match = model_df[model_df["Player"].str.lower() == player_name.lower()]

    if match.empty:
        print("Player not found.")
        continue

    row = match.iloc[[0]]

    X_player = row[feature_cols]

    probs = model.predict_proba(X_player)[0]

    print("\nCareer Probabilities:\n")
    for idx, prob in enumerate(probs):
        original_label = idx_to_label[idx]  # ✅ map back to original label
        print(f"{LABEL_NAMES[original_label]:22} {prob * 100:6.2f}%")

    plot_probabilities(probs, idx_to_label)
