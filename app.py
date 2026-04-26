from flask import Flask, render_template, request

app = Flask(__name__)

# Temporary sample data so the search bar works now.
# Later you can replace this with live NBA API data or your teammate's backend.
sample_players = {
    "lebron james": {
        "name": "LeBron James",
        "team": "Los Angeles Lakers",
        "pts": 27.1,
        "reb": 7.5,
        "ast": 7.4,
        "projection": "All-time great / superstar career path"
    },
    "stephen curry": {
        "name": "Stephen Curry",
        "team": "Golden State Warriors",
        "pts": 24.8,
        "reb": 4.7,
        "ast": 6.4,
        "projection": "Elite shooter / generational offensive player"
    },
    "jayson tatum": {
        "name": "Jayson Tatum",
        "team": "Boston Celtics",
        "pts": 27.0,
        "reb": 8.1,
        "ast": 4.9,
        "projection": "Franchise star / MVP-level upside"
    },
    "nikola jokic": {
        "name": "Nikola Jokic",
        "team": "Denver Nuggets",
        "pts": 26.4,
        "reb": 12.4,
        "ast": 9.0,
        "projection": "Hall of Fame big / offensive engine"
    },
    "giannis antetokounmpo": {
        "name": "Giannis Antetokounmpo",
        "team": "Milwaukee Bucks",
        "pts": 30.1,
        "reb": 11.5,
        "ast": 6.2,
        "projection": "Hall of Fame two-way superstar"
    },
    "luka doncic": {
        "name": "Luka Doncic",
        "team": "Dallas Mavericks",
        "pts": 33.2,
        "reb": 9.1,
        "ast": 9.8,
        "projection": "Generational heliocentric superstar"
    }
}

top_players = [
    {
        "name": "Nikola Jokic",
        "pts": 26.4,
        "reb": 12.4,
        "ast": 9.0
    },
    {
        "name": "Giannis Antetokounmpo",
        "pts": 30.1,
        "reb": 11.5,
        "ast": 6.2
    },
    {
        "name": "Luka Doncic",
        "pts": 33.2,
        "reb": 9.1,
        "ast": 9.8
    },
    {
        "name": "Jayson Tatum",
        "pts": 27.0,
        "reb": 8.1,
        "ast": 4.9
    },
    {
        "name": "Stephen Curry",
        "pts": 26.3,
        "reb": 4.5,
        "ast": 5.1
    }
]


@app.route("/")
def home():
    return render_template("home.html", players=top_players)


@app.route("/search")
def search():
    player_name = request.args.get("player", "").strip()
    player = sample_players.get(player_name.lower())

    if player:
        return render_template("player.html", player=player, search_name=player_name)

    return render_template("player.html", player=None, search_name=player_name)


if __name__ == "__main__":
    app.run(debug=True)