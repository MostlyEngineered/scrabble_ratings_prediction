import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
SEED = 100
np.random.seed(SEED)

import pandas as pd
import os
# import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from enum import Enum
import xgboost as xgb

from datetime import datetime

# class Bots(enum.Enum):

bots = ["STEEBot", "BetterBot", "HastyBot"]
submission_directory = "submissions"

BOTS = Enum("BOTS", bots)

data_folder = "data"

train_data_file = os.path.join(data_folder, "train.csv")
submission_data_file = os.path.join(data_folder, "test.csv")
turns_data_file = os.path.join(data_folder, "turns.csv")
games_data_file = os.path.join(data_folder, "games.csv")

train_data = pd.read_csv(train_data_file)
submission_data = pd.read_csv(submission_data_file)
turns_data = pd.read_csv(turns_data_file)
games_data = pd.read_csv(games_data_file)

submission_players = submission_data["nickname"].unique()
train_players = train_data["nickname"].unique()

total_players = list(set(train_players + train_players))
print(
    f"Number of submission players, {len(submission_players)}\nNumber of train players, {len(train_players)}\nNumber of total players, {len(total_players)}"
)

games_by_nickname = train_data[["nickname", "score"]].groupby("nickname").count().sort_values("score", ascending=False)


def add_game_metadata(base_data, game_metadata):
    return base_data.merge(game_metadata, on="game_id", how="left")


def is_bot(nickname):
    return nickname in bots


def bot_nickname_to_enum(nickname):
    return BOTS[nickname].value


def string_col_to_enum(nickname):
    return BOTS[nickname].value


def make_xy_data(base_data):
    base_data = add_game_metadata(base_data, games_data)

    is_bot_data = base_data["nickname"].apply(is_bot)
    base_data["is_bot"] = is_bot_data
    player_data = base_data[base_data["is_bot"] == False].reset_index(drop=True)
    bot_data = base_data[base_data["is_bot"] == True].reset_index(drop=True)

    assert len(player_data) == len(bot_data), "bot and player data different lengths"
    player_data["bot_score"] = bot_data["score"]
    player_data["bot_played"] = bot_data["nickname"]
    player_data["bot_number_played"] = player_data["bot_played"].apply(bot_nickname_to_enum)
    player_data["game_margin"] = player_data["score"] - player_data["bot_score"]

    x_columns = [
        "score",
        "rating",
        "time_control_name",
        "game_end_reason",
        "winner",
        "lexicon",
        "initial_time_seconds",
        "increment_seconds",
        "rating_mode",
        "max_overtime_minutes",
        "game_duration_seconds",
        "game_margin",
        "bot_number_played"
    ]

    # x_columns = [
    #     "score",
    #     "winner",
    #     "initial_time_seconds",
    #     "increment_seconds",
    #     "max_overtime_minutes",
    #     "game_duration_seconds",
    #     "game_margin",
    #     "bot_number_played"
    # ]

    X = player_data[x_columns]

    for col in X.columns:
        if isinstance(X[col].dtype, object):
            vals = X[col].unique()
            ENUM_VALS = Enum("BOTS", vals)


    y = player_data["rating"]
    game_ids = player_data['game_id']

    return X, y, game_ids

class StatModel:
    def __init__(self, model, train_test_data):
        self.model = model
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_data
        self.model.fit(self.x_train, self.y_train)
        self.score = self.model.score(self.x_test, self.y_test)

        print(f"For {self.model}, score is: {self.score}")

    def __lt__(self, other):
        return self.score < other.score

    def generate_submission_data(self, output_csv):
        submission = pd.DataFrame([])
        submit_x, submit_y, game_ids = make_xy_data(submission_data)
        submit_y = self.model.predict(submit_x)
        submission['game_id'] = game_ids
        submission['rating'] = submit_y
        submission.to_csv(output_csv, index=False)


if __name__ == "__main__":
    data_x, data_y, _ = make_xy_data(train_data)

    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.85, random_state=SEED)
    train_test_data = train_test_split(data_x, data_y, train_size=0.85, random_state=SEED)

    models = [
        RandomForestRegressor(random_state=0),
        xgb.XGBRegressor()
    ]

    stat_models = [StatModel(x, train_test_data) for x in models]
    stat_models.sort(reverse=True)
    best_model = stat_models[0]

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    submission_file = f"submission_{best_model.__class__.__name__}_{date_time}.csv"

    best_model.generate_submission_data(os.path.join(submission_directory, submission_file))

