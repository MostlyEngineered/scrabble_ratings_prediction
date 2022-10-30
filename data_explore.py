import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
SEED = 100
np.random.seed(SEED)

import pandas as pd
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from enum import Enum

# class Bots(enum.Enum):

bots = ["STEEBot", "BetterBot", "HastyBot"]

BOTS = Enum("BOTS", bots)

data_folder = "data"

train_data_file = os.path.join(data_folder, "train.csv")
test_data_file = os.path.join(data_folder, "test.csv")
turns_data_file = os.path.join(data_folder, "turns.csv")
games_data_file = os.path.join(data_folder, "games.csv")

train_data = pd.read_csv(train_data_file)
test_data = pd.read_csv(test_data_file)
turns_data = pd.read_csv(turns_data_file)
games_data = pd.read_csv(games_data_file)

test_players = test_data["nickname"].unique()
train_players = train_data["nickname"].unique()

total_players = list(set(train_players + train_players))
print(
    f"Number of test players, {len(test_players)}\nNumber of train players, {len(train_players)}\nNumber of total players, {len(total_players)}"
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

    # x_columns = [
    #     "score",
    #     "rating",
    #     "time_control_name",
    #     "game_end_reason",
    #     "winner",
    #     "lexicon",
    #     "initial_time_seconds",
    #     "increment_seconds",
    #     "rating_mode",
    #     "max_overtime_minutes",
    #     "game_duration_seconds",
    #     "game_margin",
    #     "bot_number_played"
    # ]

    x_columns = [
        "score",
        "rating",
        "winner",
        "initial_time_seconds",
        "increment_seconds",
        "max_overtime_minutes",
        "game_duration_seconds",
        "game_margin",
        "bot_number_played"
    ]

    X = player_data[x_columns]
    y = player_data["rating"]

    return X, y


if __name__ == "__main__":

    data_x, data_y = make_xy_data(train_data)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.7, random_state=SEED)


    rf_model = RandomForestRegressor(max_depth=2, random_state=0)
    rf_model.fit(x_train, y_train)
