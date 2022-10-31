import numpy as np
import pandas as pd


pd.set_option("display.max_columns", None)
SEED = 100
np.random.seed(SEED)

import pandas as pd
import os

import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# from imblearn.under_sampling import RandomUnderSampler

from enum import Enum
import xgboost as xgb

from datetime import datetime
import matplotlib.pyplot as plt

# class Bots(enum.Enum):

bots = ["STEEBot", "BetterBot", "HastyBot"]
submission_directory = "submissions"

BOTS = Enum("BOTS", bots)

data_folder = "data"
histogram_bins = 100

enums = []


def add_game_metadata(base_data, game_metadata):
    return base_data.merge(game_metadata, on="game_id", how="left")


def is_bot(nickname):
    return nickname in bots


def bot_nickname_to_enum(nickname):
    return BOTS[nickname].value


def normal_distribution(x, mu, sigma):
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = (-1 / 2) * (((x - mu) / sigma) ** 2)
    return coeff * np.exp(exponent)


def improve_normalization(player_data, resample_size=1000000):
    # Remove overrepresented data
    counts, bins = np.histogram(player_data["rating"], bins=histogram_bins)
    max_bin = np.argmax(counts)
    bounds = (bins[max_bin], bins[max_bin + 1])
    adj_player_data = player_data.loc[
        (player_data["rating"] <= bins[max_bin]) | (player_data["rating"] >= bins[max_bin + 1])
    ]
    mean_adj_player_data = np.mean(adj_player_data["rating"])
    std_adj_player_data = np.std(adj_player_data["rating"])

    ret_data = pd.DataFrame([], columns=player_data.columns)
    bin_width = bins[1] - bins[0]
    bin_half_width = bin_width / 2
    desired_counts = np.array([], dtype=int)
    for bin in bins:
        bin_midpoint = bin + bin_half_width
        bin_count = int(
            resample_size * normal_distribution(bin_midpoint, mean_adj_player_data, std_adj_player_data) * bin_width
        )
        # print(f"bin count is {bin_count}")
        desired_counts = np.append(desired_counts, bin_count)
        bin_slice = player_data.loc[(player_data["rating"] >= bin) & (player_data["rating"] < (bin + bin_width))]
        if bin_slice.empty:
            continue
        data_multiple = bin_count / len(bin_slice)
        repeat_data = [bin_slice] * int(np.floor(data_multiple)) + [
            bin_slice.sample(int(np.mod(data_multiple, 1) * len(bin_slice)))
        ]
        concat_data = [ret_data] + repeat_data
        ret_data = pd.concat(concat_data, ignore_index=True, axis=0)

    return ret_data


class DataBlock:
    def __init__(self, data_files):
        # Boilerplate reading files
        self.train_ratio = 0.7  # Ratio of total data used for training (vs validation test)
        self.data_files = data_files
        self.initial_data_dict = {k: pd.read_csv(v) for (k, v) in zip(data_files.keys(), data_files.values())}
        self.train_data = self.initial_data_dict["train_data_file"]
        self.initial_submission_file = self.initial_data_dict["submission_data_file"]

        self.preprocessed_train_data = preprocess_train_data(self.train_data, self.initial_data_dict['games_data_file'])

        self.data_x, self.data_y, self.game_ids = make_xy_data(self.preprocessed_train_data)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data_x, self.data_y, train_size=self.train_ratio, random_state=SEED
        )




class StatModel:
    def __init__(self, model, train_test_data):
        self.model = model
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_data
        self.model.fit(self.x_train, self.y_train)
        self.test_predictions = self.model.predict(self.x_test)
        self.score = sklearn.metrics.mean_squared_error(self.y_test, self.test_predictions, squared=False)

        print(f"For {self.model}, score is: {self.score}")
        self.submit_y = None
        self.submission = pd.DataFrame([])

    def __lt__(self, other):
        return self.score < other.score

    def generate_submission_data(self, initial_submission_data, games_data, output_csv):
        submission_data = preprocess_train_data(initial_submission_data, games_data, improve_normalization_flag=False)
        submit_x, submit_y, game_ids = make_xy_data(submission_data)
        self.submit_y = self.model.predict(submit_x)
        self.submission["game_id"] = game_ids
        self.submission["rating"] = self.submit_y
        self.submission.to_csv(output_csv, index=False)

    def plot_error(self, n_bins=100):
        self.errors = np.subtract(self.test_predictions, self.y_test)

        fig, ax = plt.subplots(tight_layout=True)

        hist = ax.hist(self.errors, bins=n_bins)
        ax.set_ylabel("Count")
        ax.set_xlabel("Error")
        ax.set_title("Error")
        plt.show()

    def plot_y_and_pred_hists(self, n_bins=100):
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        # We can set the number of bins with the *bins* keyword argument.
        counts_0, bins_0, bars_0 = axs[0].hist(self.test_predictions, bins=n_bins)
        counts_1, bins_1, bars_1 = axs[1].hist(self.y_test, bins=n_bins)

        axs[0].set_ylabel("Count")
        axs[0].set_title("Predictions")
        axs[1].set_title("Actual")

        ybounds = axs[0].get_ylim()
        ylength = ybounds[1] - ybounds[0]
        # pred_mean = np.mean(self.test_predictions)
        # test_mean = np.mean(self.y_test)
        sigma_multiple = 1.5
        for i, data in enumerate([self.test_predictions, self.y_test]):
            mean_val = np.mean(data)
            st_dev = np.std(data)
            axs[i].plot([mean_val, mean_val], list(ybounds), linestyle="--", color="black")
            mean_text = f"mean\n{mean_val:.1f}"
            xbounds = axs[i].get_xlim()
            xlength = xbounds[1] - xbounds[0]
            axs[i].text(mean_val + (xlength * 0.05), ybounds[0] + (ylength * 0.85), mean_text)
            plus_s = mean_val + (sigma_multiple * st_dev)
            minus_s = mean_val - (sigma_multiple * st_dev)
            axs[i].plot([plus_s, plus_s], list(ybounds), linestyle="--", color="gray")
            axs[i].plot([minus_s, minus_s], list(ybounds), linestyle="--", color="gray")
            plus_s_text = f"+{sigma_multiple}s\n{plus_s:.1f}"
            minus_s_text = f"-{sigma_multiple}s\n{minus_s:.1f}"
            axs[i].text(plus_s + (xlength * 0.05), ybounds[0] + (ylength * 0.7), plus_s_text)
            axs[i].text(minus_s + (xlength * 0.05), ybounds[0] + (ylength * 0.7), minus_s_text)

        # axs[0].plot([pred_mean, pred_mean], list(ybounds), linestyle='--', color='black')
        # axs[1].plot([test_mean, test_mean], list(ybounds), linestyle='--', color='black')
        # pred_plus_3s = pred_mean + 3 * np.std(self.test_predictions)
        # pred_minus_3s = pred_mean - 3 * np.std(self.test_predictions)
        # test_plus_3s = test_mean + 3 * np.std(self.y_test)
        # test_minus_3s = test_mean - 3 * np.std(self.y_test)

        axs[0].set_ylim(ybounds[0], ybounds[1])

        plt.show()

    def plot_score_vs_rating(self):
        fig, ax = plt.subplots(tight_layout=True)
        plt.scatter(self.x_train["score"], self.y_train, alpha=0.025)
        ax.set_ylabel("Rating")
        ax.set_xlabel("Score")

        plt.show()


class AnalysisBlock:
    def __init__(self, data_files, models):
        self.data = DataBlock(data_files)
        self.models = models

        train_test_data = self.data.x_train, self.data.x_test, self.data.y_train, self.data.y_test
        self.stat_models = [StatModel(x, train_test_data) for x in models]
        self.stat_models.sort(reverse=True)
        self.best_model = self.stat_models[0]

    def save_submission_file(self):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        submission_file = f"submission_{self.best_model.__class__.__name__}_{date_time}.csv"

        self.best_model.generate_submission_data(
            self.data.initial_data_dict["submission_data_file"], self.data.initial_data_dict["games_data_file"], os.path.join(submission_directory, submission_file)
        )

    def plots(self):
        self.best_model.plot_error()
        self.best_model.plot_y_and_pred_hists()
        self.best_model.plot_score_vs_rating()



def preprocess_train_data(base_data, games_data, improve_normalization_flag=True):

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

    if improve_normalization_flag:
        player_data = improve_normalization(player_data)

    return player_data

def make_xy_data(base_data):
    global enums

    x_columns = [
        "score",
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
        "bot_number_played",
    ]

    X = base_data[x_columns]
    X = X.infer_objects()
    string_data = X.select_dtypes(exclude=[np.number, int, float])

    enum_data = pd.DataFrame([])
    for col in string_data.columns:
        if isinstance(string_data[col].dtype, object):
            vals = string_data[col].unique()
            enum_col = col + "_enum"
            ENUM_VALS = Enum(enum_col, vals.tolist())
            enums = enums + [ENUM_VALS]
            enum_data[enum_col] = string_data[col].apply(lambda x: (ENUM_VALS[x].value))

    X = X.drop(columns=string_data.columns.to_list())
    X_initial_columns = X.columns
    X = pd.concat([X, enum_data], ignore_index=True, axis=1)
    X.columns = X_initial_columns.to_list() + enum_data.columns.to_list()

    y = base_data["rating"]
    game_ids = base_data["game_id"]

    return X, y, game_ids


if __name__ == "__main__":
    train_data_file = os.path.join(data_folder, "train.csv")
    submission_data_file = os.path.join(data_folder, "test.csv")
    turns_data_file = os.path.join(data_folder, "turns.csv")
    games_data_file = os.path.join(data_folder, "games.csv")

    data_files = {
        "train_data_file": train_data_file,
        "submission_data_file": submission_data_file,
        "turns_data_file": turns_data_file,
        "games_data_file": games_data_file,
    }

    models = [RandomForestRegressor(random_state=0), xgb.XGBRegressor()]
    analysis = AnalysisBlock(data_files, models)
    analysis.save_submission_file()
    analysis.plots()
