import pandas as pd
import os
import matplotlib
import numpy as np
import re
import string
import enum

# ENUM_VALS = Enum(enum_col, vals.tolist())

data_folder = 'data'

class Board:
    def __init__(self):
        self.grid = np.empty([15, 15], dtype=str)
        self.bonus = np.empty([15, 15], dtype=object)

        self.populate_bonus()

    def add_word(self, position, word):
        # Pattern must be one number and one letter
        number = re.findall('\d+', position)[0]
        letter = re.findall('\D+', position)[0]
        num_position = position.find(number)
        if num_position == 0:
            # Number is first
            direction = 'right'
        else:
            # Letter is first
            direction = 'down'
        letter_pos = string.ascii_uppercase.index(letter)
        number_pos = int(number) - 1

        i = letter_pos
        j = number_pos
        for letter in word:
            self.grid[j, i] = letter
            if direction == 'right':
                i += 1
            else:
                j += 1

    def populate_bonus(self):
        TripleWords = [[0, 0], [0, 7], [0, 14], [14, 0], [14, 7], [14, 14], [7, 0], [7, 14]]
        DoubleWords = [[1, 1], [2, 2], [3, 3], [4, 4],
                       [10, 10], [11, 11], [12, 12], [13, 13],
                       [1, 13], [2, 12], [3, 11], [4, 10],
                       [13, 1], [12, 2], [11, 3], [10, 4]]
        TripleLetter = []


        DoubleLetter = [[0, 3], [3, 0], [0, 11], [11, 0],
                        [14, 3], [3, 14], [14, 11], [11, 14],
                        [7, 3], [3, 7], [7, 11], [11, 7],
                        [2, 6], [6, 2], [12, 6], [6, 12],
                        [2, 8], [8, 2], [12, 8], [8, 12],
                        [6, 8], [8, 6], [6, 6], [8, 8]
                        ]
        bonus_tiles = {'3WS': TripleWords, '2WS': DoubleWords, '2LS': DoubleLetter}
        for key in bonus_tiles.keys():
            for tile in bonus_tiles[key]:
                self.bonus[tile[0], tile[1]] = key
        # self.bonus[0, 0], self.bonus[0, 7], self.bonus[0, 14], self.bonus[14, 0], self.bonus[14, 7], self.bonus[14, 14], self.bonus[7, 0], self.bonus[7, 14] = ['3WS'] * 8
        # self.bonus[[0, 0], [0, 7], [0, 14], [14, 0], [14, 7], [14, 14], [7, 0], [7, 14]]

if __name__ == "__main__":
    board = Board()
    board.add_word('C2', 'DAY')
    turns_data_file = os.path.join(data_folder, "turns.csv")
    turns_data = pd.read_csv(turns_data_file)

    games = turns_data['game_id'].unique()

    for game in games:
        game_info = turns_data.loc[turns_data['game_id'] == game]
