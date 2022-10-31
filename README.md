# scrabble_ratings_prediction

Competition at
https://www.kaggle.com/competitions/scrabble-player-rating

Download data into data folder in this project:
https://www.kaggle.com/competitions/scrabble-player-rating/data

# Progression Pathway
1) Initial submission to check format is right and that problem is correctly understood
2) Get general strings to enums to work
3) Simple dictionary usage from turns to generate statistics of dictionary usage (good players probably use obscure words)
4) Generate point statistics of turns to plug into model
5) Details point responses between players in turns to generate how much better they are (can they control endgames)

# Observations
There are several oversampled ratings in the dataset.  Some form of normalization will help remove this bias.
From looking at the charts the curve should be fit to 1444 to 2135 being 2 sigma margin
