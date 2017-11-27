import pandas as pd


fifa2018 = pd.read_csv('fifa2018.csv')                         #########
useful_columns = ['age', 'height_cm', 'weight_kg', 'overall', 'potential', 'pac', 'sho', 'pas', 'dri', 'def', 'phy', 'skill_moves', 'weak_foot', 'work_rate_att', 'work_rate_def', 'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression', 'interceptions', 'positioning', 'vision', 'penalties', 'composure', 'marking', 'standing_tackle', 'sliding_tackle']
english_premier_league_players = fifa2018[fifa2018['league'] == 'English Premier League']
print(english_premier_league_players[useful_columns])