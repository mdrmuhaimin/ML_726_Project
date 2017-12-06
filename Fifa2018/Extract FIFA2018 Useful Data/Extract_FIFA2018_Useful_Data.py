import pandas as pd

fifa2018 = pd.read_csv('fifa2018.csv')   
id = pd.read_csv('fifa_api_id.csv')
useful_columns = ['ID', 
                'age',
                'height_cm',
                'weight_kg',
                'overall',
                'pac',
                'sho',
                'pas',
                'dri',
                'def',
                'phy',
                'skill_moves',
                'weak_foot',
                'work_rate_att',
                'work_rate_def',
                'crossing',
                'finishing',
                'heading_accuracy',
                'short_passing',
                'volleys',
                'dribbling',
                'curve',
                'free_kick_accuracy',
                'long_passing',
                'ball_control',
                'acceleration',
                'sprint_speed',
                'agility',
                'reactions',
                'balance',
                'shot_power',
                'jumping',
                'stamina',
                'strength',
                'long_shots',
                'aggression',
                'interceptions',
                'positioning',
                'vision',
                'penalties',
                'composure',
                'marking',
                'standing_tackle',
                'sliding_tackle']

fifa2018 = pd.read_csv('fifa2018.csv')                         

players = fifa2018[useful_columns]
filtered_players = players[players['ID'].isin(id['fifa_api_id'])]

print(filtered_players)

filtered_players.to_csv('f18_players.csv')
