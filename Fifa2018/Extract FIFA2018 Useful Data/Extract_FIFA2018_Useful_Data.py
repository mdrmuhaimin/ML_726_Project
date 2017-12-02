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
                    'work_rate_def']

fifa2018 = pd.read_csv('fifa2018.csv')                         

players = fifa2018[useful_columns]
filtered_players = players[players['ID'].isin(id['fifa_api_id'])]

print(filtered_players)

players.to_csv('filtered_fifa2018_players.csv')
