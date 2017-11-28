import sqlite3
import pandas as pd
import numpy as np
import datetime as DT

conn = sqlite3.connect("database.sqlite")

'''
cur = conn.cursor()
#cur.execute("select * from Country;")
#cur.execute("select * from League;")
#cur.execute("select * from Match;")
#cur.execute("select * from Player;")
cur.execute("select * from Player_Attributes;")
#cur.execute("select * from sqlite_sequence;")
#cur.execute("select * from Team;")
#cur.execute("select * from Team_Attributes;")
results = cur.fetchall()
'''

#df_country = pd.read_sql_query("select * from Country;", conn)
#df_league = pd.read_sql_query("select * from League;", conn)
#df_match = pd.read_sql_query("select * from Match;", conn)

df = pd.read_sql_query("select * from Player;", conn)


#df_player_attributes = pd.read_sql_query("select * from Player_Attributes;", conn)

#df_sqlite_sequence = pd.read_sql_query("select * from sqlite_sequence;", conn)
#df_team = pd.read_sql_query("select * from Team;", conn)
#df_team_attributes = pd.read_sql_query("select * from Team_Attributes;", conn)

#df_player_attributes = df_player[df_player_attributes['date'] > '2016-01-01']
#df_player_attributes = df_player_attributes.drop('date', axis = 1)
now = '2016-06-01'

df['birthday'] = df['birthday'].apply('{:06}'.format)

now = pd.Timestamp(DT.datetime.now())
df['birthday'] = pd.to_datetime(df['birthday'], format='%m%d%y')    # 1
df['birthday'] = df['birthday'].where(df['birthday'] < now, df['birthday'] -  np.timedelta64(100, 'Y'))   # 2
df['age'] = (now - df['birthday']).astype('<m8[Y]')    # 3
print(df)
#df_player['age']=now - df_player['birthday']
#df_player.to_csv('players.csv', sep=',', header=True, index=False)

#cur.close()
conn.close()
