from pyspark.sql import SparkSession, functions, types, Row
from datetime import datetime
import os

spark = SparkSession.builder.appName('Fifa_16_Processor').getOrCreate()
sc = spark.sparkContext
assert sc.version >= '2.2'  # make sure we have Spark 2.2+
EPL_15_16_Teams = ['Chelsea', 'Arsenal', 'Manchester United', 'Manchester City',
                   'Tottenham Hotspur', 'Liverpool', 'Everton', 'Stoke City', 'West Ham United', 'Leicester City', 'Crystal Palace',
                   'Watford', 'Southampton', 'Swansea City', 'Sunderland', 'West Bromwich Albion', 'Burnley', 'Middlesbrough',
                   'Hull City', 'Bournemouth']

f16_f17_Column_translation = [
    ('player_name', 'Name'),
    ('overall_rating', 'Rating'),
    ('crossing', 'Crossing'),
    ('finishing', 'Finishing'),
    ('heading_accuracy', 'Heading'),
    ('short_passing', 'Short_Pass'),
    ('volleys', 'Volleys'),
    ('dribbling', 'Dribbling'),
    ('curve', 'Curve'),
    ('free_kick_accuracy', 'Freekick_Accuracy'),
    ('long_passing', 'Long_Pass'),
    ('ball_control', 'Ball_Control'),
    ('acceleration', 'Acceleration'),
    ('sprint_speed', 'Speed'),
    ('agility', 'Agility'),
    ('reactions', 'Reactions'),
    ('balance', 'Balance'),
    ('shot_power', 'Shot_Power'),
    ('jumping', 'Jumping'),
    ('stamina', 'Stamina'),
    ('strength', 'Strength'),
    ('long_shots', 'Long_shots'),
    ('aggression', 'Aggression'),
    ('interceptions', 'Interceptions'),
    ('positioning', 'Attacking_Position'),
    ('vision', 'Vision'),
    ('penalties', 'penalties'),
    ('marking', 'Marking'),
    ('standing_tackle', 'Standing_Tackle'),
    ('sliding_tackle', 'Sliding_Tackle'),
    ('gk_diving', 'GK_Diving'),
    ('gk_handling', 'GK_Handling'),
    ('gk_kicking', 'GK_Kicking'),
    ('gk_positioning', 'GK_Positioning'),
    ('gk_reflexes', 'GK_Reflexes'),
    ('height', 'Height'),
    ('weight', 'Weight')
]

missing_epl_players = spark.createDataFrame([
    Row(id=138985, player_api_id=-1, player_name='Papy Djilobodji', player_fifa_api_id=197937, birthday='1988-12-01 00:00:00', height=193, weight=181)
]).select( 'id', 'player_api_id', 'player_name', 'player_fifa_api_id','birthday', 'height', 'weight') #Select statement to enforce the order of the colum

def calculate_age(birthday):
    birthday = datetime.strptime(birthday, '%Y-%m-%d %H:%M:%S').date()
    age_in = datetime.strptime('2015-10-01', '%Y-%m-%d').date()
    return age_in.year - birthday.year - ((age_in.month, age_in.day) < (birthday.month, birthday.day))


def main():
    player_attr_data = spark.read.jdbc('jdbc:sqlite:input/database.sqlite', table='Player_Attributes')

    # Get name, birthday, weight and height
    player_data = spark.read.jdbc('jdbc:sqlite:input/database.sqlite', table='Player')
    player_data = player_data.union(missing_epl_players)
    get_age = functions.udf(calculate_age, types.IntegerType())
    player_data = player_data.withColumn('Age', get_age(player_data.birthday)).drop('birthday', 'player_api_id', 'id')

    # Get fifa 17 player data and associated fifa_api_id
    fifa_17_player_data = spark.read.csv('input/EPL_Players_Data.csv', header=True).withColumnRenamed('fifa_api_id', 'player_fifa_api_id').cache()
    player_api_id = fifa_17_player_data.select('player_fifa_api_id')

    # Make workrate column same as Fifa 17 data and delete all unnecessary column
    player_attr_data = player_attr_data.withColumn('Work_Rate', player_attr_data.attacking_work_rate + ' / ' + player_attr_data.defensive_work_rate)
    player_attr_data = player_attr_data.drop('player_api_id', 'potential', 'preferred_foot', 'attacking_work_rate', 'defensive_work_rate')

    # Get appropriate version of player
    player_attr_data = player_attr_data.withColumn('date_only', functions.to_date(player_attr_data.date))
    player_attr_data = player_attr_data.filter(player_attr_data.date_only >= "2015-08-01").filter(player_attr_data.date_only <= "2016-02-01")
    player_data_min_date = player_attr_data.groupBy('player_fifa_api_id').max('id').withColumnRenamed('max(id)', 'id').drop('player_fifa_api_id')
    player_attr_data = player_data_min_date.join(player_attr_data, 'id')
    player_attr_data = player_attr_data.drop('date', 'date_only')


    #Join player data from with EPL players from Fifa 17
    print('Players to be imported from Fifa 17', player_api_id.count())
    player_attr_data = player_api_id.join(player_attr_data, 'player_fifa_api_id', 'left')
    print('Total imported player from Fifa 17', player_attr_data.count())

    #Getting player that are available in Fifa 17 but not in Fifa 16
    non_existent_player_data = player_attr_data.select('player_fifa_api_id').filter(player_attr_data.id.isNull())
    print('Total player attr available only in Fifa 17', non_existent_player_data.count())

    #Getting player data that are available in both Fifa 16 and Fifa 17
    player_attr_data = player_attr_data.filter(player_attr_data.id.isNotNull())
    print('Total player attr available in both Fifa 17 and 16', player_attr_data.count())

    #Adding name, weight and Height with player data
    player_attr_data = player_attr_data.join(player_data, 'player_fifa_api_id', 'left')
    print('Total player with found name, weight, height, age', player_attr_data.count())
    missing_players_in_name_weight_height = player_attr_data.filter(player_attr_data.player_name.isNull()).cache()
    print('Missing players from name data', missing_players_in_name_weight_height.count())
    if missing_players_in_name_weight_height.count() > 0:
        missing_players_in_name_weight_height.select('id', 'player_fifa_api_id').show()
        print('Above players are missing in name data')
    missing_players_in_name_weight_height.unpersist()
    return

    #TODO: Non_existent_player_data attributes from Fifa 17
    non_existent_player_data = fifa_17_player_data.join(non_existent_player_data, 'player_fifa_api_id')
    print(non_existent_player_data.count())
    non_existent_player_data.show()
    return

if __name__ == "__main__":
    main()
