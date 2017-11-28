from pyspark.sql import SparkSession, functions, types
import os

spark = SparkSession.builder.appName('Fifa_17_Processor').getOrCreate()
sc = spark.sparkContext
assert sc.version >= '2.2'  # make sure we have Spark 2.2+
EPL_15_16_Teams = ['Chelsea', 'Arsenal', 'Manchester United', 'Manchester City',
                   'Tottenham Hotspur', 'Liverpool', 'Everton', 'Stoke City', 'West Ham United', 'Leicester City', 'Crystal Palace',
                   'Watford', 'Southampton', 'Swansea City', 'Sunderland', 'West Bromwich Albion', 'Burnley', 'Middlesbrough',
                   'Hull City', 'Bournemouth']

def get_player_id (url):
    return url.split('/')[2]


def main():
    player_data = spark.read.format("csv")\
        .option("header", "true")\
        .option("mode", "DROPMALFORMED")\
        .load('FullData.csv')
    player_data = player_data[player_data.Club.isin(EPL_15_16_Teams)]\
        .drop('Nationality', 'National_Position', 'National_Kit', 'Club_Position', 'Club_Kit', 'Club_Joining', 'Contract_Expiry',
              'Preffered_Foot', 'Birth_Date')
    player_name = spark.read.format("csv") \
        .option("header", "true") \
        .option("mode", "DROPMALFORMED") \
        .load('PlayerNames.csv')

    count_before_join = player_data.count()
    player_data = player_data.join(player_name, "Name")
    count_after_join = player_data.count()

    assert count_after_join == count_before_join
    #Find duplicates
    # player_data.groupBy("Name").count().sort('count',  ascending=False).show()

    parse_player_id = functions.udf(get_player_id, types.StringType())
    player_data = player_data.withColumn('fifa_api_id', parse_player_id(player_data.url))
    player_data = player_data.drop('url')
    output_id_dir = 'player_api_id'
    output_dir = 'processed_player_data'
    if not os.path.exists(output_id_dir):
        os.makedirs(output_id_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    player_data.select('fifa_api_id').write.csv(output_id_dir, mode='overwrite', header=True)
    player_data.write.csv(output_dir, mode='overwrite', header=True)
    return

if __name__ == "__main__":
    main()
