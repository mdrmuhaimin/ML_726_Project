from pyspark.sql import SparkSession, functions, types
from pyspark.ml.regression import (GBTRegressor,
                                   RandomForestRegressor)
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import TrainValidationSplit
from ml_tools import *
import os

spark = SparkSession.builder.appName('Fifa_17_Processor').getOrCreate()
sc = spark.sparkContext
assert sc.version >= '2.2'  # make sure we have Spark 2.2+

SAVE_COMPLETE_FEATURES_WITH_TARGET = True
OUTPUT_DIR = 'output'
INPUT_DIR = 'inputs'


def get_trainers(trainRatio, estimator_gridbuilders, feature_cols, label_col, metricName=None ):
    column_names = dict(featuresCol='feature_cols',
                        labelCol=label_col,
                        predictionCol="{}_pred".format(label_col))
    discreete_columns = ['f16_Att_WR', 'f17_Att_WR', 'f16_Def_WR', 'f17_Def_WR']
    indexed_discreete_columns = ['f16_Att_WR_indexed', 'f17_Att_WR_indexed', 'f16_Def_WR_indexed', 'f17_Def_WR_indexed']
    hot_encoded_discreete_columns = ['f16_Att_WR_he', 'f17_Att_WR_he', 'f16_Def_WR_he', 'f17_Def_WR_he']
    feature_cols = list(set(feature_cols) - set(discreete_columns))
    feature_cols = feature_cols + hot_encoded_discreete_columns
    indexers = [StringIndexer(inputCol=column, outputCol='{}_indexed'.format(column)) for column in discreete_columns]
    hot_encoders = [OneHotEncoder(inputCol='{}_indexed'.format(column), outputCol='{}_he'.format(column)) for column in discreete_columns]
    feature_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=column_names["featuresCol"])

    ev = (RegressionEvaluator()
          .setLabelCol(column_names["labelCol"])
          .setPredictionCol(column_names["predictionCol"])
          )
    if metricName:
        ev = ev.setMetricName(metricName)
    tvs_list = []
    for est, pgb in estimator_gridbuilders:
        est = est.setParams(**column_names)
        all_stages = indexers + hot_encoders
        all_stages = all_stages + [feature_assembler, est]
        pl = Pipeline(stages=all_stages)

        paramGrid = pgb.build()
        tvs_list.append(TrainValidationSplit(estimator=pl,
                                             estimatorParamMaps=paramGrid,
                                             evaluator=ev,
                                             trainRatio=trainRatio))
    return tvs_list


def get_best_model(training_data, test_data):

    feature_cols = training_data.columns[2:]
    label_cols = training_data.columns[1]

    estimator_gridbuilders = [
        estimator_gridbuilder(
            RandomForestRegressor(),
            dict(
                maxDepth=[10],
                maxBins=[15],
                numTrees=[80]
            )
        )
        # ,estimator_gridbuilder(
        #     GBTRegressor(maxIter=100),
        #     dict()
        # )
    ]
    metricName = 'r2'
    tvs_list = get_trainers(.2, estimator_gridbuilders, feature_cols, label_cols, metricName)
    ev = tvs_list[0].getEvaluator()
    scorescale = 1 if ev.isLargerBetter() else -1
    model_name_scores = []
    for tvs in tvs_list:
        model = tvs.fit(training_data)
        test_pred = model.transform(test_data)
        score = ev.evaluate(test_pred) * scorescale
        model_name_scores.append((model, get_estimator_name(tvs.getEstimator()), score))
    best_model, best_name, best_score = max(model_name_scores, key=lambda triplet: triplet[2])
    print(
        "Best model is %s with validation data %s score %f" % (best_name, ev.getMetricName(), best_score * scorescale))
    return best_model

# def conver_to_int_func (value):
#     print(value)
#     return int(value)

def validate_work_rate ( wr ):
    wr = wr.lower().strip()
    if wr != 'high' and wr != 'medium' and wr != 'low':
        wr = 'medium'
    return wr


def main():
    f16_player_data = spark.read.format('csv')\
        .option('header', 'true')\
        .option('mode', 'DROPMALFORMED') \
        .load('{}/f16_players.csv'.format(INPUT_DIR))
    f17_player_data = spark.read.format('csv')\
        .option('header', 'true')\
        .option('mode', 'DROPMALFORMED') \
        .load('{}/f17_players.csv'.format(INPUT_DIR))
    f18_player_data = spark.read.format('csv')\
        .option('header', 'true')\
        .option('mode', 'DROPMALFORMED') \
        .load('{}/f18_players.csv'.format(INPUT_DIR))

    get_att_wr = functions.udf(lambda wr: validate_work_rate(wr.split('/')[0]), types.StringType())
    get_def_wr = functions.udf(lambda wr: validate_work_rate(wr.split('/')[1]), types.StringType())
    get_height_as_int = functions.udf(lambda height: int(height.split(' ')[0]) if len(height.split(' ')) > 1 else int(height), types.IntegerType())
    get_weight_as_lb = functions.udf(lambda weight: round(int(weight.split(' ')[0]) * 2.20462) if len(weight.split(' ')) > 1 else int(weight), types.IntegerType())
    convert_to_int = functions.udf(lambda value: int(value), types.IntegerType())
    # convert_to_int = functions.udf(conver_to_int_func, types.IntegerType())


    f17_player_data = f17_player_data.withColumn('Height', get_height_as_int(f17_player_data.Height))
    f17_player_data = f17_player_data.withColumn('Weight', get_weight_as_lb(f17_player_data.Weight))
    f16_player_data = f16_player_data.withColumn('Height', get_height_as_int(f16_player_data.Height))
    f16_player_data = f16_player_data.withColumn('Weight', get_weight_as_lb(f16_player_data.Weight))

    for column in f16_player_data.columns:
        if column == 'Work_Rate':
            continue
        f16_player_data = f16_player_data.withColumn(column, convert_to_int(f16_player_data[column]))

    for column in f17_player_data.columns:
        if column == 'Work_Rate':
            continue
        f17_player_data = f17_player_data.withColumn(column, convert_to_int(f17_player_data[column]))

    for column in f18_player_data.columns:
        f18_player_data = f18_player_data.withColumn(column, convert_to_int(f18_player_data[column]))

    # f16_player_data.toPandas().to_csv('test_f16_players.csv', sep=',', encoding='utf-8')
    # f17_player_data.toPandas().to_csv('test_f17_players.csv', sep=',', encoding='utf-8')

    f16_player_data = f16_player_data.withColumn('Att_WR', get_att_wr(f16_player_data.Work_Rate))
    f16_player_data = f16_player_data.withColumn('Def_WR', get_def_wr(f16_player_data.Work_Rate))
    f16_player_data = f16_player_data.drop('Work_Rate')

    f17_player_data = f17_player_data.withColumn('Att_WR', get_att_wr(f17_player_data.Work_Rate))
    f17_player_data = f17_player_data.withColumn('Def_WR', get_def_wr(f17_player_data.Work_Rate))
    f17_player_data = f17_player_data.drop('Work_Rate')

    f17_player_data = f17_player_data.withColumnRenamed('fifa_api_id', 'player_fifa_api_id')

    f16_column_names = f16_player_data.columns
    f17_column_names = f17_player_data.columns

    for column in f16_column_names:
        if column == 'player_fifa_api_id':
            continue
        f16_player_data = f16_player_data.withColumnRenamed(column, 'f16_{}'.format(column))
    for column in f17_column_names:
        if column == 'player_fifa_api_id':
            continue
        f17_player_data = f17_player_data.withColumnRenamed(column, 'f17_{}'.format(column))

    training_attributes = ['pac','sho','pas','dri','def','phy']
    training_test_data = f16_player_data.join(f17_player_data, 'player_fifa_api_id').withColumnRenamed('player_fifa_api_id', 'id')
    (training_data, test_data) = training_test_data.randomSplit([0.8, 0.2])

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if SAVE_COMPLETE_FEATURES_WITH_TARGET == True:
        training_test_data.join(f18_player_data, 'id').toPandas().to_csv(os.path.join(OUTPUT_DIR,'complete_target_feature_set.csv'), sep=',', encoding='utf-8')
    df_attributes = []
    for attribute in training_attributes:
        training_data_with_feature = f18_player_data.select('id', attribute).join(training_data, 'id')
        test_data_with_feature = f18_player_data.select('id', attribute).join(test_data, 'id')
        training_test_data_with_feature = f18_player_data.select('id', attribute).join(training_test_data, 'id')
        model = get_best_model(training_data_with_feature, test_data_with_feature)
        print("Best parameters on test data:\n", get_best_tvs_model_params(model))
        data_pred = model.transform(training_test_data_with_feature)
        data_pred = data_pred.drop('feature_cols')
        print('Predicted for attribute {}'.format(attribute))
        df_attributes.append(data_pred.select('id', '{}_pred'.format(attribute)))
        data_pred.toPandas().to_csv(os.path.join(OUTPUT_DIR, 'prediction_players_{}.csv'.format(attribute)), sep=',', encoding='utf-8')
        figurename = '{}_pred_vs_label'.format(attribute)
        create_graph(data_pred, attribute, '{}_pred'.format(attribute), fraction=5.e5 / data_pred.count(), base_name=os.path.join(OUTPUT_DIR, figurename))
        print('Graph created for attribute {}'.format(attribute))

    final_df = None
    for df in df_attributes:
        if final_df == None:
            final_df = df
        else:
            final_df = final_df.join(df, 'id')
    final_df.toPandas().to_csv(os.path.join(OUTPUT_DIR, 'prediction_players.csv'), sep=',', encoding='utf-8')
    return

if __name__ == '__main__':
    main()
