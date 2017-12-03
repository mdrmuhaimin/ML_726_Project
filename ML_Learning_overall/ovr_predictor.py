from pyspark.sql import SparkSession, functions, types
from pyspark.ml.regression import (GBTRegressor,
                                   RandomForestRegressor)
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import TrainValidationSplit
from ml_tools import *
import os

spark = SparkSession.builder.appName('Fifa_Ovr_Predictor').getOrCreate()
sc = spark.sparkContext
assert sc.version >= '2.2'  # make sure we have Spark 2.2+

OUTPUT_DIR = 'output'
INPUT_DIR = 'inputs'
SAVE_COMPLETE_FEATURES_WITH_TARGET = True


def get_model_params(model_param):
    model, name, score = model_param
    return get_best_tvs_model_params(model)


def get_trainers(trainRatio, estimator_gridbuilders, feature_cols, label_col, metricName=None ):
    column_names = dict(featuresCol='feature_cols',
                        labelCol=label_col,
                        predictionCol="{}_pred".format(label_col))

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
        all_stages = [feature_assembler, est]
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
                numTrees=[60]
            )
        ),
        estimator_gridbuilder(
            RandomForestRegressor(),
            dict(
                maxDepth=[10],
                maxBins=[15],
                numTrees=[80]
            )
        ),
        estimator_gridbuilder(
            RandomForestRegressor(),
            dict(
                maxDepth=[10],
                maxBins=[15],
                numTrees=[100]
            )
        ),
        estimator_gridbuilder(
            RandomForestRegressor(),
            dict(
                maxDepth=[20],
                maxBins=[15],
                numTrees=[60]
            )
        ),
        estimator_gridbuilder(
            RandomForestRegressor(),
            dict(
                maxDepth=[20],
                maxBins=[15],
                numTrees=[80]
            )
        ),
        estimator_gridbuilder(
            RandomForestRegressor(),
            dict(
                maxDepth=[20],
                maxBins=[15],
                numTrees=[100]
            )
        ),
        estimator_gridbuilder(
            GBTRegressor(),
            dict(
                maxIter=[10]
            )
        )
        , estimator_gridbuilder(
            GBTRegressor(),
            dict(
                maxIter=[50]
            )
        )
        , estimator_gridbuilder(
            GBTRegressor(),
            dict(
                maxIter=[100]
            )
        ), estimator_gridbuilder(
            GBTRegressor(),
            dict(
                maxIter=[150]
            )
        ), estimator_gridbuilder(
            GBTRegressor(),
            dict(
                maxIter=[200]
            )
        )
    ]
    metricName = 'rmse'
    tvs_list = get_trainers(.6, estimator_gridbuilders, feature_cols, label_cols, metricName)
    ev = tvs_list[0].getEvaluator()
    scorescale = 1 if ev.isLargerBetter() else -1
    model_name_scores = []
    for tvs in tvs_list:
        model = tvs.fit(training_data)
        test_pred = model.transform(test_data)
        score = ev.evaluate(test_pred) * scorescale
        model_name_scores.append((model, get_estimator_name(tvs.getEstimator()), score))
    print(list(map(get_model_params, model_name_scores)))
    best_model, best_name, best_score = min(model_name_scores, key=lambda triplet: triplet[2])
    print(
        "Best model is %s with validation data %s score %f" % (best_name, ev.getMetricName(), best_score * scorescale))
    return best_model


def main():
    predicted_attr_data = spark.read.format('csv')\
        .option('header', 'true')\
        .option('mode', 'DROPMALFORMED') \
        .load('{}/prediction_players.csv'.format(INPUT_DIR)) \
        .drop('pac','sho','pas','dri','def','phy')
    player_ovr_data = spark.read.format('csv')\
        .option('header', 'true')\
        .option('mode', 'DROPMALFORMED') \
        .load('{}/f18_ovr.csv'.format(INPUT_DIR))

    convert_to_int = functions.udf(lambda value: int(value), types.IntegerType())
    roundup_int = functions.udf(lambda value: round(float(value)), types.IntegerType())



    for column in predicted_attr_data.columns:
        predicted_attr_data = predicted_attr_data.withColumn(column, roundup_int(predicted_attr_data[column]))

    for column in player_ovr_data.columns:
        player_ovr_data = player_ovr_data.withColumn(column, convert_to_int(player_ovr_data[column]))

    predicted_attr_data = player_ovr_data.select('id', 'age', 'overall').join(predicted_attr_data, 'id')

    # Selecting players from actual data who are also present in predicted players data
    # training_data = predicted_attr_data.select('id').join(player_ovr_data, 'id')
    training_data = player_ovr_data
    high_level_attributes = ['pac','sho','pas','dri','def','phy']

    for attr in high_level_attributes:
        predicted_attr_data = predicted_attr_data.withColumnRenamed('{}_pred'.format(attr), attr)

    training_data = training_data.select('id', 'overall','age', 'pac','sho','pas','dri','def','phy')
    predicted_attr_data = predicted_attr_data.select('id', 'overall','age', 'pac','sho','pas','dri','def','phy')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if SAVE_COMPLETE_FEATURES_WITH_TARGET == True:
        training_data.toPandas().to_csv(os.path.join(OUTPUT_DIR,'training_feature_target_set.csv'), sep=',', encoding='utf-8')
        predicted_attr_data.toPandas().to_csv(os.path.join(OUTPUT_DIR,'feature_to_be_predicted.csv'), sep=',', encoding='utf-8')

    (training_data, test_data) = training_data.randomSplit([0.8, 0.2])
    # test_data = predicted_attr_data
    model = get_best_model(training_data, test_data)
    print("Best parameters on test data:\n", get_best_tvs_model_params(model))
    data_pred = model.transform(predicted_attr_data)
    data_pred = data_pred.withColumn('overall_pred', roundup_int(data_pred['overall_pred']))
    data_pred = data_pred.drop('feature_cols')
    print('Predicted overall')
    data_pred.toPandas().to_csv(os.path.join(OUTPUT_DIR, 'predicted_ovr.csv'), sep=',', encoding='utf-8')
    figurename = 'pred_vs_label'
    create_graph(data_pred, 'overall', 'overall_pred', fraction=5.e5 / data_pred.count(), base_name=os.path.join(OUTPUT_DIR, figurename))
    print('Graph created for attribute ovr')

    return

if __name__ == '__main__':
    main()
