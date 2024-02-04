from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, collect_list
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from annoy import AnnoyIndex
import time

def main(spark):
    # Reading the data
    train = spark.read.parquet('hdfs:/user/ny675_nyu_edu/train_data_processed.parquet')
    validation = spark.read.parquet('hdfs:/user/ny675_nyu_edu/val_data_processed.parquet')
    test = spark.read.parquet('hdfs:/user/ny675_nyu_edu/test_data_processed.parquet')

    # Casting data types
    train = train.withColumn("count", col("count").cast("integer"))
    train = train.withColumn("recording_id", col("recording_id").cast("integer"))
    validation = validation.withColumn("count", col("count").cast("integer"))
    validation = validation.withColumn("recording_id", col("recording_id").cast("integer"))
    test = test.withColumn("recording_id", col("recording_id").cast("integer"))

    start1 = time.time()

    # Train ALS model
    als = ALS(rank=50, maxIter=10, regParam=0.001, userCol="user_id", itemCol="recording_id", ratingCol="count", coldStartStrategy="drop", nonnegative=True)
    als_model = als.fit(train)

    print("fitting time:", time.time()-start1)

    # Extract item factors
    item_factors = als_model.itemFactors

    # Define Annoy index
    annoy_index = AnnoyIndex(len(item_factors.first()[1]), 'angular')

    # Populate the index with the item vectors from the ALS model
    for item in item_factors.collect():
        item_id = item[0]
        item_vector = item[1]
        annoy_index.add_item(item_id, item_vector)

    # Build the index
    annoy_index.build(10) # 10 trees, but you can adjust this parameter as needed.

    # Save the index to disk
    annoy_index.save('annoy_index.ann')

    print("Annoy index is built and saved to 'annoy_index.ann'")

    start2 = time.time()

    # Generate recommendations using Annoy
    test_users = test.select("user_id").distinct().withColumnRenamed("user_id", "id")
    test_user_factors = als_model.userFactors.join(test_users, on="id").collect()

    # Generate recommendations using Annoy
    recommendations_list = []

    # Loop over user factors
    for user_factor in test_user_factors:
        user_vector = user_factor[1]
        recommendations = annoy_index.get_nns_by_vector(user_vector, 100)
        recommendations_list.append(Row(user_id=user_factor[0], predicted=recommendations))

    als_pre_test_annoy = spark.createDataFrame(recommendations_list)
    print("Test Prediction done")

    listened_test = test.groupBy('user_id').agg(collect_list('recording_id').alias('listened'))

    # Evaluate the recommendations
    pre_true_test_annoy = als_pre_test_annoy.join(listened_test, "user_id").rdd.map(lambda row: (row[1], row[2]))
    test_map_annoy = RankingMetrics(pre_true_test_annoy).meanAveragePrecisionAt(100)

    print("The MAP for test set using Annoy:", test_map_annoy)

    print('Time used with Annoy', time.time()-start2)

    start3 = time.time()

    # Generate recommendations using Annoy
    val_users = validation.select("user_id").distinct().withColumnRenamed("user_id", "id")
    val_user_factors = als_model.userFactors.join(val_users, on="id").collect()

    # Generate recommendations using Annoy
    recommendations_list3 = []

    # Loop over user factors
    for user_factor in val_user_factors:
        user_vector = user_factor[1]
        recommendations = annoy_index.get_nns_by_vector(user_vector, 100)
        recommendations_list3.append(Row(user_id=user_factor[0], predicted=recommendations))

    als_pre_val_annoy = spark.createDataFrame(recommendations_list3)
    print("Validation Prediction done")

    listened_test3 = validation.groupBy('user_id').agg(collect_list('recording_id').alias('listened'))

    # Evaluate the recommendations
    pre_true_val_annoy = als_pre_val_annoy.join(listened_test3, "user_id").rdd.map(lambda row: (row[1], row[2]))
    val_map_annoy = RankingMetrics(pre_true_val_annoy).meanAveragePrecisionAt(100)

    print("The MAP for validation set using Annoy:", val_map_annoy)

    print('Time used with Annoy', time.time()-start3)

    start4 = time.time()

    # Generate recommendations using Annoy
    train_users = train.select("user_id").distinct().withColumnRenamed("user_id", "id")
    train_user_factors = als_model.userFactors.join(train_users, on="id").collect()

    # Generate recommendations using Annoy
    recommendations_list4 = []

    # Loop over user factors
    for user_factor in train_user_factors:
        user_vector = user_factor[1]
        recommendations = annoy_index.get_nns_by_vector(user_vector, 100)
        recommendations_list4.append(Row(user_id=user_factor[0], predicted=recommendations))

    als_pre_train_annoy = spark.createDataFrame(recommendations_list4)
    print("Train Prediction done")

    listened_test4 = train.groupBy('user_id').agg(collect_list('recording_id').alias('listened'))

    # Evaluate the recommendations
    pre_true_train_annoy = als_pre_train_annoy.join(listened_test4, "user_id").rdd.map(lambda row: (row[1], row[2]))
    train_map_annoy = RankingMetrics(pre_true_train_annoy).meanAveragePrecisionAt(100)

    print("The MAP for train set using Annoy:", train_map_annoy)

    print('Time used with Annoy', time.time()-start4)

if __name__ == "__main__":
    # Create the spark
    # Create the spark session object
    spark = SparkSession.builder.appName('als_annoy').getOrCreate()

    # Call our main routine
    main(spark)