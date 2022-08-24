# Databricks notebook source
# MAGIC %run ./Transformation

# COMMAND ----------

# MAGIC %run ./Training_UDF

# COMMAND ----------

# MAGIC %run ./Monitoring_UDF

# COMMAND ----------

from mlflow.tracking.client import MlflowClient


# COMMAND ----------

delta_path='dbfs:/FileStore/Vanquishers/'
features=transformation(delta_path)
features.drop(['selling_price','index'],axis=1,inplace=True)
current_prod_run_1 = get_run_from_registered_model('Vanquishers_CAR_DATASET_RF_MODEL', stage="Production")
predict = mlflow.pyfunc.spark_udf(spark, f"runs:/{current_prod_run_1.info.run_id}/model")
spark_df=spark.createDataFrame(features)
for i in spark_df.columns:
  spark_df=spark_df.withColumn(i+'T',spark_df[i].cast(IntegerType())).drop(i).withColumnRenamed(i+'T', i)


# COMMAND ----------

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))

display(prediction_df)


# COMMAND ----------

# prediction_df.write.mode('overwrite').saveAsTable('Predictions')

# COMMAND ----------

delta_partitioned_path = f"dbfs:/FileStore/Vanquishers_BATCH/batch-predictions-partitioned.delta"

prediction_df.write.partitionBy("Owner").mode("OVERWRITE").format("delta").save(delta_partitioned_path)
spark.sql(f"OPTIMIZE delta.`{delta_partitioned_path}` ZORDER BY (age_of_car)")

# COMMAND ----------


