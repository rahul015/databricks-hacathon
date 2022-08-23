# Databricks notebook source
# MAGIC %run ./Transformation

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, lit, expr, rand
import uuid
from databricks import feature_store
from pyspark.sql.types import StringType, DoubleType,IntegerType
from databricks.feature_store import feature_table, FeatureLookup
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

def feature_store(delta_path):
    df=transformation(delta_path)
    spark_df=spark.createDataFrame(df)
    for i in spark_df.columns:
      spark_df=spark_df.withColumn(i+'T',spark_df[i].cast(IntegerType())).drop(i).withColumnRenamed(i+'T', i)
    numeric_cols = [x.name for x in spark_df.schema.fields if (x.dataType ==IntegerType()) and (x.name != "index")]
    numeric_cols = [x.name for x in spark_df.schema.fields if (x.dataType ==IntegerType()) and (x.name != "index")]
    @feature_table
    def select_numeric_features(data):
        return data.select(['index']+numeric_cols)

    numeric_features_df = select_numeric_features(spark_df)
    display(numeric_features_df)
    clean_username='Vanquishers'
    import uuid

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {clean_username}")
    table_name = f"{clean_username}.car_dataset"
    print(table_name)
    from databricks import feature_store

    fs = feature_store.FeatureStoreClient()
    fs.create_table(
        name=table_name,
        primary_keys=["index"],
        schema=numeric_features_df.schema,
        description="Numeric features of car data"
    )

    fs.write_table(
        name=table_name,
        df=numeric_features_df,
        mode="overwrite"
    )

# COMMAND ----------


