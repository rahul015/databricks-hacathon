# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# file_path = f"dbfs:/FileStore/CAR_DETAILS_FROM_CAR_DEKHO-1.csv"
# car_df = (spark.read.option('header','true').option('inferschema','true').format("csv").load(file_path))
# train_df, test_df = car_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# car_df.write.format('delta').save("dbfs:/FileStore/Vanquishers/")

# COMMAND ----------

file_path="dbfs:/FileStore/Vanquishers/"

car_df=spark.read.format('delta').load(file_path)
train_df, test_df = car_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

train_df.display()

# COMMAND ----------

train_df.dtypes

# COMMAND ----------


from databricks import automl

summary = automl.regress(train_df, target_col="selling_price", primary_metric="rmse")

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# MAGIC %md Now we can test the model that we got from AutoML against our test data. We'll be using <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf" target="_blank">mlflow.pyfunc.spark_udf</a> to register our model as a UDF and apply it in parallel to our test data.

# COMMAND ----------

# Load the best trial as an MLflow Model
import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("selling_price").columns))
display(pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="selling_price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE on test dataset: {rmse:.3f}")

# COMMAND ----------


