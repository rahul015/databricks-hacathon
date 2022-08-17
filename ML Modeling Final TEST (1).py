# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

file_path = f"dbfs:/FileStore/CAR_DETAILS_FROM_CAR_DEKHO-1.csv"
car_df = (spark.read.option('header','true').option('inferschema','true').format("csv").load(file_path))
train_df, test_df = car_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

car_df.write.format('delta').save("dbfs:/FileStore/Vanquishers/")

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



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC OWN ANALYSIS ON THE DATA

# COMMAND ----------

from databricks.feature_store import feature_table, FeatureLookup

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

file_path="dbfs:/FileStore/Vanquishers/"

# COMMAND ----------

df1=spark.read.format('delta').load(file_path).coalesce(1).withColumn("ind", monotonically_increasing_id())

# COMMAND ----------

df1=df1.withColumn("index",df1.ind.cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Engineering

# COMMAND ----------

df1.display()

# COMMAND ----------

from pyspark.sql.functions import col
def transform_func(df):
  df=df.withColumn('no_of_year',2020-col('year'))
  df=df.drop('year')
  df=df.drop('name')
  df=df.drop('ind')
  return df

# COMMAND ----------

transform_df=transform_func(df1)

# COMMAND ----------

df=transform_df.toPandas()

# COMMAND ----------

import pandas as pd 
df=pd.get_dummies(df)

# COMMAND ----------

df.drop(['fuel_Electric','seller_type_Dealer','transmission_Automatic','owner_Fourth & Above Owner'],axis=1,inplace=True)

# COMMAND ----------

df.head()

# COMMAND ----------

df.rename(columns = {'seller_type_Trustmark Dealer':'seller_type_Trustmark_Dealer','owner_First Owner':'owner_First_Owner','owner_Second Owner':'owner_Second_Owner','owner_Test Drive Car':'owner_Test_Drive_Car','owner_Third Owner':'owner_Third_Owner'}, inplace = True)

# COMMAND ----------

spark_df=spark.createDataFrame(df)

# COMMAND ----------

clean_username='Vanquishers'

# COMMAND ----------

import uuid

spark.sql(f"CREATE DATABASE IF NOT EXISTS {clean_username}")
table_name = f"{clean_username}.car_dataset"

print(table_name)

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

help(fs.create_table)

# COMMAND ----------

for i in spark_df.columns:
  spark_df=spark_df.withColumn(i+'T',spark_df[i].cast(IntegerType())).drop(i).withColumnRenamed(i+'T', i)

# COMMAND ----------

## select numeric features and aggregate the review scores, exclude target column "price"
numeric_cols = [x.name for x in spark_df.schema.fields if (x.dataType ==IntegerType()) and (x.name != "selling_price") and (x.name != "index")]
numeric_cols

# COMMAND ----------

## select numeric features and aggregate the review scores, exclude target column "price"
numeric_cols = [x.name for x in spark_df.schema.fields if (x.dataType ==IntegerType()) and (x.name != "selling_price") and (x.name != "index")]
@feature_table
def select_numeric_features(data):
    return data.select(['index']+numeric_cols)

numeric_features_df = select_numeric_features(spark_df)
display(numeric_features_df)

# COMMAND ----------

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

# fs.create_table(
#     name=table_name,
#     primary_keys=["index"],
#     df=numeric_features_df,
#     schema=numeric_features_df.schema,
#     mode='overwrite',
#     description="Numeric features of car data"
# )

# COMMAND ----------

print(f"Feature table description : {fs.get_table(table_name).description}")
print(f"Feature table data source : {fs.get_table(table_name).path_data_sources}")

# COMMAND ----------

# Display most recent table
feature_df = fs.read_table(name=table_name)
display(feature_df)

# COMMAND ----------

## inference data -- index (key), price (target) and a online feature (make up a fictional column - diff of review score in a month) 
inference_data_df = transform_df.select('index',"selling_price")
display(inference_data_df)

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="selling_price", exclude_columns="index")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("selling_price", axis=1)
    y = training_pd["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------



# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
 #Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# COMMAND ----------

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

# COMMAND ----------

rf_random.fit(X_train,y_train)

# COMMAND ----------

print(rf_random.best_params_)
print(rf_random.best_score_)

# COMMAND ----------

# cleaned_username='Vanquishers'
cleaned_username='test'

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

try:
    client.delete_registered_model(f"feature_store_car_{cleaned_username}") # Deleting model if already created
except:
    None

# COMMAND ----------

rf_random.best_params_

# COMMAND ----------

# import mlflow
# import mlflow.sklearn
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from mlflow.models.signature import infer_signature

# rf = RandomForestRegressor(n_estimators=rf_random.best_params_['n_estimators'], min_samples_split=rf_random.best_params_['min_samples_split'], min_samples_leaf=rf_random.best_params_['min_samples_leaf'], max_features=rf_random.best_params_['min_samples_leaf'], max_depth=rf_random.best_params_['max_depth'])
# rf.fit(X_train, y_train)

# input_example = X_train.head(3)
# signature = infer_signature(X_train, pd.DataFrame(y_train))

# with mlflow.start_run(run_name="feature_store_car_test") as run:
#     mlflow.sklearn.log_model(rf, "model", input_example=input_example, signature=signature)
#     mlflow.log_metric("mse", mean_squared_error(y_test, rf.predict(X_test)))
#     mlflow.log_param("n_estimators", n_estimators)
#     mlflow.log_param("max_depth", max_depth)
#     run_id = run.info.run_id

# COMMAND ----------

feature_lookups = [FeatureLookup(table_name = table_name, feature_names = None, lookup_key = "index") ]

## fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
training_set = fs.create_training_set(inference_data_df, feature_lookups, label="selling_price", exclude_columns="index")

# COMMAND ----------

# Disable model autologging and instead log explicitly via the FeatureStore
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name='feature_store_car_test') as run:

    rf = RandomForestRegressor(n_estimators=rf_random.best_params_['n_estimators'], min_samples_split=rf_random.best_params_['min_samples_split'], min_samples_leaf=rf_random.best_params_['min_samples_leaf'], max_features=rf_random.best_params_['min_samples_leaf'], max_depth=rf_random.best_params_['max_depth'])
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

    fs.log_model(
        model=rf,
        artifact_path="feature-store-test",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=f"feature_store_car_{cleaned_username}",
        input_example=X_train[:5],
        signature=infer_signature(X_train, y_train)
    )

# COMMAND ----------

run.info.run_id

# COMMAND ----------

model_name = f"feature_store_car_test"
model_name

# COMMAND ----------

model_uri = f"runs:/{run.info.run_id}/model"


# COMMAND ----------

# model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

client.transition_model_version_stage(
    name=model_version_details.name,
    version=model_version_details.version,
    stage="Staging"
)

# COMMAND ----------


# client.transition_model_version_stage(
#     name=model_details.name,
#     version=model_details.version,
#     stage="Production"
# )

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_version_details.name,
  version=model_version_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = inference_data_df.drop("selling_price") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_car_{cleaned_username}/1", 
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

## select numeric features and aggregate the review scores
review_columns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
                 "review_scores_communication", "review_scores_location", "review_scores_value"]
@feature_table
def select_numeric_features(data):
    result = (data.select(["index"] + numeric_cols)
              .withColumn("average_review_score", expr("+".join(review_columns)) / lit(len(review_columns)))
              .drop(*review_columns)
             )
    return result

condensed_review_df = select_numeric_features(airbnb_df)
display(condensed_review_df)

# COMMAND ----------

fs.write_table(
    name=table_name,
    df=condensed_review_df,
    mode="overwrite"
)
