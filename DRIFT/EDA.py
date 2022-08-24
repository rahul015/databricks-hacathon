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

delta_path='dbfs:/FileStore/Vanquishers/'
df=transformation(delta_path)

# COMMAND ----------



# COMMAND ----------

from sklearn import tree
plt.figure(figsize=(20,20))_ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)
