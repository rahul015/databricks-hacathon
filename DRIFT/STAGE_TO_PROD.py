# Databricks notebook source
from delta.tables import DeltaTable
import tempfile
import os
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

# MAGIC %run ./Monitoring_UDF

# COMMAND ----------

# MAGIC %run ./Training_UDF

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("stats_threshold_limit", "0.5")
dbutils.widgets.text("p_threshold", "0.05")
dbutils.widgets.text("min_model_r2_threshold", "0.1")

stats_threshold_limit = float(dbutils.widgets.get("stats_threshold_limit"))       # how much we should allow basic summary stats to shift 
p_threshold = float(dbutils.widgets.get("p_threshold"))                           # the p-value below which to reject null hypothesis 
min_model_r2_threshold = float(dbutils.widgets.get("min_model_r2_threshold"))     # minimum model improvement

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Staging
registry_model_name='Vanquishers_CAR_DATASET_RF_MODEL'
current_staging_run_1 = get_run_from_registered_model(registry_model_name, stage="Staging")
current_prod_run_1 = get_run_from_registered_model(registry_model_name, stage="Production")

metric_to_check = "test_r2"
staging_r2,prod_r2=compare_model_perfs_new(current_staging_run_1, current_prod_run_1, min_model_r2_threshold, metric_to_check)

# COMMAND ----------

client=MlflowClient()

# COMMAND ----------

if staging_r2>prod_r2:
    client.transition_model_version_stage(
    name='Vanquishers_CAR_DATASET_RF_MODEL',
    version=fetch_model_version('Vanquishers_CAR_DATASET_RF_MODEL',stage='Staging').version,
    stage="None")

# COMMAND ----------


