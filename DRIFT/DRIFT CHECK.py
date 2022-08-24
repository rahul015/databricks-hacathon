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

# MAGIC %run ./Training_UDF

# COMMAND ----------

# MAGIC %run ./Monitoring_UDF

# COMMAND ----------

month_1_error_delta_path = 'dbfs:/FileStore/Vanquishers_inbound/'

month_1_err_pdf = transformation(month_1_error_delta_path)

# Compute summary statistics on new incoming data
summary_stats_month_1_err_pdf = create_summary_stats_pdf(month_1_err_pdf)

# COMMAND ----------

current_prod_pdf_1_transform = transformation('dbfs:/FileStore/Vanquishers/')

# COMMAND ----------

registry_model_name = 'Vanquishers_CAR_DATASET_RF_MODEL'
run_name='Vanquisher_Drift_Model_Runs'
delta_path='dbfs:/FileStore/Vanquishers/'
project_home_dir = f"/FileStore/Vanquishers/"
project_local_tmp_dir = "/dbfs" + project_home_dir + "tmp/"

model_params = {'n_estimators': 1100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 15}

misc_params = {"target_col": "selling_price",
               "num_cols": ['km_driven','age_of_car','fuel_Diesel', 'fuel_Petrol', 'fuel_CNG', 'fuel_LPG', 'fuel_Electric', 'seller_type_Individual', 'seller_type_Dealer', 'seller_type_Trustmark_Dealer', 'transmission_Manual', 'transmission_Automatic','Owner']}



# COMMAND ----------

# Get the original MLflow run associated with the model registered under Production
current_prod_run_1 = get_run_from_registered_model(registry_model_name, stage="Production")

# Load in original versions of Delta table used at training time for current Production model
current_prod_pdf_1 = load_delta_table_from_run(current_prod_run_1).toPandas()

# Load summary statistics pandas DataFrame for data which the model currently in Production was trained and evaluated against
current_prod_stats_pdf_1 = load_summary_stats_pdf_from_run(current_prod_run_1, project_local_tmp_dir)

# COMMAND ----------

current_prod_stats_pdf_1.head()

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("stats_threshold_limit", "0.5")
dbutils.widgets.text("p_threshold", "0.05")
dbutils.widgets.text("min_model_r2_threshold", "0.1")

stats_threshold_limit = float(dbutils.widgets.get("stats_threshold_limit"))       # how much we should allow basic summary stats to shift 
p_threshold = float(dbutils.widgets.get("p_threshold"))                           # the p-value below which to reject null hypothesis 
min_model_r2_threshold = float(dbutils.widgets.get("min_model_r2_threshold"))     # minimum model improvement

# COMMAND ----------

print("CHECKING PROPORTION OF NULLS.....")
check_null_proportion(month_1_err_pdf, null_proportion_threshold=.5)

# COMMAND ----------

current_prod_stats_pdf_1.head()

# COMMAND ----------

statistic_list = ["mean", "median", "std", "min", "max"]

# Check if the new summary stats deviate from previous summary stats by a certain threshold
unique_feature_diff_array_month_1 = check_diff_in_summary_stats(summary_stats_month_1_err_pdf, 
                                                                current_prod_stats_pdf_1, 
                                                                misc_params['num_cols'] + [misc_params['target_col']], # Include the target col in this analysis
                                                                stats_threshold_limit, 
                                                                statistic_list)
unique_feature_diff_array_month_1

# COMMAND ----------

print(f"Let's look at the box plots of the features that exceed the stats_threshold_limit of {stats_threshold_limit}")
plot_boxplots(unique_feature_diff_array_month_1, current_prod_pdf_1_transform, month_1_err_pdf)

# COMMAND ----------

print("\nCHECKING VARIANCES WITH LEVENE TEST.....")
check_diff_in_variances(current_prod_pdf_1_transform, month_1_err_pdf, misc_params['num_cols'], p_threshold)

# COMMAND ----------

print("\nCHECKING KS TEST.....")
drift=check_dist_ks_bonferroni_test(current_prod_pdf_1_transform, month_1_err_pdf, misc_params['num_cols'], p_threshold)

# COMMAND ----------



# COMMAND ----------

if drift==0:
    dbutils.notebook.exit('stop')

# COMMAND ----------

mlflow.set_experiment('/Vanquishers/DRIFT/DRIFT CHECK')

# COMMAND ----------

model_1_run_drift = train_sklearn_rf_model(run_name, month_1_error_delta_path, model_params, misc_params, seed=42)

# COMMAND ----------

# Register model to MLflow Model Registry
registry_model_name = 'Vanquishers_CAR_DATASET_RF_MODEL'

model_1_run_id = model_1_run_drift.info.run_id
model_1_version = mlflow.register_model(model_uri=f"runs:/{model_1_run_id}/model", name=registry_model_name)

# Transition model to PROD
model_1_version = transition_model(model_1_version, stage="Staging")
print(model_1_version)

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Staging
current_staging_run_1 = get_run_from_registered_model(registry_model_name, stage="Staging")

metric_to_check = "test_r2"
compare_model_perfs(current_staging_run_1, current_prod_run_1, min_model_r2_threshold, metric_to_check)

# COMMAND ----------



# COMMAND ----------

# Get the original MLflow run associated with the model registered under Staging
registry_model_name='Vanquishers_CAR_DATASET_RF_MODEL'
current_staging_run_1 = get_run_from_registered_model(registry_model_name, stage="Staging")

metric_to_check = "test_r2"
staging_r2,prod_r2=compare_model_perfs_new(current_staging_run_1, current_prod_run_1, min_model_r2_threshold, metric_to_check)

# COMMAND ----------

staging_r2

# COMMAND ----------



# COMMAND ----------

# month_1_model_version = transition_model(model_1_version, stage="Production")
# print(month_1_model_version)
