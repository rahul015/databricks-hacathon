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

# MAGIC %run ./Transformation

# COMMAND ----------

run_name='Vanquisher_Drift_Model_Runs'
delta_path='dbfs:/FileStore/Vanquishers/'
project_home_dir = f"/FileStore/Vanquishers/"
project_local_tmp_dir = "/dbfs" + project_home_dir + "tmp/"

model_params = {'n_estimators': 1100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 15}

misc_params = {"target_col": "selling_price",
               "num_cols": ['km_driven','age_of_car','fuel_Diesel', 'fuel_Petrol', 'fuel_CNG', 'fuel_LPG', 'fuel_Electric', 'seller_type_Individual', 'seller_type_Dealer', 'seller_type_Trustmark_Dealer', 'transmission_Manual', 'transmission_Automatic','Owner']}



# COMMAND ----------

model_1_run = train_sklearn_rf_model(run_name, delta_path, model_params, misc_params, seed=42)

# COMMAND ----------

# Register model to MLflow Model Registry
registry_model_name = 'Vanquishers_CAR_DATASET_RF_MODEL'

model_1_run_id = model_1_run.info.run_id
model_1_version = mlflow.register_model(model_uri=f"runs:/{model_1_run_id}/model", name=registry_model_name)

# COMMAND ----------

# Transition model to PROD
model_1_version = transition_model(model_1_version, stage="Production")
print(model_1_version)

# COMMAND ----------


