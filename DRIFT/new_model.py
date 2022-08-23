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

# MAGIC %run ./Transformation

# COMMAND ----------

# DBTITLE 1,Training Functions
def get_delta_version(delta_path):
  """
  Function to get the most recent version of a Delta table give the path to the Delta table
  
  :param delta_path: (str) path to Delta table
  :return: Delta version (int)
  """
  # DeltaTable is the main class for programmatically interacting with Delta tables
  delta_table = DeltaTable.forPath(spark, delta_path)
  # Get the information of the latest commits on this table as a Spark DataFrame. 
  # The information is in reverse chronological order.
  delta_table_history = delta_table.history() 
  
  # Retrieve the lastest Delta version - this is the version loaded when reading from delta_path
  delta_version = delta_table_history.first()["version"]
  
  return delta_version

# COMMAND ----------

def create_summary_stats_pdf(pdf):
  """
  Create a pandas DataFrame of summary statistics for a provided pandas DataFrame.
  Involved calling .describe on pandas DataFrame provided and additionally add
  median values and a count of null values for each column.
  
  :param pdf: pandas DataFrame
  :return: pandas DataFrame of sumary statistics for each column
  """
  stats_pdf = pdf.describe(include="all")

  # Add median values row
  median_vals = pdf.median()
  stats_pdf.loc["median"] = median_vals

  # Add null values row
  null_count = pdf.isna().sum()
  stats_pdf.loc["null_count"] = null_count

  return stats_pdf

# COMMAND ----------

def log_summary_stats_pdf_as_csv(pdf):
  """
  Log summary statistics pandas DataFrame as a csv file to MLflow as an artifact
  """
  temp = tempfile.NamedTemporaryFile(prefix="summary_stats_", suffix=".csv")
  temp_name = temp.name
  try:
    pdf.to_csv(temp_name)
    mlflow.log_artifact(temp_name, "summary_stats.csv")
  finally:
    temp.close() # Delete the temp file

# COMMAND ----------

def create_sklearn_rf_pipeline(model_params, seed=42):
  """
  Create the sklearn pipeline required for the RandomForestRegressor.
  We compose two components of the pipeline separately - one for numeric cols, one for categorical cols
  These are then combined with the final RandomForestRegressor stage, which uses the model_params dict
  provided via the args. The unfitted pipeline is returned.
  
  For a robust pipeline in practice, one should also have a pipeline stage to add indicator columns for those features
  which have been imputed. This can be useful to encode information about those instances which have been imputed with
  a given value. We refrain from doing so here to simplify the pipeline, and focus on the overall workflow.
  
  :param model_params: (dict) Dictionary of model parameters to pass into sklearn RandomForestRegressor
  :param seed : (int) Random seed to set via random_state arg in RandomForestRegressor
 
  :return: sklearn pipeline
  """
  # Create pipeline component for numeric Features
#   numeric_transformer = Pipeline(steps=[
#       ("imputer", SimpleImputer(strategy='median'))])

#   # Create pipeline component for categorical Features
#   categorical_transformer = Pipeline(steps=[
#       ("imputer", SimpleImputer(strategy="most_frequent")),
#       ("ohe", OneHotEncoder(handle_unknown="ignore"))])

  # Combine numeric and categorical components into one preprocessor pipeline
  # Use ColumnTransformer to apply the different preprocessing pipelines to different subsets of features
  # Use selector (make_column_selector) to select which subset of features to apply pipeline to
#   preprocessor = ColumnTransformer(transformers=[
#       ("numeric", numeric_transformer, selector(dtype_exclude="category")),
#       ("categorical", categorical_transformer, selector(dtype_include="category"))
#   ])

  pipeline = Pipeline(steps=[("rf", RandomForestRegressor(random_state=seed, **model_params))])
  
  return pipeline

# COMMAND ----------

def load_delta_table_from_run(run):
  """
  Given an MLflow run, load the Delta table which was used for that run,
  using the path and version tracked at tracking time.
  Note that by default Delta tables only retain a commit history for 30 days, meaning
  that previous versions older than 30 days will be deleted by default. This property can
  be updated using the Delta table property delta.logRetentionDuration.
  For more information, see https://docs.databricks.com/delta/delta-batch.html#data-retention
  
  :param run: mlflow.entities.run.Run
  :return: Spark DataFrame
  """
  delta_path = run.data.params["delta_path"]
  delta_version = run.data.params["delta_version"]
  print(f"Loading Delta table from path: {delta_path}; version: {delta_version}")
  df = spark.read.format("delta").option("versionAsOf", delta_version).load(delta_path)
  
  return df  

# COMMAND ----------

def get_run_from_registered_model(registry_model_name, stage="Staging"):
    """
    Get Mlflow run object from registered model

    :param registry_model_name: (str) Name of MLflow Registry Model
    :param stage: (str) Stage for this model. One of "Staging" or "Production"

    :return: mlflow.entities.run.Run
    """
    model_version = fetch_model_version(registry_model_name, stage)
    run_id = model_version.run_id
    run = mlflow.get_run(run_id)

    return run

# COMMAND ----------

# Our dataset
# file_path='dbfs:/FileStore/Vanquishers/'
# df=transformation(file_path)

# COMMAND ----------



# COMMAND ----------

def train_sklearn_rf_model(run_name, delta_path, model_params, misc_params, seed=42):
  """
  Function to trigger training and evaluation of an sklearn RandomForestRegressor model.
  Parameters, metrics and artifacts are logged to MLflow during this process.
  Return the MLflow run object 
  
  :param run_name: (str) name to give to MLflow run
  :param delta_path: (str) path to Delta table to use as input data
  :param model_params: (dict) Dictionary of model parameters to pass into sklearn RandomForestRegressor
  :param misc_params: (dict) Dictionary of params to use 
  
  :return: mlflow.entities.run.Run  
  """  
  with mlflow.start_run(run_name=run_name) as run:

    # Enable MLflow autologging
    mlflow.autolog(log_input_examples=True, silent=True)
    
    # Load Delta table from delta_path
#     df = spark.read.format("delta").load(delta_path)   
    df = transformation(delta_path) 
    
    # Log Delta path and version
    mlflow.log_param("delta_path", delta_path)
    delta_version = get_delta_version(delta_path)
    mlflow.log_param("delta_version", delta_version)
    
    # Track misc parameters used in pipeline creation (preprocessing) as json artifact
    mlflow.log_dict(misc_params, "preprocessing_params.json")
    target_col = misc_params["target_col"]  
    num_cols = misc_params["num_cols"]    

    # Convert Spark DataFrame to pandas, as we will be training an sklearn model
    pdf = df.toPandas()     
    
    # Create summary statistics pandas DataFrame and log as a csv to MLflow
    summary_stats_pdf = create_summary_stats_pdf(pdf)
    log_summary_stats_pdf_as_csv(summary_stats_pdf)  
    
    # Track number of total instances and "month"
    num_instances = pdf.shape[0]
    mlflow.log_param("num_instances", num_instances)  # Log number of instances
    
    # Split data
    X = pdf.drop([misc_params["target_col"], "index"], axis=1)
    y = pdf[misc_params["target_col"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Track train/test data info as params
    num_training = X_train.shape[0]
    mlflow.log_param("num_training_instances", num_training)
    num_test = X_test.shape[0]
    mlflow.log_param("num_test_instances", num_test)

    # Fit sklearn pipeline with RandomForestRegressor model
    rf_pipeline = create_sklearn_rf_pipeline(model_params)
    rf_pipeline.fit(X_train, y_train)
    # Specify data schema which the model will use as its ModelSignature
    input_schema = Schema([
      ColSpec("integer", "km_driven"),
      ColSpec("integer", "no_of_year"),
      ColSpec("integer", "fuel_CNG"),
      ColSpec("integer", "fuel_Diesel"),
      ColSpec("integer", "fuel_LPG"),
      ColSpec("integer", "fuel_Petrol"),
      ColSpec("integer", "seller_type_Individual"),
      ColSpec("integer", "seller_type_Trustmark_Dealer"),
      ColSpec("integer", "transmission_Manual"),
      ColSpec("integer", "owner_First_Owner"),
      ColSpec("integer", "owner_Second_Owner"),
      ColSpec("integer", "owner_Test_Drive_Car"),
      ColSpec("integer", "owner_Third_Owner")
    ])
    output_schema = Schema([ColSpec("double")])
    signature = ModelSignature(input_schema, output_schema)
    mlflow.sklearn.log_model(rf_pipeline, "model", signature=signature)

    # Evaluate the model
    predictions = rf_pipeline.predict(X_test)
    test_mse = mean_squared_error(y_test, predictions) 
    r2 = r2_score(y_test, predictions)
    mlflow.log_metrics({"test_mse": test_mse,
                       "test_r2": r2})

  return run

# COMMAND ----------

def transition_model(model_version, stage):
    """
    Transition a model to a specified stage in MLflow Model Registry using the associated 
    mlflow.entities.model_registry.ModelVersion object.

    :param model_version: mlflow.entities.model_registry.ModelVersion. ModelVersion object to transition
    :param stage: (str) New desired stage for this model version. One of "Staging", "Production", "Archived" or "None"

    :return: A single mlflow.entities.model_registry.ModelVersion object
    """
    client = MlflowClient()
    
    model_version = client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )

    return model_version  
  

# COMMAND ----------

def fetch_model_version(registry_model_name, stage="Staging"):
    """
    For a given registered model, return the MLflow ModelVersion object
    This contains all metadata needed, such as params logged etc

    :param registry_model_name: (str) Name of MLflow Registry Model
    :param stage: (str) Stage for this model. One of "Staging" or "Production"

    :return: mlflow.entities.model_registry.ModelVersion
    """
    client = MlflowClient()
    filter_string = f'name="{registry_model_name}"'
    registered_model = client.search_registered_models(filter_string=filter_string)[0]

    if len(registered_model.latest_versions) == 1:
        model_version = registered_model.latest_versions[0]

    else:
        model_version = [model_version for model_version in registered_model.latest_versions if model_version.current_stage == stage][0]

    return model_version

# COMMAND ----------

def load_summary_stats_pdf_from_run(run, local_tmp_dir):
  """
  Given an MLflow run, download the summary stats csv artifact to a local_tmp_dir and load the
  csv into a pandas DataFrame
  
  :param run: mlflow.entities.run.Run
  :param local_tmp_dir: (str) path to a local filesystem tmp directory
  :return pandas DataFrame containing statistics computed during training
  """
  # Use MLflow clitent to download the csv file logged in the artifacts of a run to a local tmp path
  client = MlflowClient()
  if not os.path.exists(local_tmp_dir):
      os.mkdir(local_tmp_dir)
  local_path = client.download_artifacts(run.info.run_id, "summary_stats.csv", local_tmp_dir)
  print(f"Summary stats artifact downloaded in: {local_path}")
  
  # Load the csv into a pandas DataFrame
  summary_stats_path = local_path + "/" + os.listdir(local_path)[0]
  summary_stats_pdf = pd.read_csv(summary_stats_path, index_col="Unnamed: 0")
  
  return summary_stats_pdf 

# COMMAND ----------

# DBTITLE 1,Monitoring functions
def check_null_proportion(new_pdf, null_proportion_threshold):
  """
  Function to compute the proportions of nulls for all columns in Spark DataFrame and return any features that exceed the specified null threshold.
  
  :param df: (pd.DataFrame) The dataframe that contains new incoming data
  :param null_proportion_threshold: (float) A numeric value ranging from 0 and 1 that specifies the tolerable fraction of nulls. 
  """
  missing_stats = pd.DataFrame(new_pdf.isnull().sum() / len(new_pdf)).transpose()
  null_dict = {}
  null_col_list = missing_stats.columns[(missing_stats >= null_proportion_threshold).iloc[0]]
  for feature in null_col_list:
    null_dict[feature] = missing_stats[feature][0]
  try:
    assert len(null_dict) == 0
  except:
    print("Alert: There are feature(s) that exceed(s) the expected null threshold. Please ensure that the data is ingested correctly")
    print(null_dict)

# COMMAND ----------

def check_diff_in_summary_stats(new_stats_pdf, prod_stats_pdf, num_cols, stats_threshold_limit, statistic_list):
  """
  Function to check if the new summary stats significantly deviates from the summary stats in the production data by a certain threshold. 
  
  :param new_stats_pdf: (pd.DataFrame) summary statistics of incoming data
  :param prod_stats_pdf: (pd.DataFrame) summary statistics of production data
  :param num_cols: (list) a list of numeric columns
  :param stats_threshold_limit: (double) a float < 1 that signifies the threshold limit
  :param compare_stats_name: (string) can be one of mean, std, min, max
  :param feature_diff_list: (list) an empty list to store the feature names with differences
  """ 
  feature_diff_list = []
  for feature in num_cols: 
    print(f"\nCHECKING {feature}.........")
    for statistic in statistic_list: 
      val = prod_stats_pdf[[str(feature)]].loc[str(statistic)][0]
      upper_val_limit = val * (1 + stats_threshold_limit)
      lower_val_limit = val * (1 - stats_threshold_limit)

      new_metric_value = new_stats_pdf[[str(feature)]].loc[str(statistic)][0]

      if new_metric_value < lower_val_limit:
        feature_diff_list.append(str(feature))
        print(f"\tThe {statistic} {feature} in the new data is at least {stats_threshold_limit *100}% lower than the {statistic} in the production data. Decreased from {round(val, 2)} to {round(new_metric_value,2)}.")

      elif new_metric_value > upper_val_limit:
        feature_diff_list.append(str(feature))
        print(f"\tThe {statistic} {feature} in the new data is at least {stats_threshold_limit *100}% higher than the {statistic} in the production data. Increased from {round(val, 2)} to {round(new_metric_value, 2)}.")

      else:
        pass
  
  return np.unique(feature_diff_list)

# COMMAND ----------

def plot_boxplots(unique_feature_diff_array, reference_pdf, new_pdf):
  sns.set_theme(style="whitegrid")
  fig, ax = plt.subplots(len(unique_feature_diff_array), 2, figsize=(15,8))
  fig.suptitle("Distribution Comparisons between Incoming Data and Production Data")
  ax[0, 0].set_title("Production Data")
  ax[0, 1].set_title("Incoming Data")

  for i in range(len(unique_feature_diff_array)):
    p1 = sns.boxplot(ax=ax[i, 0], x=reference_pdf[unique_feature_diff_array[i]])
    p1.set_xlabel(str(unique_feature_diff_array[i]))
    p1.annotate(str(unique_feature_diff_array[i]), xy=(10,0.5))
    p2 = sns.boxplot(ax=ax[i, 1], x=new_pdf[unique_feature_diff_array[i]])
    p2.annotate(str(unique_feature_diff_array[i]), xy=(10,0.5))

# COMMAND ----------

def check_diff_in_variances(reference_df, new_df, num_cols, p_threshold):
  """
  This function uses the Levene test to check if each column's variance in new_df is significantly different from reference_df
  From docs: The Levene test tests the null hypothesis that all input samples are from populations with equal variances. Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.
  
  :param reference_df(pd.DataFrame): current dataframe in production
  :param new_df (pd.DataFrame): new dataframe
  :param num_cols (list): a list of numeric features
  
  ‘median’ : Recommended for skewed (non-normal) distributions.
  """
  var_dict = {}
  for feature in num_cols:
    levene_stat, levene_pval = stats.levene(reference_df[str(feature)], new_df[str(feature)], center="median")
    if levene_pval <= p_threshold:
      var_dict[str(feature)] = levene_pval
  try:
    assert len(var_dict) == 0
    print(f"No features have significantly different variances compared to production data at p-value {p_threshold}")
  except:
    print(f"The feature(s) below have significantly different variances compared to production data at p-value {p_threshold}")
    print(var_dict)

# COMMAND ----------

def check_dist_ks_bonferroni_test(reference_df, new_df, num_cols, p_threshold, ks_alternative="two-sided"):
    """
    Function to take two pandas DataFrames and compute the Kolmogorov-Smirnov statistic on 2 sample distributions
    where the variable in question is continuous.
    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous
    distribution. If the KS statistic is small or the p-value is high, then we cannot reject the hypothesis that 
    the distributions of the two samples are the same.
    The alternative hypothesis can be either ‘two-sided’ (default), ‘less’ or ‘greater’.
    This function assumes that the distributions to compare have the same column name in both DataFrames.
    
    see more details here: https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test

    :param reference_df: pandas DataFrame containing column with the distribution to be compared
    :param new_df: pandas DataFrame containing column with the distribution to be compared
    :param col_name: (str) Name of colummn to use as variable to create numpy array for comparison
    :param ks_alternative: Defines the alternative hypothesis - ‘two-sided’ (default), ‘less’ or ‘greater’.
    """
    ks_dict = {}
    ### Bonferroni correction 
    corrected_alpha = p_threshold / len(num_cols)
    print(f"The Bonferroni-corrected alpha level is {round(corrected_alpha, 4)}. Any features with KS statistic below this alpha level have shifted significantly.")
    for feature in num_cols:
      ks_stat, ks_pval = stats.ks_2samp(reference_df[feature], new_df[feature], alternative=ks_alternative, mode="asymp")
      if ks_pval <= corrected_alpha:
        ks_dict[feature] = ks_pval
    try:
      assert len(ks_dict) == 0
      print(f"No feature distributions has shifted according to the KS test at the Bonferroni-corrected alpha level of {round(corrected_alpha, 4)}. ")
    except:
      print(f"The feature(s) below have significantly different distributions compared to production data at Bonferroni-corrected alpha level of {round(corrected_alpha, 4)}, according to the KS test")
      print("\t", ks_dict)

# COMMAND ----------

def compare_model_perfs(current_staging_run, current_prod_run, min_model_perf_threshold, metric_to_check):
  """
  This model compares the performances of the models in staging and in production. 
  Outputs: Recommendation to transition the staging model to production or not
  
  :param current_staging_run: MLflow run that contains information on the staging model
  :param current_prod_run: MLflow run that contains information on the production model
  :param min_model_perf_threshold (float): The minimum threshold that the staging model should exceed before being transitioned to production
  :param metric_to_check (string): The metric that the user is interested in using to compare model performances
  """
  model_diff_fraction = current_staging_run.data.metrics[str(metric_to_check)] / current_prod_run.data.metrics[str(metric_to_check)]
  model_diff_percent = round((model_diff_fraction - 1)*100, 2)
  print(f"Staging run's {metric_to_check}: {round(current_staging_run.data.metrics[str(metric_to_check)],3)}")
  print(f"Current production run's {metric_to_check}: {round(current_prod_run.data.metrics[str(metric_to_check)],3)}")

  if model_diff_percent >= 0 and (model_diff_fraction - 1 >= min_model_perf_threshold):
    print(f"The current staging run exceeds the model improvement threshold of at least +{min_model_perf_threshold}. You may proceed with transitioning the staging model to production now.")
    
  elif model_diff_percent >= 0 and (model_diff_fraction - 1  < min_model_perf_threshold):
    print(f"CAUTION: The current staging run does not meet the improvement threshold of at least +{min_model_perf_threshold}. Transition the staging model to production with caution.")
  else: 
    print(f"ALERT: The current staging run underperforms by {model_diff_percent}% when compared to the production model. Do not transition the staging model to production.")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,define widgets for drift
dbutils.widgets.removeAll()
dbutils.widgets.text("stats_threshold_limit", "0.5")
dbutils.widgets.text("p_threshold", "0.05")
dbutils.widgets.text("min_model_r2_threshold", "0.1")

stats_threshold_limit = float(dbutils.widgets.get("stats_threshold_limit"))       # how much we should allow basic summary stats to shift 
p_threshold = float(dbutils.widgets.get("p_threshold"))                           # the p-value below which to reject null hypothesis 
min_model_r2_threshold = float(dbutils.widgets.get("min_model_r2_threshold"))     # minimum model improvement

# COMMAND ----------

# DBTITLE 1,define variables
run_name='Drift_model_vanquishers'
delta_path='dbfs:/FileStore/Vanquishers/'
project_home_dir = f"/FileStore/Vanquishers/"
project_local_tmp_dir = "/dbfs" + project_home_dir + "tmp/"

model_params = {'n_estimators': 1100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 15}

misc_params = {"target_col": "selling_price",
               "num_cols": ['km_driven','no_of_year','fuel_CNG','fuel_Diesel','fuel_LPG','fuel_Petrol','seller_type_Individual','seller_type_Trustmark_Dealer','transmission_Manual','owner_First_Owner','owner_Second_Owner','owner_Test_Drive_Car','owner_Third_Owner']}



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,log model
model_1_run = train_sklearn_rf_model(run_name, delta_path, model_params, misc_params, seed=42)

# COMMAND ----------

# Register model to MLflow Model Registry
registry_model_name = 'Drift_model_vanquishers'

model_1_run_id = model_1_run.info.run_id
model_1_version = mlflow.register_model(model_uri=f"runs:/{model_1_run_id}/model", name=registry_model_name)

# COMMAND ----------

# Transition model to PROD
model_1_version = transition_model(model_1_version, stage="Production")
print(model_1_version)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Check Drift
month_1_error_delta_path = 'dbfs:/FileStore/Vanquishers_inbound/'

month_1_err_pdf = transformation(month_1_error_delta_path).toPandas()

# Compute summary statistics on new incoming data
summary_stats_month_1_err_pdf = create_summary_stats_pdf(month_1_err_pdf)

# COMMAND ----------

current_prod_pdf_1_transform = transformation('dbfs:/FileStore/Vanquishers/').toPandas()

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Production
current_prod_run_1 = get_run_from_registered_model(registry_model_name, stage="Production")

# Load in original versions of Delta table used at training time for current Production model
current_prod_pdf_1 = load_delta_table_from_run(current_prod_run_1).toPandas()

# Load summary statistics pandas DataFrame for data which the model currently in Production was trained and evaluated against
current_prod_stats_pdf_1 = load_summary_stats_pdf_from_run(current_prod_run_1, project_local_tmp_dir)

# COMMAND ----------

print("CHECKING PROPORTION OF NULLS.....")
check_null_proportion(month_1_err_pdf, null_proportion_threshold=.5)

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
check_dist_ks_bonferroni_test(current_prod_pdf_1_transform, month_1_err_pdf, misc_params['num_cols'], p_threshold)

# COMMAND ----------

model_1_run_drift = train_sklearn_rf_model(run_name, month_1_error_delta_path, model_params, misc_params, seed=42)

# COMMAND ----------

# Register model to MLflow Model Registry
registry_model_name = 'Drift_model_vanquishers'

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

month_1_model_version = transition_model(model_1_version, stage="Production")
print(month_1_model_version)

# COMMAND ----------


