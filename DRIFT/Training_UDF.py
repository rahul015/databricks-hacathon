# Databricks notebook source
# MAGIC %run ./Transformation

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

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
#     pdf = df.toPandas()
    pdf = df.copy()
    
    # Create summary statistics pandas DataFrame and log as a csv to MLflow
    summary_stats_pdf = create_summary_stats_pdf(pdf)
    log_summary_stats_pdf_as_csv(summary_stats_pdf)  
    
    # Track number of total instances and "month"
    num_instances = pdf.shape[0]
    mlflow.log_param("num_instances", num_instances)  # Log number of instances
    
    # Split data
    table_name='Vanquishers.car_dataset'
    feature_df = fs.read_table(name=table_name)
    feature_df1=feature_df.toPandas()
    X = feature_df1.drop([misc_params["target_col"], "index"], axis=1)
    y = feature_df1[misc_params["target_col"]]
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
      ColSpec("integer", "age_of_car"),
      ColSpec("integer", "fuel_CNG"),
      ColSpec("integer", "fuel_Diesel"),
      ColSpec("integer", "fuel_LPG"),
      ColSpec("integer", "fuel_Petrol"),
      ColSpec("integer", "fuel_Electric"),
      ColSpec("integer", "seller_type_Individual"),
      ColSpec("integer", "seller_type_Trustmark_Dealer"),
      ColSpec("integer", "seller_type_Dealer"),
      ColSpec("integer", "transmission_Manual"),
      ColSpec("integer", "transmission_Automatic"),
      ColSpec("integer", "Owner")
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

# def train_sklearn_rf_model(run_name, delta_path, model_params, misc_params, seed=42):
#   """
#   Function to trigger training and evaluation of an sklearn RandomForestRegressor model.
#   Parameters, metrics and artifacts are logged to MLflow during this process.
#   Return the MLflow run object 
  
#   :param run_name: (str) name to give to MLflow run
#   :param delta_path: (str) path to Delta table to use as input data
#   :param model_params: (dict) Dictionary of model parameters to pass into sklearn RandomForestRegressor
#   :param misc_params: (dict) Dictionary of params to use 
  
#   :return: mlflow.entities.run.Run  
#   """  
#   with mlflow.start_run(run_name=run_name) as run:

#     # Enable MLflow autologging
#     mlflow.autolog(log_input_examples=True, silent=True)
    
#     # Load Delta table from delta_path
# #     df = spark.read.format("delta").load(delta_path)   
#     df = transformation(delta_path) 
    
#     # Log Delta path and version
#     mlflow.log_param("delta_path", delta_path)
#     delta_version = get_delta_version(delta_path)
#     mlflow.log_param("delta_version", delta_version)
    
#     # Track misc parameters used in pipeline creation (preprocessing) as json artifact
#     mlflow.log_dict(misc_params, "preprocessing_params.json")
#     target_col = misc_params["target_col"]  
#     num_cols = misc_params["num_cols"]    

#     # Convert Spark DataFrame to pandas, as we will be training an sklearn model
#     pdf = df.toPandas()     
    
#     # Create summary statistics pandas DataFrame and log as a csv to MLflow
#     summary_stats_pdf = create_summary_stats_pdf(pdf)
#     log_summary_stats_pdf_as_csv(summary_stats_pdf)  
    
#     # Track number of total instances and "month"
#     num_instances = pdf.shape[0]
#     mlflow.log_param("num_instances", num_instances)  # Log number of instances
    
#     # Split data
#     X = pdf.drop([misc_params["target_col"], "index"], axis=1)
#     y = pdf[misc_params["target_col"]]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#     # Track train/test data info as params
#     num_training = X_train.shape[0]
#     mlflow.log_param("num_training_instances", num_training)
#     num_test = X_test.shape[0]
#     mlflow.log_param("num_test_instances", num_test)

#     # Fit sklearn pipeline with RandomForestRegressor model
#     rf_pipeline = create_sklearn_rf_pipeline(model_params)
#     rf_pipeline.fit(X_train, y_train)
#     # Specify data schema which the model will use as its ModelSignature
#     input_schema = Schema([
#       ColSpec("integer", "km_driven"),
#       ColSpec("integer", "age_of_car"),
#       ColSpec("integer", "fuel_CNG"),
#       ColSpec("integer", "fuel_Diesel"),
#       ColSpec("integer", "fuel_LPG"),
#       ColSpec("integer", "fuel_Petrol"),
#       ColSpec("integer", "seller_type_Individual"),
#       ColSpec("integer", "seller_type_Trustmark_Dealer"),
#       ColSpec("integer", "transmission_Manual"),
#       ColSpec("integer", "owner_First_Owner"),
#       ColSpec("integer", "owner_Second_Owner"),
#       ColSpec("integer", "owner_Test_Drive_Car"),
#       ColSpec("integer", "owner_Third_Owner")
#     ])
#     output_schema = Schema([ColSpec("double")])
#     signature = ModelSignature(input_schema, output_schema)
#     mlflow.sklearn.log_model(rf_pipeline, "model", signature=signature)

#     # Evaluate the model
#     predictions = rf_pipeline.predict(X_test)
#     test_mse = mean_squared_error(y_test, predictions) 
#     r2 = r2_score(y_test, predictions)
#     mlflow.log_metrics({"test_mse": test_mse,
#                        "test_r2": r2})

#   return run

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


