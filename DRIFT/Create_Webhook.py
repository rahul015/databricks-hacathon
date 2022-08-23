# Databricks notebook source
from mlflow.utils.rest_utils import http_request
from mlflow.utils.databricks_utils import get_databricks_host_creds
import json
import urllib

# COMMAND ----------

job_id = 1022354934509754  # job_id need to be updated
model_name = 'Vanquishers_CAR_DATASET_RF_MODEL'

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
# This ojbect comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
# Lastly, extract the databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

url = f"https://{instance}"
endpoint = "/api/2.0/mlflow/registry-webhooks/create"
host_creds = get_databricks_host_creds("databricks")

# COMMAND ----------

job_json = {"model_name": model_name,
            "events": ["MODEL_VERSION_TRANSITIONED_TO_STAGING"],
            "description": "Vanquishers Job webhook trigger",
            "status": "Active",
            "job_spec": {"job_id": job_id,
                         "workspace_url": url,
                         "access_token": token}
           }


response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="POST",
    json=job_json
)

# COMMAND ----------

teams_incoming_webhook = "https://accenture.webhook.office.com/webhookb2/2368ff28-b83f-4d15-b0f1-a9342feea6e7@e0793d39-0939-496d-b129-198edd916feb/IncomingWebhook/b31525d2b443441fa0ebf51968f4dd4d/02abc593-23f9-4fa5-9973-746ef14db27a" 

http_json = {"model_name": model_name,
  "events": ["MODEL_VERSION_TRANSITIONED_TO_STAGING"],
  "description": "Vanquishers teams webhook trigger",
  "status": "Active",
  "http_url_spec": {
    "url": teams_incoming_webhook,
    "enable_ssl_verification": "false"}}


response = http_request(
  host_creds=host_creds, 
  endpoint=endpoint,
  method="POST",
  json=http_json
)

# COMMAND ----------

endpoint = f"/api/2.0/mlflow/registry-webhooks/list/?model_name={model_name.replace(' ', '%20')}"

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="GET"
)

print(response.json())

# COMMAND ----------

# delete_hook = "e5b02ed3ef3b400a8f6910c180b82b00"
# new_json = {"id": delete_hook}
# endpoint = f"/api/2.0/mlflow/registry-webhooks/delete"

# response = http_request(
#     host_creds=host_creds, 
#     endpoint=endpoint,
#     method="DELETE",
#     json=new_json
# )

# print(response.json())

# COMMAND ----------


