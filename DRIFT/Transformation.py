# Databricks notebook source
#Importing required libraries
import pandas as pd
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

# Getting all the dummy column names for categorical columns
columns=[]
fuel=['Diesel','Petrol','CNG','LPG','Electric']
seller_type=['Individual','Dealer','Trustmark_Dealer']
transmission=['Manual','Automatic']
for i in fuel:
    columns.append('fuel_'+i)
for i in seller_type:
    columns.append('seller_type_'+i)
for i in transmission:
    columns.append('transmission_'+i)
print(columns)

# COMMAND ----------

#Label Encoding
def label_encoding(z):
    if z=='Test Drive Car':
        return 1
    elif z=='First Owner':
        return 1
    elif z=='Second Owner':
        return 2
    elif z=='Third Owner':
        return 3
    else:
        return 4

# COMMAND ----------

#Getting the age of the car using year column
def age_of_car(a):
    return 2020-a

# COMMAND ----------



# COMMAND ----------

# Doing all transformations in a single function
def transformation(delta_path):
    df=spark.read.format('delta').load(delta_path).coalesce(1).withColumn("index", monotonically_increasing_id())
    df=df.withColumn("index",df.index.cast(IntegerType()))
    df1=df.toPandas()
    df1['Owner']=df1['owner'].apply(label_encoding)
    df1['age_of_car']=df1['year'].apply(age_of_car)
    df1.drop('owner',axis=1,inplace=True)
    df1.drop('name',axis=1,inplace=True)
    df1.drop('year',axis=1,inplace=True) 
    df1=pd.get_dummies(df1)
    df1.rename(columns = {'seller_type_Trustmark Dealer':'seller_type_Trustmark_Dealer'}, inplace = True)
    for i in columns:
        if i not in df1.columns:
            df1[i]=0
    return df1

    

# COMMAND ----------


