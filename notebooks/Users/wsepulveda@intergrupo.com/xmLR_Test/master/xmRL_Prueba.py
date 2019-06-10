# Databricks notebook source
# MAGIC %run "/Users/wsepulveda@intergrupo.com/xmLR_Test/main/db_test"

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:36:32 2019

@author: WSEPULVE
"""
#comentario
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

class simple_RL():
    
    def carga_datos(self):
        
        #datasets = pd.read_csv('E:\\Proyecto XM\\xmLR_Test\\Salary_Data.csv')
        #datasets = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('/mnt/xmdatalake/Salary_Data.csv').toPandas()
        #datasets= datasets.toPandas()
        
        pushdown_query = "(select * from Salary_Data) sal"
        datasets = spark.read.jdbc(url=jdbcUrl, table=pushdown_query, properties=connectionProperties).toPandas()
        
        return datasets
        
    def entrenamiento(self,datasets):
                        
        X = datasets.iloc[:, :-1].values
        Y = datasets.iloc[:, 1].values
        
        from sklearn.model_selection import train_test_split
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
        
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_Train, Y_Train)
        
        pickle_out = open("/dbfs/mnt/xmdatalake/regressor.sav","wb")
        pickle.dump(regressor, pickle_out)
        pickle_out.close()
        
        df_X_Test = pd.DataFrame(X_Test)
        df_X_Test.to_csv("/dbfs/mnt/xmdatalake/df_X_Test.csv")
        
        self.xtest = X_Test
        self.model = regressor  
        
    def prediccion(self):
        
        with open("/dbfs/mnt/xmdatalake/regressor.sav", 'rb') as file:
                model=pickle.load(file) 
        #model = self.model
        
        xtest = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('/mnt/xmdatalake/df_X_Test.csv').toPandas()
        df = xtest[xtest.columns[1]]
        xtest = np.asarray(df)
        xtest = xtest.reshape(-1, 1)
        
        #Y_Pred = regressor.predict(xtest)
        
        Y_Pred = model.predict(xtest)
        #print(Y_Pred)
        
        df_Y_Pred = pd.DataFrame(Y_Pred)
        df_Y_Pred.to_csv("/dbfs/mnt/xmdatalake/df_Y_Pred.csv")
