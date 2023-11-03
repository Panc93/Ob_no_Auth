# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:09:50 2023

@author: ParnikaPancholi
"""
import configparser
from sqlalchemy import create_engine
import snowflake.connector
# Read configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

def Sqlconnection():
  
    # Get credentials from the configuration
    db_username = config['Sql']['db_user']
    db_password = config['Sql']['db_password']
    db_host = config['Sql']['db_host']
    db_database = config['Sql']['db_name']
    
    # Create database connection
    Sqlengine = create_engine(f'mysql+mysqlconnector://{db_username}:{db_password}@{db_host}/{db_database}')

    return Sqlengine

def Snowflakeconnection():
     # Access Snowflake connection parameters
     account = config['snowflake']['account']
     warehouse = config['snowflake']['warehouse']
     database = config['snowflake']['database']
     schema = config['snowflake']['schema']
     user = config['snowflake']['user']
     password = config['snowflake']['password']
    
     conn = snowflake.connector.connect(
            user=user,
            password=password,
            account= account,
            warehouse = warehouse,
            database= database,
            schema =schema
            )
     # Create SQLAlchemy engine
     Snowflakeengine = create_engine('snowflake://', creator=lambda: conn)
     return Snowflakeengine
          
    
def Sqlclose():
    Sqlengine.dispose()
    print('Sql Connection close')

def Snoflakeclose():
    Snowflakeengine.dispose()
    print('Snowflake Connection close')
