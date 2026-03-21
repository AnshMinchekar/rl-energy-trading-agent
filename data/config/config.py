# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:03:11 2024

@author: mjulschm
"""

import pandas as pd
import yaml
import os
import logging
from datetime import timedelta
class Config:
    
    @property
    def main(self):
        return self.config.get("main")
    @property
    def storage(self):
        return self.config.get("storage")
    @property
    def heatpump(self):
        return self.config.get("heatpump")
    @property
    def ev(self):
        return self.config.get("ev")
    @property
    def farm(self):
        return self.config.get("farm")
    @property
    def household(self):
        return self.config.get("household")
    @property
    def res(self):
        return self.config.get("res")
    @property
    def industry(self):
        return self.config.get("industry")
    @property
    def ext_grid(self):
        return self.config.get("ext_grid")
    @property
    def start_time(self):
        return pd.to_datetime(self.config["main"]["simulation_start_time"],format="%d.%m.%Y %H:%M")
    
    @property
    def end_time(self):
        return pd.to_datetime(self.config["main"]["simulation_end_time"],format="%d.%m.%Y %H:%M")
    
    _instance = None
    def __new__(cls, config_file=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            if config_file is None:
                # Use the correct path for config.yaml
                config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
            cls._instance._load_config(config_file)
            cls._instance._load_scenario_data()
        return cls._instance


    def _load_config(self, config_file):
        try:
            with open(config_file, "r") as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Failed to load config-data: {e}")
            raise



    def _load_scenario_data(self):
        try:
            scenario_data_path = os.path.join(os.path.dirname(__file__), "scenario_data")
            end_time=self.end_time+timedelta(minutes=self.config["main"]["timestep"])*(self.config["ev"]["forecast_steps"]+2)
            for file in os.listdir(scenario_data_path):
                if file.endswith('.csv'):
                    name = os.path.splitext(file)[0]
                    file_path = os.path.join(scenario_data_path, file)
                    df = pd.read_csv(file_path, sep=";", decimal=",")
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'],format="%d.%m.%Y %H:%M")
                        filtered_df = df[(df['time'] >= self.start_time) & (df['time'] <= end_time)]
                        # Read the CSV file and append the DataFrame to the list
                        setattr(self, name, filtered_df)
                    else:
                        setattr(self,name, df)
                
                if file.endswith('.parquet'):
                    name = os.path.splitext(file)[0]
                    file_path = os.path.join(scenario_data_path, file)


                    # Load only specific columns to save memory
                    columns_needed = ["Type"] + list(pd.date_range(self.start_time, end_time, freq="15min").astype(str))
                    available_columns = pd.read_parquet(file_path, engine='pyarrow').columns
                    valid_columns = [col for col in columns_needed if col in available_columns]
                    try:
                        
                        datareduced = pd.read_parquet(file_path, columns=valid_columns)
                        if datareduced.empty or datareduced.drop(columns=["Type"], errors="ignore").dropna(how="all").empty:
                            continue
                        else:
                        # Load only the necessary columns (much more memory-efficient)
        
                            datareduced.index = datareduced["Type"]
                            datareduced = datareduced.drop(columns=["Type"])
                            # Convert columns to datetime format
                            datareduced.columns = pd.to_datetime(datareduced.columns, format="%Y-%m-%d %H:%M:%S")
                            datareduced = datareduced.dropna(how='all')
                            setattr(self, name, datareduced)
                    except Exception as e:
                        print(f"❌ Error processing file {file}: {e}")
                                        
        except Exception as e:
            logging.error(f"Failed to load sceanrio-data: {e}")
            raise
            

config=Config()