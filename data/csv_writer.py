
# Import necessary libraries and functions

import pandas as pd
from datetime import datetime
import logging
import os
import re
import numpy as np
logging.basicConfig(level=logging)
logger = logging.getLogger(__name__)


class EntryWriter:
    def get_next_index(self, base_filename, folder="."):
        pattern = re.compile(rf"{re.escape(base_filename)}_(\d+)\.csv")
        max_index = 0
        for fname in os.listdir(folder):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)
        return max_index + 1

    def __init__(self, base_filename="data_results"):
        self.file_index = self.get_next_index(base_filename) 
        self.data_entry_count=0
        self.HN_entry_count=0
        # Create a dictionary of file handles for each HN
        self.current_file = self._open_file("data_results")
        self.HN_file=self._open_file("HN_results")


    def _open_file(self, filename):
        """Open one file per HN and return as dictionary."""
        filename = f"{filename}_{self.file_index}.csv"
        file = open(filename, "w", buffering=1)  # Line buffering
        return file
    
    def close(self):
        """Close all open files."""
        if hasattr(self, 'current_file') and self.current_file:
            self.current_file.close()
        if hasattr(self, 'HN_file') and self.HN_file:
            self.HN_file.close()
    
    def flush(self):
        """Flush all file buffers."""
        if hasattr(self, 'current_file') and self.current_file:
            self.current_file.flush()
        if hasattr(self, 'HN_file') and self.HN_file:
            self.HN_file.flush()

    
    def write_entry(self, results, price, margin_sell, margin_buy, date):
            """Write an entry to the current file, creating a new file if needed."""
            self_consumption_direkt=0
            for node, group in  results["agents"].groupby('Node'):
                # Sum for "res" Agent Type
                res_sum = abs(group[group['Agent Type'] == 'res']['Energy sold [kWh]'].sum())
                # Sum for Agent Type that is not "res" or "ext_grid"
                other_sum = abs(group[~group['Agent Type'].isin(['res', 'ext_grid', "storage"])]['Energy bought [kWh]'].sum())
                # Take the minimum of both sums
                self_consumption_direkt += min(res_sum,other_sum)
                ext_energy_ev_old= 0
            entry = pd.DataFrame(
                [
                    [
                        date.strftime("%d.%m.%Y %H:%M"),
                        results["supply"][results["supply"].loc[:,"Agent Type"]=="ext_grid"].loc[:,"Energy [kWh]"].values[0]-results["demand"][results["demand"].loc[:,"Agent Type"]=="ext_grid"].loc[:,"Energy [kWh]"].values[0],
                        results["demand"]["Energy [kWh]"].sum()-results["demand"][results["demand"].loc[:,"Agent Type"]=="ext_grid"].loc[:,"Energy [kWh]"].values[0],
                        results["supply"].loc[results["supply"]["Agent Type"].isin(["res"])]["Energy [kWh]"].sum(),
                        results["Node_results"]["HN Self-Consumption [kWh]"].sum(),
                        self_consumption_direkt,
                        results["Node_results"]["HN Energy LEC [kWh]"][ results["Node_results"]["HN Energy LEC [kWh]"]>0].sum(),
                        results["demand"].loc[results["demand"]["Agent Type"].isin(["farm","household","industry"])]["Energy [kWh]"].sum(),
                        results["demand"].loc[results["demand"]["Agent Type"].isin(["heatpump"])]["Energy [kWh]"].sum(),
                        results["demand"].loc[results["demand"]["Agent Type"].isin(["EV"])]["Energy [kWh]"].sum(),
                        results["agents"].loc[results["agents"]["Agent Type"].isin(["storage"])]["Energy bought [kWh]"].sum(),
                        results["agents"].loc[results["agents"]["Agent Type"].isin(["storage"])]["Energy sold [kWh]"].sum(),
                        price/100,
                        results["supply"][results["supply"].loc[:,"Agent Type"]=="ext_grid"].loc[:,"Energy [kWh]"].values[0]*margin_buy/100+results["demand"][results["demand"].loc[:,"Agent Type"]=="ext_grid"].loc[:,"Energy [kWh]"].values[0]*margin_sell/100,
                        abs(results["agents"][results["agents"]["Agent Type"]=="res"]["Revenue Energy LEC [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"]=="res"]["Revenue Energy External [€]"].sum()), 
                        abs(results["agents"][results["agents"]["Agent Type"].isin(["farm","household","industry"])]["Revenue Energy LEC [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"].isin(["farm","household","industry"])]["Revenue Energy External [€]"].sum()),
                        abs(results["agents"][results["agents"]["Agent Type"]=="heatpump"]["Revenue Energy LEC [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"]=="heatpump"]["Revenue Energy External [€]"].sum()), 
                        abs(results["agents"][results["agents"]["Agent Type"]=="EV"]["Revenue Energy LEC [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"]=="EV"]["Revenue Energy External [€]"].sum()),
                        abs(results["agents"].loc[(results["agents"]["Agent Type"] == "storage") &(results["agents"]["Revenue Energy External [€]"] > 0), "Revenue Energy External [€]"].sum()),
                        abs(results["agents"].loc[(results["agents"]["Agent Type"] == "storage") &(results["agents"]["Revenue Energy External [€]"] < 0), "Revenue Energy External [€]"].sum()),
                        abs(results["agents"].loc[(results["agents"]["Agent Type"] == "storage") &(results["agents"]["Revenue Energy LEC [€]"] > 0), "Revenue Energy LEC [€]"].sum()),
                        abs(results["agents"].loc[(results["agents"]["Agent Type"] == "storage") &(results["agents"]["Revenue Energy LEC [€]"] < 0), "Revenue Energy LEC [€]"].sum()),
                        results["agents"]["Fees and Levies External [€]"].sum(),
                        results["agents"]["Fees and Levies LEC [€]"].sum(),
                        abs(results["agents"][results["agents"]["Agent Type"].isin(["farm","household","industry"])]["Fees and Levies External [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"].isin(["farm","household","industry"])]["Fees and Levies LEC [€]"].sum()),
                        abs(results["agents"][results["agents"]["Agent Type"]=="heatpump"]["Fees and Levies External [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"]=="heatpump"]["Fees and Levies LEC [€]"].sum()), 
                        abs(results["agents"][results["agents"]["Agent Type"]=="EV"]["Fees and Levies External [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"]=="EV"]["Fees and Levies LEC [€]"].sum()),
                        abs(results["agents"][results["agents"]["Agent Type"]=="storage"]["Fees and Levies External [€]"].sum()),  
                        abs(results["agents"][results["agents"]["Agent Type"]=="storage"]["Fees and Levies LEC [€]"].sum()),
                        results["Input_Grid"]["Current loading max [%]"].values[0],
                        results["Input_Grid"]["Voltage deviation max [p.u.]"].values[0],
                        results["Input_Grid"]["Transformer loading [kVA]"].values[0]
                    ]
                ],
                columns=[
                    "Date",
                    "Energy Exchange External Grid [kWh]",
                    "Total Demand [kWh]",
                    "RES Generation [kWh]",
                    "Self Consumption total [kWh]",
                    "Self Consumption RES [kWh]",
                    "Energy Exchange LEC [kWh]",
                    "Energy Consumption inflexible Loads [kWh]",
                    "Energy Consumption Heatpumps [kWh]",
                    "Energy Consumption EVs [kWh]",
                    "Energy Brought Storages [kWh]",
                    "Energy Sold Storages [kWh]",
                    "Price Spotmarket [Euro/kWh]",
                    "Profit external grid [Euro]",
                    "Revenue RES LEC [Euro]",
                    "Revenue RES External [Euro]",
                    "Payment to LEC by inflexible Loads [Euro]",
                    "Payment to Retailer by inflexible Loads [Euro]",
                    "Payment to LEC by Heatpumps [Euro]",
                    "Payment to Retailer by Heatpumps [Euro]",
                    "Payment to LEC by EVs [Euro]",
                    "Payment to Retailer by EVs [Euro]",
                    "Payment to Retailer by Storages [Euro]",
                    "Payment from Retailer to Storages [Euro]",
                    "Payment to LEC by Storages [Euro]",
                    "Payment from LEC to Storages [Euro]",
                    "Total Fees and Levies External [Euro]",
                    "Total Fees and Levies LEC [Euro]",
                    "Fees and Levies External inflexible Loads [Euro]",
                    "Fees and Levies LEC inflexible Loads [Euro]",
                    "Fees and Levies External Heatpump [Euro]",
                    "Fees and Levies LEC Heatpump [Euro]",
                    "Fees and Levies External EV [Euro]",
                    "Fees and Levies LEC EV [Euro]",
                    "Fees and Levies External Storage internal [Euro]",
                    "Fees and Levies LEC Storage internal [Euro]",
                    "Max. Line Loading [%]",
                    "Max. Voltage Deviation [p.u.]",
                    "Transformer Usage [kVA]"
                    
                ],
            )
            dt = pd.to_datetime(date)
            if dt.day == 1 and dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                # Close the current files and start new ones for the new month
                self.current_file.close()
                self.HN_file.close()
                self.file_index += 1  # Increment file index for new month
                self.current_file = self._open_file("data_results")
                self.HN_file = self._open_file("HN_results")
                self.data_entry_count = 0
                self.HN_entry_count = 0  # Reset HN counter for new file
            if self.data_entry_count==0:
                self.current_file.write(';'.join(map(str, entry)) + '\n')
                self.data_entry_count+=1
            entry_string = ';'.join(map(str, entry.loc[0]))
            self.current_file.write(entry_string+ '\n')
            self.current_file.flush()
            del  entry

            for hn_id in range(len(results["Node_results"])):
                if hn_id == 5:  #only write for selected HN, where storage agent is in.
                    columns=[
                        "Date",
                        "Agent ID",
                        "Agent Type",
                        "SOC [rel]",
                        "T ind. [C°]",
                        "Energy Exchange Self-Supply [kWh]",
                        "Energy Exchange LEC [kWh]",
                        "Energy Exchange Retailer [kWh]",
                        "Revenue Energy LEC [€]",
                        "Revenue Energy Retailer [€]",
                        "Fees and Levies LEC [€]",
                        "Fees and Levies Retailer [€]"]
                    entry = pd.DataFrame(columns=columns)
                    ids=results["agents"][results["agents"]["Node"] == hn_id]["Agent ID"]
                    for agent_id in ids:
                        SOC_rel=None
                        T_in=None
                        agent_type="inflexible"
                        energy_ss=results["agents"][results["agents"]["Agent ID"] == agent_id]["HN Self-Consumption [kWh]"]
                        energy_lec=results["agents"][results["agents"]["Agent ID"] == agent_id]["Energy LEC [kWh]"]
                        energy_retailer=results["agents"][results["agents"]["Agent ID"] == agent_id]["Energy External [kWh]"]
                        revenue_lec=results["agents"][results["agents"]["Agent ID"] == agent_id]["Revenue Energy LEC [€]"]
                        revenue_retailer=results["agents"][results["agents"]["Agent ID"] == agent_id]["Revenue Energy External [€]"]
                        fees_lec=results["agents"][results["agents"]["Agent ID"] == agent_id]["Fees and Levies LEC [€]"]
                        fees_external=results["agents"][results["agents"]["Agent ID"] == agent_id]["Fees and Levies External [€]"]
                        if agent_id in results["EV"]["Agent ID"].values:
                            agent_type="EV"
                            SOC_rel=results["EV"][results["EV"]["Agent ID"]==agent_id]["SOC_rel"].values[0]
                        if agent_id in results["Heatpump"]["Agent ID"].values :    
                            agent_type="Heatpump"
                            T_in=results["Heatpump"][results["Heatpump"]["Agent ID"]==agent_id]["Temperature Indoor [°C]"].values[0]
                        if agent_id in results["storage"]["Agent ID"].values:    
                            agent_type="BESS"
                            SOC_rel=results["storage"][results["storage"]["Agent ID"]==agent_id]["SOC_rel"].values[0]
                        new_row = [
                                date.strftime("%d.%m.%Y %H:%M"),
                                agent_id,
                                agent_type,
                                SOC_rel,
                                T_in,
                                energy_ss.values[0],
                                energy_lec.values[0],
                                energy_retailer.values[0],
                                revenue_lec.values[0],
                                revenue_retailer.values[0],
                                fees_lec.values[0],
                                fees_external.values[0]]
                        entry = pd.DataFrame([new_row], columns=entry.columns)
                        if self.HN_entry_count == 0:
                            self.HN_file.write(';'.join(map(str, entry.columns)) + '\n')
                        for n in range(0,len(entry)):
                            entry_string = ';'.join(map(str, entry.loc[n]))
                            self.HN_file.write(entry_string + '\n')
                            self.HN_entry_count += 1
                    self.HN_file.flush()  # Flush after writing all agents for this node
