# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:44:41 2023
@autor: mjulschm
"""


import gc
from mesa_model.model import model
from data.csv_writer import EntryWriter

writer = EntryWriter()
      
for n in range(model.total_steps):
            model.step()
            result = model.results[int(model.stepcount)]
            writer.write_entry(result, model.market_price[model.market_price["time"]==model.current_date]["price"].values[0], model.market_price_margin_sell, model.market_price_margin_buy, model.current_date)                
            if n % 96 == 0:
                gc.collect()
                print(f"Simulation completed a day")  
writer.close()
print("Simulation complete - files closed")
    
