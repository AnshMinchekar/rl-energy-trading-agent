# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:44:41 2023
@autor: mjulschm
"""

import gc
import sys
import time
sys.stdout.reconfigure(line_buffering=True)  # flush every newline, even when piped/redirected

from mesa_model.model import model
from data.csv_writer import EntryWriter

writer = EntryWriter()

_start_time = time.time()
for n in range(model.total_steps):
            model.step()
            result = model.results[int(model.stepcount)]
            writer.write_entry(result, model.market_price[model.market_price["time"]==model.current_date]["price"].values[0], model.market_price_margin_sell, model.market_price_margin_buy, model.current_date)
            if n % 96 == 0:
                gc.collect()
                elapsed = time.time() - _start_time
                pct = 100 * n / model.total_steps
                print(
                    f"[{model.current_date.strftime('%d.%m.%Y')}]"
                    f"  step {n:>6}/{model.total_steps}"
                    f"  ({pct:5.1f}%)"
                    f"  elapsed {elapsed/60:6.1f} min",
                    flush=True,
                )
writer.close()
print("Simulation complete - files closed", flush=True)
