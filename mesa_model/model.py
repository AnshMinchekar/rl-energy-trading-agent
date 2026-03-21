# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:36:13 2023

@author: mjulschm
"""
import sys
from datetime import datetime, timedelta
import numpy as np
from mesa_model.agents import *
import sys
import pandas as pd
from optimization.market_optimizer import MarketOptimizer
import math
import simbench as sb
import pandapower as pp
import mesa
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import gc
import os
import math
from collections import defaultdict
from data.data_handler import Market_StaticData
from data.data_handler import Agent_Bid_StaticData, Agent_Ask_StaticData, Agent_Ask_RoundData, Agent_Bid_RoundData, Market_VariableData
from collections import defaultdict
from optimization.HN_optimizer import HNOptimizer


#---------------------------------------------------------------------------------
def compute_price(model):
    "Pricing rule"
    demand_prices=sorted([model.data["demand"].iloc[a,1] for a in range(0,len(model.data["demand"])) if model.data["demand"].iloc[a,2]>=0.001])
    supply_prices=sorted([model.data["supply"].iloc[a,1] for a in range(0,len(model.data["supply"])) if model.data["supply"].iloc[a,2]>=0.001])[::-1]
    
    market_clearing_price=(demand_prices[0]+supply_prices[0])/2
    model.data["clearing_price [€/kWh]"]=market_clearing_price
    return(market_clearing_price)

def step_subnet(subnet, agents):
    """Step all agents in one subnet sequentially."""
    for agent in agents:
        agent.step()           
#---------------------------------------------------       
class LEM(mesa.Model):
    "A model with some number of agents."
    def __init__(self, sb_grid, date):
        super().__init__()
        self.demand=0
        self.supply=0
        self.sref=config.main["sref"]
        self.round= int(round(math.log10(self.sref),0)) #Anzahl Nachkommastellen, auf die gerundet wird in pu Größen (immer 1 kW)
        self.results={}
        self.timestep=timedelta(minutes=config.main["timestep"])
        self.counter_max=20  #Nach wie vielen Zeitschritten die Agenten sich selbe wieder optimieren
        self.current_date=datetime.strptime(date, "%d.%m.%Y %H:%M")-timedelta(minutes=15)
        self.start_date_simulation=datetime.strptime(config.main["begin_scenario_data"], "%d.%m.%Y %H:%M")
        self.end_date=datetime.strptime(config.main["simulation_end_time"], "%d.%m.%Y %H:%M")
        self.start_date=datetime.strptime(config.main["simulation_start_time"], "%d.%m.%Y %H:%M")
        self.total_steps=int((self.end_date-self.start_date)/timedelta(minutes=15))
        self.stepcount=(self.current_date-self.start_date)/timedelta(minutes=15)
        self.temperature_df=config.temperature.loc[:, ["time", "Temperatur-Dortmund"]]
        self.temperature=float(self.temperature_df.loc[self.temperature_df["time"]==self.current_date+self.timestep].iloc[0,1])
        self.gridfee_LEC=config.main["gridfee_LEC"]
        self.gridfee_ext=config.main["gridfee_ext"]
        self.levies_LEC=config.main["levies_LEC"]
        self.levies_ext=config.main["levies_ext"]
        self.margin_charge_ext=config.ev["margin_charge_ext"]


        "Set up grid "
        self.grid = sb.get_simbench_net(sb_grid)
        self.grid.ev=pd.DataFrame(columns=["agent_id", "car", "home_node", "work_node", "work_loading_power", "profile"])
        self.grid.heatpump=pd.DataFrame(columns=["agent_id", "bus",  "C [Wh/(m^2*K)", "R  [K/kW]", "power [kW]", "size [m^2]", "cosphi"],)
        #self.grid.storage=pd.DataFrame(columns=["id", "bus", "capacity [kWh]", "power [kW]"])
        pp.toolbox.create_continuous_elements_index(self.grid)
        self.slack=self.grid.ext_grid["bus"][0]
        "Add Standard EV Information, adjust for specific buses in the following"
        self.grid.bus.loc[:, "ev_charger_p_max"] = config.ev["bus-info"]["standard"][0]
        self.grid.bus.loc[:, "num_ev"] = config.ev["bus-info"]["standard"][1]
        self.grid.bus.loc[:, "bus_type"] = "None"
    
        pp.toolbox.create_continuous_elements_index(self.grid)
        
        max_p_heatpump=0
        min_p_heatpump=0
        "Set up inflexible agents"           
        id_count=1
        i=0
        counter_farm = 0
        counter_industry = 0
        counter_household = 0
        evs_household = 0
        evs_farm = 0
        evs_industry = 0
        total_evs_created = 0
        for a in [self.grid.load, self.grid.sgen]:
            a["agent_id"]=0
            a["type"]=""
            for aa in a.index:
                    profile_str = a.profile[aa]
                    if profile_str.startswith("G"):
                        typ = "industry"
                    elif profile_str.startswith("L"):
                        typ = "farm"
                    elif profile_str.startswith("HLS") or profile_str.startswith("APLS"):
                        typ = "charging_station"
                    elif profile_str.startswith("PV"):
                        typ = "pv"
                    elif profile_str.startswith("WP"):
                        typ = "wind"
                    elif profile_str.startswith("BM"):
                        typ = "biomass"
                    elif profile_str.startswith("Hydro"):
                        typ = "hydro"
                    elif profile_str.startswith("H"):
                        typ = "household"
                    elif profile_str.startswith("Soil") or profile_str.startswith("Air"):
                        typ = "heatpump"
                        max_p_heatpump=max(a.at[aa,"p_mw"], max_p_heatpump)
                        min_p_heatpump=min(a.at[aa,"p_mw"], min_p_heatpump)
                    else:
                        typ = "unknown"
                    if typ in ["farm", "industry", "household", "pv", "wind", "biomass", "hydro"]:
                        a.at[aa,"agent_id"]=id_count  
                        a.at[aa,"type"]=typ
                        bus=a.at[aa,"bus"]
                        if typ in ["pv", "wind", "biomass", "hydro"]:
                            factor=a.at[aa,"p_mw"]*getattr(config, "res")["scale"][0]*1000
                            curve_array=[config.res_profile.columns[n] for n in range(0, len(config.res_profile.columns)) if config.res_profile.columns[n].startswith(typ)]
                            if curve_array:
                                curve=curve_array[i%len(curve_array)]
                            if typ=="pv":
                                i+=1
                            res(self, typ, factor, curve, bus)
                        elif typ in ["farm", "industry", "household"]:
                            factor=a.at[aa,"p_mw"]*getattr(config, typ)["scale"][0]*1000
                            self.grid.bus.loc[bus, "bus_type"]=typ
                            
                            # Get current counter for this type
                            if typ == "farm":
                                current_counter = counter_farm
                                counter_farm += 1
                            elif typ == "industry":
                                current_counter = counter_industry
                                counter_industry += 1
                            elif typ == "household":
                                current_counter = counter_household
                                counter_household += 1
                            
                            # Handle charger power iteration
                            charger_power_config = config.ev["bus-info"][typ][0]
                            if isinstance(charger_power_config, list):
                                charger_power = charger_power_config[current_counter % len(charger_power_config)]
                            else:
                                charger_power = charger_power_config
                            
                            # Handle num_evs based on percentage
                            ev_percentage_config = config.ev["bus-info"][typ][1]
                            if isinstance(ev_percentage_config, list):
                                ev_percentage = ev_percentage_config[current_counter % len(ev_percentage_config)]
                            else:
                                ev_percentage = ev_percentage_config
                            
                            # Calculate if we should create an EV based on percentage
                            if typ == "household":
                                current_ev_count = evs_household
                            elif typ == "farm":
                                current_ev_count = evs_farm
                            elif typ == "industry":
                                current_ev_count = evs_industry
                            
                            # Check if we should create an EV (percentage based)
                            num_evs=0
                            while total_evs_created == 0 or (current_ev_count / (current_counter+1)) < ev_percentage:
                                if ev_percentage!=0:
                                    num_evs += 1
                                    if typ == "household":
                                        evs_household += 1
                                    elif typ == "farm":
                                        evs_farm += 1
                                    elif typ == "industry":
                                        evs_industry += 1
                                    total_evs_created += 1
                                    current_ev_count+=1
                                else:
                                    break
                            self.grid.bus.loc[bus, "ev_charger_p_max"]=charger_power
                            self.grid.bus.loc[bus, "num_ev"]=num_evs
                            cosphi=getattr(config, typ)["cosphi"][current_counter%len(getattr(config, typ)["cosphi"])]
                            getattr(sys.modules[__name__], typ)(self, typ, factor, cosphi, bus)
                        id_count+=1
                    elif typ in ["charging_station"]:
                        a.drop(aa, axis=0, inplace=True)
            pp.toolbox.create_continuous_elements_index(self.grid)
        ext_grid(self)
        self.grid.ext_grid["agent_id"]=id_count
        self.id_ext_grid=id_count
        id_count+=1
        self.grid.ext_grid["type"]="ext. grid"

        "Set up flexible agents" 
        self.grid.storage["id"]="None"
        for a in range (0,len(self.grid.storage)):
            if self.grid.storage.loc[a]["bus"]!=4:
                storage(self, self.grid.storage.loc[a]["max_e_mwh"]*1000, self.grid.storage.loc[a]["p_mw"]*-1000, self.grid.storage.loc[a]["bus"],self.grid.storage.loc[a]["efficiency_percent"], 1-self.grid.storage.loc[a]["self-discharge_percent_per_day"], "optimisation")
            self.grid.storage.loc[a]["id"]=id_count
            id_count+=1
        a=0
        storage(self, self.grid.storage.loc[a]["max_e_mwh"]*1000, self.grid.storage.loc[a]["p_mw"]*-1000, 5,self.grid.storage.loc[a]["efficiency_percent"], 1-self.grid.storage.loc[a]["self-discharge_percent_per_day"], "learning")
        
        for aa in self.grid.load.index:
                profile_str = self.grid.load.at[aa,"profile"]
                if profile_str.startswith("Soil") or profile_str.startswith("Air"):
                        bus_type=self.grid.bus.at[self.grid.load.at[aa,"bus"],"bus_type"]
                        size=self.grid.load.at[aa,"p_mw"]/max_p_heatpump*1300 #(max size 1300 m2)
                        R=config.heatpump["bus-info"][bus_type][0]/(size/120)  #120m2 is standard size in config
                        C=config.heatpump["bus-info"][bus_type][1]
                        cosphi=math.cos(math.radians(self.grid.load.at[aa,"p_mw"]/self.grid.load.at[aa,"sn_mva"]))
                        heatpump(self, R, C, self.grid.load.at[aa,"p_mw"]*1000*(size/120), size, cosphi, self.grid.load.at[aa,"bus"],"optimisation")
                        self.grid.heatpump.loc[len(self.grid.heatpump)]=[id_count, self.grid.load.at[aa,"bus"],C,R,self.grid.load.at[aa,"p_mw"]*1000*(size/120),size,cosphi]
                        id_count+=1          
        for i in range(self.grid.bus["num_ev"].sum()):
            profile = config.ev["driving_profile"][i%len(config.ev["driving_profile"])]
            for a in range(0,len(self.grid.bus)):
                if self.grid.bus["num_ev"][0:a].sum()>=i:
                    home_node=a
                    break
            if i/self.grid.bus["num_ev"].sum()<config.ev["work_inside"]:
                    work_node_str="inside"
                    work=self.grid.bus[(self.grid.bus["bus_type"]=="industry")| (self.grid.bus["bus_type"]=="farm")]
                    ii=i
                    if i>=len(work):
                        ii=i%(i//len(work)*len(work))
                    ii=len(work)-ii-1
                    work_node=self.grid.bus.index[(self.grid.bus["bus_type"]=="industry")| (self.grid.bus["bus_type"]=="farm")][ii]
                    work_loading_power=self.grid.bus.loc[work_node,"ev_charger_p_max"]
            else:
                    work_node_str="outside"
                    work_node=np.nan
                    work_loading_power=config.ev["bus-info"]["outside"][0]
            iii=i
            if i>=len(config.cars["type"]):
                    iii=i%(i//len(config.cars["type"])*len(config.cars["type"]))
            car=config.cars["type"][iii]   
            id_count+=1
            if home_node==self.slack:
                home_node=2
            if work_node==self.slack:
                work_node=4

            a = EV(self, car, home_node,  work_node, work_loading_power, profile, "optimisation")
            self.grid.ev.loc[i]=[id_count,car, home_node,  work_node, work_loading_power, profile]
        for agent in self.agents:
            agent.subnet = agent.model.grid.bus.subnet[agent.model.grid.bus.index == agent.bus].values[0]
        self.agent_groups = defaultdict(list)
        for agent in self.agents:
            self.agent_groups[agent.subnet].append(agent)
        [pp.toolbox.drop_elements(self.grid, "load", n) for n in self.grid.load.index]
        [pp.toolbox.drop_elements(self.grid, "sgen", n) for n in self.grid.sgen.index]
        [pp.toolbox.drop_elements(self.grid, "storage", n) for n in self.grid.storage.index]
        self.market_price_margin_buy=1
        self.market_price_margin_sell=0.3
        self.market_price_margin_charge=[agents.margin_charge for agents in self.agents if agents.flex==2][0]
        self.HEM_dict=self.build_HEM_dict()
        self.solver="gurobi"



    def initialize_static_data(self):
        # Include ALL LEC participants, not just those currently bidding/asking
        # This allows agents (like RL storage) to start bidding later without causing ID mismatches
        asks_static=[Agent_Ask_StaticData(
                    agent_id=a.unique_id,
                    bus=a.bus,
                    cosphi=a.cosphi,
                    asking_type=a.type,
                    agent_type=a.typ
                )
               for a in self.agents if a.LEC_participation==True]

        bids_static=[
                Agent_Bid_StaticData(
                    agent_id=a.unique_id,
                    bus=a.bus,
                    cosphi=a.cosphi,
                    bidding_type=a.type,
                    agent_type=a.typ
                )
               for a in self.agents if a.LEC_participation==True]
            
        market_static = Market_StaticData(
            n_bids=len(bids_static),
            n_asks=len(asks_static),
            n_nodes=len(self.grid.bus),
            slack_bus=self.slack,
            gridfee_LEC=self.gridfee_LEC,
            levies_LEC=self.levies_LEC,
            gridfee_ext=self.gridfee_ext,
            levies_ext=self.levies_ext,
            slack_agent_id=self.id_ext_grid,
            s_ref=self.sref,
            timestep=self.timestep.seconds,
            margin_buy=self.market_price_margin_buy,
            margin_sell=self.market_price_margin_sell,               
            solver=self.solver
        )

        return market_static, bids_static, asks_static

    def get_round_data(self):
        new_ask_data = [Agent_Ask_RoundData(
                agent_id=a.unique_id,
                p_min=a.ask[0],
                p_max=a.ask[1],
                f_coef=a.coefficients_ask)
                for a in self.agents if a.ask!=0 and a.LEC_participation==True]

        new_bid_data = [Agent_Bid_RoundData(
                agent_id=a.unique_id,
                p_min=a.bid[0],
                p_max=a.bid[1],
                f_coef=a.coefficients_bid)
                for a in self.agents if a.bid!=0 and a.LEC_participation==True]
        
        new_market_variable=[Market_VariableData(
                date=self.current_date.strftime("%d.%m.%Y %H:%M"),
                timestep=int(self.stepcount),
                temperature=self.temperature,
                ext_price=[agents.price for agents in self.agents if agents.flex==999][0])
                ]
        
        new_non_LEC_data = [
            {
                'agent_id': a.unique_id,
                'p_buy': getattr(a, 'optimal_power_buy_current', 0) if hasattr(a, 'optimal_power_buy_current') else (a.bid[0] if (hasattr(a, 'bid') and a.bid != 0 and hasattr(a.bid, '__getitem__')) else 0),
                'p_sell': getattr(a, 'optimal_power_sell_current', 0) if hasattr(a, 'optimal_power_sell_current') else (a.ask[0] if (hasattr(a, 'ask') and a.ask != 0 and hasattr(a.ask, '__getitem__')) else 0),
                'p_out': getattr(a, 'optimal_power_out_current', 0) if hasattr(a, 'optimal_power_out_current') else 0,
                'bus': a.bus,
                "type": a.typ,
                "HN_self_consumption": 0}
            for a in self.agents if hasattr(a, 'LEC_participation') and a.LEC_participation == False
        ]
        
        # Group by bus and calculate HN self-consumption
        from collections import defaultdict
        bus_groups = defaultdict(list)
        for entry in new_non_LEC_data:
            bus_groups[entry['bus']].append(entry)
        
        # Calculate self-consumption for each bus
        bus_self_consumption = {}
        for bus, entries in bus_groups.items():
            total_p_buy = sum(e['p_buy'] for e in entries)
            total_p_sell = sum(e['p_sell'] for e in entries)
            bus_self_consumption[bus] = min(total_p_buy, total_p_sell)
        
        # Distribute self-consumption proportionally to each agent on the bus
        for entry in new_non_LEC_data:
            bus = entry['bus']
            bus_sc = bus_self_consumption.get(bus, 0)
            
            if bus_sc > 0:
                # Get all entries for this bus
                bus_entries = bus_groups[bus]
                
                # Calculate agent's share based on their contribution
                if entry['p_buy'] > 0:
                    # For buyers: proportional to their p_buy
                    total_bus_buy = sum(e['p_buy'] for e in bus_entries)
                    if total_bus_buy > 0:
                        entry['HN_self_consumption'] = bus_sc * (entry['p_buy'] / total_bus_buy)
                elif entry['p_sell'] > 0:
                    # For sellers: proportional to their p_sell (negative value)
                    total_bus_sell = sum(e['p_sell'] for e in bus_entries)
                    if total_bus_sell > 0:
                        entry['HN_self_consumption'] = -bus_sc * (entry['p_sell'] / total_bus_sell)
        
        # Sort by bus
        new_non_LEC_data = sorted(new_non_LEC_data, key=lambda x: x['bus'])
        
        return new_bid_data, new_ask_data, new_market_variable, new_non_LEC_data

    def build_HEM_dict(self):
        agents_by_bus = defaultdict(list)

        # First, group agents by bus
        for agent in self.agents:
            agents_by_bus[agent.bus].append(agent)
        
        # Get all unique buses
        all_buses = list(agents_by_bus.keys())
        total_buses = len(all_buses)
        target_yes_percentage = 1
        
        # LEC assignment logic at bus level: 60% yes, 40% no
        yes_count = 0
        no_count = 0
        current_mode = True # Start with yes
        
        for bus in all_buses:
            if bus==self.slack:
                continue  # Skip slack bus
            # Assign LEC to all agents on this bus
            for agent in agents_by_bus[bus]:
                agent.LEC_participation = current_mode
            
            # Update counters and switch mode if needed
            if current_mode == True:
                yes_count += 1
                # Check if we should switch to "no"
                current_yes_percentage = yes_count / (yes_count + no_count)
                if current_yes_percentage >= target_yes_percentage and no_count < total_buses * (1 - target_yes_percentage):
                    current_mode = False
            else:  # current_mode == False
                no_count += 1
                # Check if we should switch back to "yes"
                current_yes_percentage = yes_count / (yes_count + no_count)
                if current_yes_percentage < target_yes_percentage:
                    current_mode = True

        return dict(agents_by_bus)

    def build_agent_data_list(self, bus, agents):
        "Convert agent objects to data dictionaries for the optimizer."
        agent_data_list = []

        for agent in agents:
            if agent.flex == 0:  # Unflexible load with power profile
                agent_data = {
                'flex': 0, 
                'unique_id': agent.unique_id, 
                'bus': agent.bus,
                'power_profile': agent.a_power_profile
            }
            
            elif agent.flex == 1:  # Battery
                agent_data = {
                    'flex': 1, 
                    'unique_id': agent.unique_id, 
                    'bus': agent.bus, 
                    'max_power': agent.max_power, 
                    'efficiency': agent.efficiency, 
                    'capacity': agent.capacity, 
                    'max_prognosis': agent.max_prognosis
                }
                
            elif agent.flex == 2:  # EV with profile data
                agent_data = {
                    'flex': 2, 
                    'unique_id': agent.unique_id, 
                    'bus': agent.bus,
                    'max_capacity': agent.max_capacity,
                    'efficiency': agent.efficiency,
                    'home_base_loading_power': agent.home_base_loading_power,
                    'work_loading_power': agent.work_loading_power,
                    'consumption_1km': agent.consumption_1km,
                    'max_prognosis': agent.max_prognosis,
                    'profile_df': agent.profile_df,
                }
                
            elif agent.flex == 3:  # Heat pump with full physics parameters
                agent_data = {
                    'flex': 3, 
                    'unique_id': agent.unique_id, 
                    'bus': agent.bus,
                    'P_max_cap': agent.P_max_cap,
                    'cop': agent.cop,
                    'R': agent.R,
                    'C_adapted': agent.C_adapted,
                    'Q_dot_max': agent.Q_dot_max,
                    'T_min': agent.T_min,
                    'T_max': agent.T_max,
                    'T_set': agent.T_set,
                    'max_prognosis': agent.max_prognosis
                }
            else:
                # Skip agents with flex values not in [0, 1, 2, 3]
                continue
                
            agent_data_list.append(agent_data)
        
        return agent_data_list
    


    def step(self):
        """Advance the model by one step."""
        self.stepcount+=1
        self.current_date += self.timestep 
        self.temperature=float(self.temperature_df.loc[self.temperature_df["time"]==self.current_date].iloc[0,1])
        [a.update_status() for a in self.agents if a.flex in [1,2,3]] 
        self.market_price=[agents.energy_price for agents in self.agents if agents.flex==999][0]
        if self.stepcount==0:
            self.hn_optimizers={}
            for bus, agents in self.HEM_dict.items():
                # Skip if no agents or no flexible agents for this bus
                if not agents or all(agent.flex not in [1, 2, 3] for agent in agents):
                    continue
                
                agent_data_list = self.build_agent_data_list(bus, agents)
                
                # Skip if agent_data_list is empty (no valid agents)
                if not agent_data_list:
                    continue
                max_prognosis_values = []
                for agent in agents:
                    if hasattr(agent, 'max_prognosis'):
                        max_prognosis_values.append(agent.max_prognosis)
                
                max_prognosis = min(min(max_prognosis_values),45) if max_prognosis_values else 0
                horizon=max(max_prognosis_values) if max_prognosis_values else 0
                max_prognosis=min(horizon-5,max_prognosis)
                model_params = {
                    'bus': bus, 
                    'horizon': horizon,
                    'total_steps': self.total_steps,
                    'timestep': self.timestep, 
                    'sref': self.sref,
                    'gridfee_ext': self.gridfee_ext, 
                    'levies_ext': self.levies_ext,
                    'gridfee_LEC': self.gridfee_LEC,
                    'levies_LEC': self.levies_LEC,
                    'LEC_participation': agents[0].LEC_participation  # Assuming all agents on the same bus have the same LEC participation
                }
                external_data = {
                    'margin_buy': self.market_price_margin_buy,
                    'margin_sell': self.market_price_margin_sell,
                    'margin_charge': self.market_price_margin_charge,
                    'temperature_df': self.temperature_df,
                    'energy_price_df': self.market_price,
                    'solver': self.solver
                }

                # Store optimizer data
                self.hn_optimizers[bus] = {
                    'agent_data_list': agent_data_list,
                    "model_params": model_params,
                    "external_data": external_data,
                    'max_prognosis': max_prognosis
                }
                
                # Only create optimizer instance if NOT using MPI (local fallback)

                hn_optimizer = HNOptimizer(agent_data_list, model_params, external_data)
                self.hn_optimizers[bus]['optimizer'] = hn_optimizer  
                            
            # Split hn_optimizers into batches for workers
            
                    
        # Build variable_data for all buses that need optimization this round
        # and organize by worker for efficient sending
        agent_results = {}
        
        for bus, optimizer_data in self.hn_optimizers.items():
            if self.stepcount % optimizer_data["max_prognosis"] == 0:
                # Build agent states for this bus
                agent_states = {}
                for agent in self.HEM_dict[bus]:
                    if agent.flex == 0:  # Unflexible load
                        agent_states[agent.unique_id] = {}
                    elif agent.flex == 1:  # Battery
                        agent_states[agent.unique_id] = {'soc': getattr(agent, 'soc')}
                    elif agent.flex == 2:  # EV
                        agent_states[agent.unique_id] = {'soc': getattr(agent, 'SOC')}
                    elif agent.flex == 3:  # Heat pump
                        agent_states[agent.unique_id] = {'temperature': getattr(agent, 'T_in')}
                    else:
                        # Skip other agent types (like ext_grid)
                        continue

                variable_data = {
                    'current_date': self.current_date,
                    'stepcount': self.stepcount,
                    'agent_states': agent_states
                }
                optimizer_data["variable_data"] = variable_data
                result = optimizer_data["optimizer"].optimize(variable_data)
                agent_results.update(result)

        # Update agents with optimization results
        for agent in self.agents:
            if agent.unique_id in agent_results:
                results = agent_results[agent.unique_id]
                for key, value in results.items():
                    setattr(agent, key, value)
                setattr(agent, "updated", 0)

        self.agents.do("step") 
        if self.stepcount==0:
            market_static, bids_static, asks_static=self.initialize_static_data()
            self.market_optimizer=MarketOptimizer(market_static, bids_static, asks_static, self.grid, self.solver)

        new_bid_data, new_ask_data, new_market_variable, new_non_LEC_data=self.get_round_data()  
        self.market_optimizer.compute_allocation(new_bid_data, new_ask_data)
        self.data=self.market_optimizer.process_pyomo_results(new_market_variable, new_non_LEC_data)
 
        self.results[self.stepcount]=self.data 
        if len(self.results)>10:
            del self.results[min(self.results)] 



from data.config.config import config
model = LEM(config.main["sb_grid"],config.main["simulation_start_time"])


print("\n" + "="*60)
print("STORAGE AGENTS SUMMARY")
print("="*60)
for agent in model.agents:
    if agent.typ == "storage":
        print(f"  Storage ID: {agent.unique_id}, Bus: {agent.bus}, Method: {agent.method}, SOC: {agent.soc:.2f}")
print("="*60 + "\n")

del config
gc.collect()

     
        
        
        
        
