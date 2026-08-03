# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:36:13 2023

@author: mjulschm
"""
import sys
import mesa
import random
from datetime import timedelta,datetime
from data.config.config import config
from collections import deque  
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import gc
from mesa_model.sac import SACLearner
from mesa_model.storage_logic import (
    state_features, mark_to_market_reward,
    ACTION_DEADBAND, ASK_PRICE_FLOOR, STATE_DIM, FORECAST_STEPS,
)

def initialize(self, typ, factor, cosphi, bus):
    self.typ=typ
    curve=self.model.grid.load.profile[self.model.grid.load.agent_id==self.unique_id].values[0]+"_pload"
    self.a_power_profile=pd.DataFrame({"time":pd.to_datetime(config.load_profile["time"],format='%d.%m.%Y %H:%M',  errors='coerce'),"power":config.load_profile[curve].astype(float)*factor/self.model.sref})
    # O(1) per-step lookup (the DataFrame stays: the HN optimizer consumes it)
    self._power_by_time=dict(zip(self.a_power_profile["time"], self.a_power_profile["power"]))
    self.bus=bus
    self.profile=curve
    self.cosphi=float(cosphi)
    self.flex=0
    self.price=0
    self.ask=0 #only buys energy
    self.type="lin"
    self.LEC_participation=True
def provide_a_power_range(self):
    try:
        power=self._power_by_time.get(self.model.current_date)
        power=0 if power is None or pd.isna(power) else abs(power)
        return [power, power]
    except Exception as e:
        print(self.model.current_date,";", "agent_id:", self.unique_id,";", "Error when reading load_profile", str(e))
        return[0,0]

def _fit_quadratic_three_points(x1, y1, x2, y2, tol=1e-12):
    """
    Fit a quadratic y = a x^2 + b x + c through (0,0), (x1,y1), (x2,y2).
    Returns (a,b,c) or raises LinAlgError if singular.
    """
    # Fast path: all zero
    if (abs(x1) < tol and abs(y1) < tol and abs(x2) < tol and abs(y2) < tol):
        return 0.0, 0.0, 0.0

    # Build system for points (0,0), (x1,y1), (x2,y2)
    X = np.array([
        [0.0**2, 0.0, 1.0],
        [x1**2,  x1,  1.0],
        [x2**2,  x2,  1.0]
    ], dtype=float)
    Y = np.array([0.0, y1, y2], dtype=float)

    # Solve (may raise LinAlgError if singular)
    a, b, c = np.linalg.solve(X, Y)
    # Clean tiny numerical noise
    a = 0.0 if abs(a) < tol else a
    b = 0.0 if abs(b) < tol else b
    c = 0.0 if abs(c) < tol else c
    return a, b, c

def fit_quadratic_concave(x1, y1, x2, y2, tol=1e-12):
    """
    Fit concave (a <= 0) quadratic through the three points.
    On error or wrong curvature, returns (zero_function, [0,0,0]).
    Affine (a≈0) is accepted.
    """
    try:
        a, b, c = _fit_quadratic_three_points(x1, y1, x2, y2, tol=tol)
    except np.linalg.LinAlgError:
        return (lambda v: 0*v, [0.0, 0.0, 0.0])

    # Curvature check: concave requires a <= 0 (allow tiny positive due to noise)
    if a > tol:
        # wrong curvature → fallback to zero
        return (lambda v: 0*v, [0.0, 0.0, 0.0])

    return _make_callable(a, b, c), [a, b, c]

def fit_quadratic_convex(x1, y1, x2, y2, tol=1e-12):
    """
    Fit convex (a >= 0) quadratic through the three points.
    On error or wrong curvature, returns (zero_function, [0,0,0]).
    Affine (a≈0) is accepted.
    """
    try:
        a, b, c = _fit_quadratic_three_points(x1, y1, x2, y2, tol=tol)
    except np.linalg.LinAlgError:
        return (lambda v: 0*v, [0.0, 0.0, 0.0])

    # Curvature check: convex requires a >= 0 (allow tiny negative due to noise)
    if a < -tol:
        # wrong curvature → fallback to zero, as requested
        return (lambda v: 0*v, [0.0, 0.0, 0.0])

    return _make_callable(a, b, c), [a, b, c]

def _make_callable(a, b, c):
    def expr(var):
        return a * var * var + b * var + c
    return expr

def fit_function_buy(agent, x2, y1, y2):
            x2=x2*(100/agent.model.sref)
            x1=0.5*x2
            y1=y1*x1
            y2=y2*x2
            function, coefs =fit_quadratic_concave(x1, y1, x2, y2)
            return function, coefs 
      
def fit_function_sell(agent, x2, y1, y2):
            x2=x2*(100/agent.model.sref)
            x1=0.5*x2
            y1=y1*x1
            y2=y2*x2
            function, coefs =fit_quadratic_convex(x1, y1, x2, y2)
            return function, coefs 
 #----------------------------------------------------------------------------
class res(mesa.Agent):
    "local renewable energy supply agent."
    def __init__(self, model, typ, factor, curve, bus):
        super().__init__(model)
        factor=float(factor)
        self.a_power_profile=pd.DataFrame({"time":pd.to_datetime(config.res_profile["time"],format='%d.%m.%Y %H:%M',  errors='coerce'),"power":config.res_profile[curve].astype(float)*factor*(-1)/self.model.sref})
        self._power_by_time=dict(zip(self.a_power_profile["time"], self.a_power_profile["power"]))
        self.price=0
        self.bus=bus
        self.profile=curve
        self.cosphi=1
        self.flex=0
        self.bid=0 #only sells energy
        self.type="lin"
        self.typ=typ
        self.LEC_participation=True
        
    
    def utility_function(self, a_power):
        return self.price*a_power*(self.model.timestep.seconds/(60*60))
   
    def ask_function(self, a_power):
        return self.price*a_power*(self.model.timestep.seconds/(60*60))

    def step(self):
       power=provide_a_power_range(self)
       min_power=power[0]
       max_power=power[1]
       self.coefficients_ask=[0, self.price*(self.model.timestep.seconds/(60*60))]
       self.ask=[min_power,max_power,self.ask_function, "lin"]
           
       

#----------------------------------------------------------------------------        
class ext_grid(mesa.Agent):
   "Ext-Grid-Connection Agent."
   
   def __init__(self, model):
        super().__init__(model)
        self.energy_price=pd.DataFrame({"time":pd.to_datetime(config.spot_price["time"],format='%d.%m.%Y %H:%M',  errors='coerce'),"price":config.spot_price["price (Ct/kWh)"].astype(float)/(100/model.sref)})
        self.energy=10000/model.sref
        self.bus=model.slack
        self.margin_buy=config.ext_grid["margin_buy"]/(100/model.sref)
        self.margin_sell=config.ext_grid["margin_sell"]/(100/model.sref)
        self.cosphi=1
        self.flex=999
        self.type="lin"
        self.typ="ext_grid"
        self.LEC_participation=True
        
   def bid_function(self, a_power):
       price=self.price
       return (price-self.margin_sell)*a_power*(self.model.timestep.seconds/(60*60))
   
   def ask_function(self, a_power):
        price=self.price
        return (price+self.margin_buy)*a_power*(self.model.timestep.seconds/(60*60))
   
   
   def step(self):
        price=self.energy_price.loc[self.energy_price["time"]==self.model.current_date,"price"]
        if price.empty:
            # A data gap must not silently clear the market at a free grid.
            print(f"WARNING [ext_grid]: no spot price for {self.model.current_date} "
                  f"- grid disabled this step", flush=True)
            self.price=0
            self.energy=0
        else:
            self.price=price.values[0]
            self.energy=10000/self.model.sref   # restore capacity after any gap
        self.coefficients_bid=[0,(self.price -self.margin_sell)*(self.model.timestep.seconds/(60*60))]
        self.coefficients_ask=[0,(self.price +self.margin_buy)*(self.model.timestep.seconds/(60*60))]
        self.bid=[0,self.energy,self.bid_function, "lin"]
        self.ask=[0,self.energy,self.ask_function, "lin"]

       
       
#----------------------------------------------------------------------------
class household(mesa.Agent):
   "Household-Agent."
   def __init__(self, model, typ, factor, cosphi, bus):
        super().__init__(model)
        initialize(self, typ, factor, cosphi, bus)
   
   def utility_function(self, a_power):
       return self.price*a_power

   def bid_function(self, a_power):
       return self.price*a_power*(self.model.timestep.seconds/(60*60))
   
   def step(self):
       power=provide_a_power_range(self)
       min_power=power[0]
       max_power=power[1]
       self.coefficients_bid=[0,self.price*(self.model.timestep.seconds/(60*60))]
       self.bid=[min_power,max_power,self.bid_function, "lin"]
          
       
#----------------------------------------------------------------------------       
class industry(mesa.Agent):
    "Industrial-Agent."
    def __init__(self, model, typ, factor, cosphi, bus):
        super().__init__(model)
        initialize(self, typ, factor, cosphi, bus)

   
    def utility_function(self, a_power):
        return self.price*a_power*(self.model.timestep.seconds/(60*60))

    def bid_function(self, a_power):
       a=self.utility_function(a_power)
       return self.utility_function(a_power)
    
    def step(self):
       power=provide_a_power_range(self)
       min_power=power[0]
       max_power=power[1]
       self.coefficients_bid=[0,self.price*(self.model.timestep.seconds/(60*60))]
       self.bid=[min_power,max_power,self.bid_function, "lin"]


     
       
#----------------------------------------------------------------------------       

class heatpump(mesa.Agent):
    def __init__(self, model, R, C, power, size, cosphi, bus, method):
        super().__init__(model)
        self.R=R # in [K/kW]
        self.C=C #in [Wh/(m^2*K)]
        self.area=size
        self.C_adapted=self.C*self.area/1000 #100m^2, 1000W/kW; -> Einheit von C in kWh/K)
        self.cop=2.6
        self.T_set=20
        self.T_in=20
        self.T_min=19
        self.T_max=24
        self.F=0.95 #ein Faktor für weniger geheizte Räume
        self.P_max_cap=power
        self.Q_dot_max= self.P_max_cap*self.cop
        self.price=10/(100/self.model.sref)
        self.bid=[0,0,0, "lin"]
        self.bus=bus
        self.cosphi=cosphi
        self.soc=0.5
        self.flex=3
        self.method=method
        self.updated=28
        self.max_prognosis=29
        self.max_buy_price=0
        self.risk_aversion=[1.05,1.0]
        self.ask=0 #only buys energy
        self.type="quad"
        self.typ="heatpump"
        self.optimal_power_buy=0
        self.LEC_participation=True
        
    def forecast_max(self):
            T_amb = float(self.model.temperature_df[self.model.temperature_df.loc[:,"time"]==self.model.current_date+self.model.timestep].values[0][1])
            T_in=self.T_in
            i=1
            Q_sum=0
            while T_in<(self.T_max)*0.9: #Loop solange Heizung möglich ist:
                        Q_dot=self.Q_dot_max #Heizleistung pro Minute
                        dT_indt=((1/(self.R*self.C_adapted)*(T_amb-T_in)+1/self.C_adapted*Q_dot))/60
                        T_in=T_in+dT_indt
                        Q_sum+=Q_dot/60
                        if T_in>=self.T_max:
                            break
                        if i==15:
                            break
                        i=i+1
            while i<15:
                Q_dot=max(min((self.T_max-T_amb)*1/self.R, self.Q_dot_max),0) #Heizleistung, so dass T_max gehalten wird
                Q_sum+=Q_dot/60 #Gesamt-Wärmeenergiemenge
                i=i+1
            P_max=Q_sum/self.cop*(60*60)/self.model.timestep.seconds
            return(P_max)
    def forecast_min(self):
            T_amb = float(self.model.temperature_df[self.model.temperature_df.loc[:,"time"]==self.model.current_date+self.model.timestep].values[0][1])
            T_in=self.T_in
            i=1
            Q_sum=0
            while T_in>=self.T_min: #Loop solange keine Heizung ntwendig ist
                        dT_indt=(1/(self.R*self.C_adapted)*(T_amb-T_in))/60 #gibt Temperaturunterschied pro Minute             
                        T_in=T_in+dT_indt
                        if i==15:
                            Q_dot=0
                            break
                        i=i+1
            while i<15:
                Q_dot=max(min((self.T_min-T_in)*self.C_adapted*60-1/self.R*(T_amb-T_in), self.Q_dot_max),0) #Heizleistung pro Minute
                dT_indt=((1/(self.R*self.C_adapted)*(T_amb-T_in)+1/self.C_adapted*Q_dot))/60 #gibt Temperaturunterschied pro Minute             
                T_in+=dT_indt
                Q_sum+=Q_dot/60 #Gesamt-Wärmeenergiemenge
                i=i+1
            P_min=Q_sum/self.cop*(60*60)/self.model.timestep.seconds
            return(P_min)
    
    def update_status(self):
            energy=0
            if len(self.model.results)!=0:
                try:
                    result=self.model.results[int(self.model.stepcount-1)]["agents"]
                except Exception:
                    result={}
                if isinstance(result, pd.DataFrame):
                    result=result[result["Agent ID"]==self.unique_id]
                    if len(result)>0:
                        energy=np.abs(result["Energy bought [kWh]"]).values[0]

            T_amb = float(self.model.temperature_df[self.model.temperature_df.loc[:,"time"]==self.model.current_date].values[0][1])
            Q_dot=np.abs(energy/self.model.timestep.seconds*(60*60)*self.cop)
            T_in=self.T_in
            i=0
            while i<self.model.timestep.total_seconds()/60: #in Minuten-Schritten
                        dT_indt=((1/(self.R*self.C_adapted)*(T_amb-T_in)+1/self.C_adapted*Q_dot))/60 #gibt Temperaturunterschied pro Minute             
                        T_in+=dT_indt
                        i=i+1
            self.T_in=T_in
            if self.T_in<self.T_min:
                self.T_in=self.T_min
                print("Heating below Tmin!")
            if self.T_in>self.T_max:
                self.T_in=self.T_max
                print("Heating above Tmax!")
            self.soc=(self.T_in-self.T_min)/(self.T_max-self.T_min)

    def step(self):
            self.bid=[0,0,0, "lin"]
            p_max=self.forecast_max()*0.8
            p_min=self.forecast_min()*1.2
            if self.LEC_participation==True:
                if self.method=="optimisation":
                    #if self.max_prognosis< self.updated: 
                        #optimize(self) 
                    buy_price_1=abs(self.max_buy_price[self.updated]*self.risk_aversion[0])*(self.model.timestep.seconds/(60*60))
                    buy_price_2=abs(self.max_buy_price[self.updated]*self.risk_aversion[1])*(self.model.timestep.seconds/(60*60))           
                    self.bid_function, self.coefficients_bid =fit_function_buy(self, p_max/self.model.sref,buy_price_1, buy_price_2)
                    if self.bid_function(1)==0:
                        p_max=0
                    self.bid=[0, p_max/self.model.sref, self.bid_function, "quad"] 
                    self.updated+=1
                if self.method=="Learning":
                    self.bid_function, self.coefficients =fit_function_buy(self, 0,0, 0)
                    self.bid=[0,0,self.bid_function, "quad"]
            if self.LEC_participation==False:
                self.optimal_power_buy_current = np.clip(self.optimal_power_buy[self.updated], p_min/self.model.sref, p_max/self.model.sref)
                self.updated+=1

                

 #----------------------------------------------------------------------------           
class farm(mesa.Agent):
    "Farm-Agent."
    def __init__(self, model, typ, factor, cosphi, bus):
        super().__init__(model)
        initialize(self,typ, factor, cosphi, bus)
 
    def utility_function(self, a_power):
         return self.price*a_power

    def bid_function(self, a_power):
        a=self.price*a_power*(self.model.timestep.seconds/(60*60))
        return a
    
    def step(self):
       power=provide_a_power_range(self)
       min_power=power[0]
       max_power=power[1]
       self.coefficients_bid=[0,self.price*(self.model.timestep.seconds/(60*60))]
       self.bid=[min_power,max_power,self.bid_function, "lin"]
    
 
#----------------------------------------------------------------------------
class storage(mesa.Agent):
    """Storage Agent with RL learning capability."""
    
    def __init__(self, model, capacity, power, node, efficiency, discharge, method):
        super().__init__(model)
        self.capacity = capacity
        timestep_frac = self.model.timestep.seconds / 86400.0
        self.discharge = discharge ** timestep_frac
        self.efficiency = efficiency
        self.max_power = power
        self.bus = node
        self.price = 3 / (100 / self.model.sref)
        self.margin = 2 / (100 / self.model.sref)
        self.soc = config.storage["SOC_start"]
        self.capital = 0
        self.cosphi = 1
        self.flex = 1
        self.method = method
        self.updated = 49
        self.max_prognosis = 48
        self.max_buy_price = 0
        self.min_sell_price = np.inf
        self.risk_aversion = [0.98, 0.95]
        self.max_power_charge = 0
        self.max_power_discharge = 0
        self.type = "lin"
        self.typ = "storage"
        self.optimal_power_buy = 0
        self.optimal_power_sell = 0
        self.LEC_participation = True
        
        # Initialize bid/ask and coefficients
        self.ask = 0
        self.bid = 0
        self.coefficients_ask = [0, 0]
        self.coefficients_bid = [0, 0]
        
        # ---- Soft Actor-Critic (SAC) learning setup ----
        if self.method == "learning":
            self._setup_learning()

    def _setup_learning(self):
        """Initialise SAC learner + bookkeeping for the learning method."""
        # 20-D state (see storage_logic.state_features)
        self.state_dim = STATE_DIM
        self.gamma = 0.996          # half-life ≈ 1.8 days: cross-day arbitrage visible
        self.reward_scale = 10.0    # lift tiny per-step € rewards above the entropy term

        self.learner = SACLearner(
            state_dim=self.state_dim,
            gamma=self.gamma,
            tau=0.005,
            lr=3e-4,
            target_entropy=-1.0,
            buffer_size=100_000,
            batch_size=256,
            warmup_steps=500,     # sim is ~8.5k steps total; keep warm-up a small fraction
            actor_update_every=2,
            updates_per_step=4,   # UTD ratio: sim (not SAC) is the bottleneck, so extra
                                  # gradient updates per step are nearly free and 4× the learning
            reward_scale=self.reward_scale,
            alpha_min=0.05,       # floor on entropy temperature — keeps exploration
                                  # alive (auto-α otherwise collapses to ~0 prematurely)
            seed=42,
        )

        # Annealed SOC-band shaping. A soft pull toward the operating band that
        # decays to zero over training, so it guides the policy out of the
        # drain/hoard corners early but leaves the *final* policy unbiased
        # (potential-based inventory term remains the only persistent shaping).
        self.soc_shaping_weight = 1.0
        self.soc_shaping_anneal = 100_000.0   # env steps over which it fades to 0
        self.current_epoch = 0                # set by the multi-epoch loop in main.py

        # Optional: load a policy pre-trained on the surrogate env, and/or run
        # frozen (deterministic, no further learning) for evaluation in the
        # full market. See train_surrogate.py.
        import os
        self.eval_mode = os.environ.get("SAC_EVAL", "0") == "1"
        _load_path = os.environ.get("SAC_LOAD_POLICY")
        if _load_path and os.path.exists(_load_path):
            self.learner.load(_load_path)
            print(f"[Storage] Loaded SAC policy from {_load_path} "
                  f"(eval_mode={self.eval_mode})", flush=True)

        # Transition assembly across step() → update_status()
        self.last_state = None
        self.last_action = None
        self.last_decision_price = None   # price the agent saw when it acted (settlement price)
        self.last_soc = None              # SOC going into the acted step
        self.last_override_active = False # safety override fired → exclude from replay

        # Logging cadence (SAC learns every step; this only groups the logs)
        self.update_frequency = 96        # 1 day
        self.episode_counter = 0
        self.episode_profits = []
        self.cumulative_reward = 0.0
        self.cumulative_profit = 0.0      # realised cash (€)
        self.cumulative_bought = 0.0
        self.cumulative_sold = 0.0
        self.last_diagnostics = {}

        # SOC operating range — soft band left to the policy; hard limits in §7
        self.soc_floor = 0.20
        self.soc_ceiling = 0.85
        self.soc_target = 0.50
        self.soc_history = []

        # Price caches / observed-history buffers (no future leakage)
        self._price_index = None
        self._price_array = None
        self._price_cache_ready = False
        self.price_history = deque(maxlen=96)          # last 24 h of observed prices
        self._price_lag_1h = deque(maxlen=4)           # 4 × 15 min = 1 h
        self._price_lag_4h = deque(maxlen=16)          # 16 × 15 min = 4 h
        self.hourly_price_ewma = {}                    # causal hour-of-day baseline
        self.hourly_ewma_beta = 0.05

        # Trade tracking for episode logging
        self.trade_count_buy = 0
        self.trade_count_sell = 0
        self.buy_price_sum = 0.0
        self.sell_price_sum = 0.0

    def provide_a_power(self):
        a_power_discharge = min(
            max(self.soc - 0.05, 0) * self.capacity * 60 * 60 / self.model.timestep.seconds,
            self.max_power
        ) / self.model.sref * self.efficiency
        
        a_power_charge = min(
            max((0.95 - self.soc), 0) * self.capacity * 60 * 60 / self.model.timestep.seconds,
            self.max_power
        ) / self.model.sref / self.efficiency
        
        return [a_power_discharge, a_power_charge]

    def _initialize_price_cache(self):
        try:
            price_df = self.model.market_price.sort_values("time")
            # Create dict for O(1) time-based lookup
            self._price_index = dict(zip(price_df["time"], price_df["price"]))
            # Array + time->position map for the known-future DA price features
            self._price_array = price_df["price"].to_numpy(dtype=float)
            self._price_pos = {t: i for i, t in enumerate(price_df["time"])}
            self._price_cache_ready = True
        except (AttributeError, TypeError, KeyError):
            self._price_cache_ready = False

    def get_future_prices(self, k=FORECAST_STEPS):
        if not self._price_cache_ready:
            self._initialize_price_cache()
        if not self._price_cache_ready:
            return []
        pos = self._price_pos.get(self.model.current_date)
        if pos is None:
            return []
        return self._price_array[pos + 1: pos + 1 + k]
    
    def get_current_price(self):
        if not self._price_cache_ready:
            self._initialize_price_cache()
        
        if self._price_cache_ready and self._price_index:
            price = self._price_index.get(self.model.current_date)
            if price is not None:
                return float(price)
        
        try:
            price_df = self.model.market_price
            price = price_df[price_df["time"] == self.model.current_date]["price"]
            if not price.empty:
                return float(price.values[0])
        except (AttributeError, TypeError, KeyError):
            pass
        return 30.0

    def build_state(self):
        t = self.model.current_date
        return state_features(
            soc=self.soc, soc_floor=self.soc_floor, soc_ceiling=self.soc_ceiling,
            price=self.get_current_price(), price_history=self.price_history,
            lag_1h=self._price_lag_1h, lag_4h=self._price_lag_4h,
            hourly_ewma=self.hourly_price_ewma,
            hour=t.hour, weekday=t.weekday(), month=t.month,
            margin_buy=getattr(self.model, "market_price_margin_buy", 1.0),
            margin_sell=getattr(self.model, "market_price_margin_sell", 0.3),
            future_prices=self.get_future_prices(),
        )

    def action_to_bid(self, action):

        self.ask = [0, 0, self.offer_function(0), "lin"]
        self.bid = [0, 0, self.offer_function(0), "lin"]
        self.coefficients_ask = [0, 0]
        self.coefficients_bid = [0, 0]

        max_discharge, max_charge = self.provide_a_power()
        eps = 1e-6
        p = self.get_current_price()
        margin_buy = getattr(self.model, "market_price_margin_buy", 1.0)
        margin_sell = getattr(self.model, "market_price_margin_sell", 0.3)
        # External-buy fee adder in the LP objective (ct/kWh) that the bid
        # must also cross to clear against the grid.
        fee_ext = (getattr(self.model, "gridfee_ext", 11.0)
                   + getattr(self.model, "levies_ext", 4.7))

        # --- Emergency charge: SOC critically low ---
        if self.soc < 0.10:
            if max_charge > eps:
                bid_price = 1000.0
                self.bid = [max_charge, max_charge, self.offer_function(bid_price), "lin"]
                self.coefficients_bid = [0, bid_price * (self.model.timestep.seconds / 3600)]
            return

        # --- Emergency discharge: SOC critically high ---
        if self.soc > 0.95:
            if max_discharge > eps:
                ask_price = ASK_PRICE_FLOOR
                self.ask = [max_discharge, max_discharge, self.offer_function(ask_price), "lin"]
                self.coefficients_ask = [0, ask_price * (self.model.timestep.seconds / 3600)]
            return

        if action > ACTION_DEADBAND:        # charge
            power = min(action * max_charge, max_charge)
            if power > eps:
                bid_price = p + margin_buy + fee_ext + 0.01
                self.bid = [0, power, self.offer_function(bid_price), "lin"]
                self.coefficients_bid = [0, bid_price * (self.model.timestep.seconds / 3600)]

        elif action < -ACTION_DEADBAND:     # discharge
            power = min(-action * max_discharge, max_discharge)
            ask_price = p - margin_sell - 0.01
            if power > eps and ask_price >= ASK_PRICE_FLOOR:
                self.ask = [0, power, self.offer_function(ask_price), "lin"]
                self.coefficients_ask = [0, ask_price * (self.model.timestep.seconds / 3600)]

    def compute_reward(self, bought, sold, p_buy, p_sell, p_decision, p_now,
                       soc_old, soc_new):
        """Mark-to-market wealth change in € (see storage_logic).

        Cash is valued at the prices actually settled (p_buy / p_sell from the
        clearing result); the inventory term is γ-corrected potential shaping
        at liquidation value. Storage is fee-exempt, so no gridfee/levies.

        Returned raw (in €); the SACLearner applies reward_scale.
        """
        margin_sell = getattr(self.model, "market_price_margin_sell", 0.3)
        return mark_to_market_reward(
            bought=bought, sold=sold, p_buy=p_buy, p_sell=p_sell,
            p_decision=p_decision, p_now=p_now,
            soc_old=soc_old, soc_new=soc_new, capacity=self.capacity,
            efficiency=self.efficiency, margin_sell=margin_sell,
            gamma=self.gamma,
            soc_floor=self.soc_floor, soc_ceiling=self.soc_ceiling,
            shaping_weight=self.soc_shaping_weight,
            total_env_steps=self.learner.total_env_steps,
            shaping_anneal=self.soc_shaping_anneal,
        )

    def update_status(self):
        """Apply the settled trade to SOC and run one SAC learning step.

        Called at time τ+1. Reads the trade that cleared at τ, updates SOC,
        builds the (s_τ, a_τ, r_τ, s_{τ+1}) transition with a leakage-free
        mark-to-market reward, pushes it to replay, and runs one SAC update.
        """
        if len(self.model.results) == 0:
            return

        try:
            step_result = self.model.results[int(self.model.stepcount - 1)]
            result = step_result["agents"]
            result = result[result["Agent ID"] == self.unique_id]
        except (KeyError, IndexError):
            return

        if len(result) == 0:
            return

        bought = float(result["Energy bought [kWh]"].to_numpy()[0])
        sold = float(result["Energy sold [kWh]"].to_numpy()[0])

        # Uniform settlement price of the cleared step (€/kWh -> ct/kWh). Every
        # agent settles at this slack price; falls back to the decision price.
        try:
            p_settle = float(step_result["Input_Grid"]["Market Price [€/kWh]"].to_numpy()[0]) * 100.0
            if np.isnan(p_settle):
                p_settle = None
        except (KeyError, IndexError, TypeError, ValueError):
            p_settle = None

        old_soc = self.soc
        energy_delta = bought * self.efficiency - sold / self.efficiency
        soc_delta = energy_delta / self.capacity
        self.soc = min(max((old_soc * self.discharge) + soc_delta, 0), 1)

        if self.method != "learning":
            return

        new_soc = self.soc
        p_now = self.get_current_price()           # price at τ+1 (now observed)
        p_decision = (self.last_decision_price
                      if self.last_decision_price is not None else p_now)

        # Update observed-price buffers (causal) BEFORE building next_state
        self.price_history.append(p_now)
        self._price_lag_1h.append(p_now)
        self._price_lag_4h.append(p_now)
        h = self.model.current_date.hour
        self.hourly_price_ewma[h] = (
            (1 - self.hourly_ewma_beta) * self.hourly_price_ewma.get(h, p_now)
            + self.hourly_ewma_beta * p_now
        )

        self.soc_history.append(new_soc)
        if len(self.soc_history) > 2000:
            self.soc_history = self.soc_history[-2000:]

        self.cumulative_bought += bought
        self.cumulative_sold += sold

        if self.last_state is not None and self.last_action is not None:
            # Settled prices for the cash leg; fall back to the surrogate's
            # price±margin model when the slack price is unavailable.
            margin_buy = getattr(self.model, "market_price_margin_buy", 1.0)
            margin_sell = getattr(self.model, "market_price_margin_sell", 0.3)
            p_buy = p_settle if p_settle is not None else p_decision + margin_buy
            p_sell = p_settle if p_settle is not None else p_decision - margin_sell

            reward = self.compute_reward(bought, sold, p_buy, p_sell,
                                         p_decision, p_now, old_soc, new_soc)
            self.cumulative_reward += reward
            self.cumulative_profit += (sold * p_sell - bought * p_buy) / 100.0  # realised cash (€)

            if bought > 0.01:
                self.trade_count_buy += 1
                self.buy_price_sum += p_buy
            if sold > 0.01:
                self.trade_count_sell += 1
                self.sell_price_sum += p_sell

            next_state = self.build_state()

            # Push every transition, including forced-override steps. SAC's actor
            # re-samples its own actions, so it is never trained toward the forced
            # action; the critic, however, learns from (s, forced_a, r, s') that
            # draining to the floor triggers a costly recharge — the exact signal
            # that teaches the agent low SOC is bad. Excluding them hid that cost.
            if not self.eval_mode:
                self.learner.push(self.last_state, self.last_action, reward, next_state, 0.0)

                diag = self.learner.learn()
                if diag:
                    # merge so actor-only fields (entropy) persist across critic-only steps
                    self.last_diagnostics.update(diag)

            if self.model.stepcount % self.update_frequency == 0:
                self.episode_counter += 1
                self._print_episode_summary()
                self._log_episode()
                self._reset_episode_counters()

    def _print_episode_summary(self):
        recent = self.soc_history[-96:] if len(self.soc_history) >= 96 else self.soc_history
        avg_soc = float(np.mean(recent)) if recent else self.soc
        min_soc = float(np.min(recent)) if recent else self.soc
        max_soc = float(np.max(recent)) if recent else self.soc
        d = self.last_diagnostics

        print(f"\n{'='*60}")
        print(f"[Storage {self.unique_id}] Episode {self.episode_counter} (SAC)")
        print(f"{'='*60}")
        print(f"  Cumulative Reward:   {self.cumulative_reward:>12.2f}")
        print(f"  Actual Profit (€):   {self.cumulative_profit:>12.4f}")
        print(f"  Energy Bought (kWh): {self.cumulative_bought:>12.2f}")
        print(f"  Energy Sold (kWh):   {self.cumulative_sold:>12.2f}")
        print(f"  Net Energy (kWh):    {self.cumulative_bought - self.cumulative_sold:>12.2f}")
        print(f"  Buffer Size:         {len(self.learner.buffer):>12d}")
        print(f"  Alpha (entropy):     {self.learner.alpha.item():>12.4f}")
        print(f"  Current SOC:         {self.soc*100:>12.1f}%")
        print(f"  SOC Range (24h):     {min_soc*100:>6.1f}% - {max_soc*100:.1f}%")
        print(f"  Avg SOC (24h):       {avg_soc*100:>12.1f}%")
        if d:
            print(f"  Critic Loss:         {d.get('critic_loss', float('nan')):>12.4f}")
            print(f"  Entropy:             {d.get('entropy', float('nan')):>12.4f}")
        print(f"{'='*60}", flush=True)

    def _log_episode(self):
        import json, os
        os.makedirs("output/sac", exist_ok=True)
        recent = self.soc_history[-96:] if len(self.soc_history) >= 96 else self.soc_history
        soc_arr = np.array(recent) if recent else np.array([self.soc])
        avg_buy_price = self.buy_price_sum / self.trade_count_buy if self.trade_count_buy > 0 else 0.0
        avg_sell_price = self.sell_price_sum / self.trade_count_sell if self.trade_count_sell > 0 else 0.0
        d = self.last_diagnostics
        record = {
            "algorithm": "sac",
            "agent_id": int(self.unique_id),
            "epoch": int(self.current_epoch),
            "episode": int(self.episode_counter),
            "timestamp": str(self.model.current_date),
            "cumulative_reward": round(float(self.cumulative_reward), 4),
            "actual_profit_eur": round(float(self.cumulative_profit), 4),
            "energy_bought_kwh": round(float(self.cumulative_bought), 4),
            "energy_sold_kwh": round(float(self.cumulative_sold), 4),
            "soc_avg": round(float(np.mean(soc_arr)), 4),
            "soc_min": round(float(np.min(soc_arr)), 4),
            "soc_max": round(float(np.max(soc_arr)), 4),
            "trade_count_buy": self.trade_count_buy,
            "trade_count_sell": self.trade_count_sell,
            "avg_buy_price": round(avg_buy_price, 4),
            "avg_sell_price": round(avg_sell_price, 4),
            "alpha": round(float(self.learner.alpha.item()), 6),
            "critic_loss": round(float(d.get("critic_loss", 0.0)), 6),
            "entropy": round(float(d.get("entropy", 0.0)), 6),
            "buffer_size": int(len(self.learner.buffer)),
            "total_updates": int(self.learner.total_updates),
        }
        with open("output/sac/episode_logs.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def _reset_episode_counters(self):
        self.episode_profits.append(self.cumulative_profit)
        self.cumulative_reward = 0.0
        self.cumulative_profit = 0.0
        self.cumulative_bought = 0.0
        self.cumulative_sold = 0.0
        self.trade_count_buy = 0
        self.trade_count_sell = 0
        self.buy_price_sum = 0.0
        self.sell_price_sum = 0.0

    def offer_function(self, price):
        """Create offer function for market."""
        def expr(power):
            return price * power * (self.model.timestep.seconds / 3600)
        return expr

    def step(self):
        """Agent step function."""
        power = self.provide_a_power()
        self.max_power_discharge = power[0]
        self.max_power_charge = power[1]
        
        # Reset
        self.coefficients_ask = [0, 0]
        self.coefficients_bid = [0, 0]
        self.ask = [0, 0, self.offer_function(0), "lin"]
        self.bid = [0, 0, self.offer_function(0), "lin"]
        
        if self.LEC_participation:
            if self.method == "optimisation":
                buy_price_1 = abs(self.max_buy_price[self.updated])
                sell_price_1 = abs(self.min_sell_price[self.updated])
                
                if np.isinf(sell_price_1):
                    self.ask_function = 0
                    self.coefficients_ask = [0, 0]
                else:
                    self.ask_function = self.offer_function(sell_price_1)
                    self.coefficients_ask = [0, sell_price_1 * (self.model.timestep.seconds / 3600)]
                
                if buy_price_1 == 0:
                    self.bid = [0, 0, self.offer_function(0), "lin"]
                else:
                    self.bid_function = self.offer_function(buy_price_1)
                    self.coefficients_bid = [0, buy_price_1 * (self.model.timestep.seconds / 3600)]
                    if self.bid_function(1) == 0 or self.max_power_charge < 0.000005:
                        self.bid = [0, 0, self.offer_function(0), "lin"]
                    else:
                        self.bid = [0, self.max_power_charge, self.bid_function, "lin"]
                
                if self.ask_function == 0 or self.ask_function(1) == 0 or self.max_power_discharge < 0.000005:
                    self.ask = [0, 0, self.offer_function(0), "lin"]
                else:
                    self.ask = [0, self.max_power_discharge, self.ask_function, "lin"]
                
                self.updated += 1
            
            elif self.method == "learning":
                state = self.build_state()
                action = self.learner.select_action(state, deterministic=self.eval_mode)

                # Hard safety overrides (emergencies only; excluded from replay)
                override = False
                if self.soc < 0.10:
                    action = 1.0
                    override = True
                elif self.soc > 0.95:
                    action = -1.0
                    override = True

                self.last_state = state
                self.last_action = action
                self.last_decision_price = self.get_current_price()
                self.last_soc = self.soc
                self.last_override_active = override

                self.action_to_bid(action)
        
        else:  # Non-LEC participation
            self.optimal_power_buy_current = np.clip(
                self.optimal_power_buy[self.updated], 0, self.max_power_charge
            )
            self.optimal_power_sell_current = np.clip(
                self.optimal_power_sell[self.updated], 0, self.max_power_discharge
            )
            self.updated += 1
#-------------------------------------------------------------------------------------------
class EV(mesa.Agent):

    def __init__(self,  model, car_type, base_node, work_node, loading_power, profile, method):
        super().__init__(model)
        self.car_type = car_type
        self.base_node = base_node
        self.bus=base_node
        self.work_node = work_node
        self.home_base_loading_power = 0
        self.work_loading_power = loading_power
        self.profile = profile	
        self.cosphi= float(config.cars["Cosphi"][config.cars["type"]==self.car_type].values[0]) #only, if connected to node within LEC 
        self.consumption_1km = 0
        self.max_capacity = 0
        self.SOC = 0
        self.SOC_rel = 0
        self.min_loading_power = 0
        #self.mean_loading_power = 0
        self.max_loading_power = 0
        self.temp_coef_charge = 0
        self.temp_coef_driving = 0
        self.efficiency=0.97
        self.base_price = 0
        self.current_loading_power = 0
        self.utility=0
        self.forecast_steps=config.ev["forecast_steps"]
        self.utility_df=config.cars.loc[config.cars["type"]==car_type,:]
        self.capital_inside_LEC=0
        self.capital_outside_LEC=0
        self.energy_inside_LEC=0
        self.energy_outside_LEC=0
        self.utility_ext=0
        self.flex=2
        self.margin_charge=self.model.margin_charge_ext
        self.method=method
        self.max_prognosis=83
        self.max_buy_price=0
        self.min_sell_price=np.inf
        self.price_out=0
        self.power_out=0
        self.updated=84
        self.ask=0 #only buys energy
        self.type="quad"
       
        """
        car_status:
        1: loading
        2: nothing
        3: driving
        4: at work
        """
        self.status = 2
        self.current_km = 0
        self.load_parameter()
        self.bid_ext=[0,0]
        self.risk_aversion=[1.35,1.2]
        self.typ="EV"
        self.profile_df=self.get_profile_df()
        self.optimal_power_buy=0
        self.optimal_power_out=0
        self.LEC_participation=True


    def load_parameter(self):
        self.max_capacity = float(config.cars.loc[config.cars["type"] == self.car_type]["capacity"].reset_index(drop=True)[0])
        self.SOC = self.max_capacity * 0.4
        self.SOC_rel = self.SOC/self.max_capacity
        self.consumption_1km = float(config.cars.loc[config.cars["type"] == self.car_type]["consumption_1km"].reset_index(drop=True)[0])
        self.home_base_loading_power = self.model.grid.bus.loc[self.base_node, 'ev_charger_p_max']
        
        for year in [2021, 2022, 2023]:
            try:
                profile_attr = f'profiles{year}'
                if hasattr(config, profile_attr):
                    setattr(self, f'profile_df_{year}', 
                            getattr(config, profile_attr).loc[getattr(config, profile_attr).index == self.profile].reset_index(drop=True))
            except Exception as e:
                print(f"Error processing {year}: {e}")
        
    
    def get_profile_df(self):
        if (self.model.current_date+timedelta(minutes=15)).year==2021:
            profile_df =self.profile_df_2021
        if (self.model.current_date+timedelta(minutes=15)).year==2022:
            profile_df = self.profile_df_2022
        if (self.model.current_date+timedelta(minutes=15)).year==2023:
            profile_df = self.profile_df_2023
        return(profile_df)
    
    def temperature_efficiency_driving(self, temp_c):
        if temp_c < 0:
            self.temp_coef_driving = 1.3
        elif temp_c > 35:
            self.temp_coef_driving = 1.1
        else:
            self.temp_coef_driving = 1.0

    def temperature_efficiency_charging(self, temp_c):
        if temp_c < 0:
            self.temp_coef_charge = 0.5
        elif temp_c < 15:
            self.temp_coef_charge = 0.7
        elif temp_c > 35:
            self.temp_coef_charge = 0.8
        else:
            self.temp_coef_charge = 1.0

    def car_status(self):
        if self.model.current_date.year !=  (self.model.current_date - timedelta(minutes=15)).year:
            self.profile_df=self.get_profile_df()
        if len(self.profile_df)==0:
               self.profile_df=self.get_profile_df()
        self.status = self.profile_df[str(self.model.current_date)].values[0]
        self.current_km = self.profile_df[str(self.model.current_date)].values[1]
        self.temperature_efficiency_driving(self.model.temperature)
        self.temperature_efficiency_charging(self.model.temperature)

        if self.status ==1:
            self.max_loading_power = self.home_base_loading_power
            self.current_bus=self.base_node
        
        if self.status == 2:
            self.max_loading_power = 0
            self.current_bus=np.nan
       
        if self.status == 3:
            self.max_loading_power = 0
            self.current_bus=np.nan

        if self.status == 4:
            self.current_bus=self.work_node
            self.max_loading_power = self.work_loading_power

    
    def charging(self, loading_power):
          if (self.max_capacity - self.SOC) >= loading_power*(self.model.timestep.seconds/(60*60)) :
                self.SOC = self.SOC + loading_power * (self.model.timestep.seconds/(60*60))*self.efficiency
                self.SOC_rel = self.SOC / self.max_capacity
          else:
                self.current_loading_power = ((self.max_capacity - self.SOC))
                self.SOC = self.max_capacity
                self.SOC_rel = self.SOC / self.max_capacity


    def driving(self):
        return self.consumption_1km * self.current_km * self.temp_coef_driving*0.8


    def update_status(self):
        result={}
        power=0
        energy=0
        self.car_status()
        if len(self.model.results)!=0:
            try:
                result=self.model.results[int(self.model.stepcount-1)]["agents"]
            except Exception:
                result={}
            if isinstance(result, pd.DataFrame):
                result=result[result["Agent ID"]==self.unique_id]   
                energy=np.abs(result["Energy bought [kWh]"]).values[0]
                #self.capital_inside_LEC+=result["Revenue Energy LEC [€]"].values[0]+result["Revenue Energy External [€]"].values[0]
                #self.energy_inside_LEC+=energy
                
        if self.bid_ext:
            if self.bid_ext[0]!=0:
                margin_charge=[a for a in self.model.agents if a.flex == 2][0].margin_charge
                price=self.model.agents[self.model.grid.ext_grid["agent_id"].values[0]-1].energy_price
                price=price[price["time"]==(self.model.current_date-self.model.timestep)]["price"].values[0]+margin_charge
                if self.bid_ext[2]>=price:
                    energy=np.abs(self.bid_ext[0]*self.model.sref*self.model.timestep.seconds/(60*60))
                    revenue=price/100*energy
                    self.capital_outside_LEC+=revenue
                    self.energy_outside_LEC+=energy
                
                
        if self.status == 1:
            self.charging(energy*4)
            self.SOC_rel = self.SOC/self.max_capacity

        if self.status == 2:
            self.current_loading_power = 0

        if self.status == 3:
            self.SOC = self.SOC - self.driving()
            self.SOC_rel = self.SOC / self.max_capacity
            self.current_loading_power = 0

        if self.status == 4:
            self.charging(energy*4)
            self.SOC_rel = self.SOC/self.max_capacity

        if self.SOC>1*self.max_capacity:
            self.SOC=1*self.max_capacity
            print("Error in SOC calculation, SOC>max_capacity")
        
        if self.SOC<0:
            self.SOC=0
            print("Error in SOC calculation, SOC<0")

    def forecast_min(self, i):
        counter_load = 0
        energy_consumption = 0
        energy_loading = []

        start_step = self.model.current_date

        if isinstance(start_step, pd.Timestamp) or isinstance(start_step, datetime):
            start_step = start_step.strftime("%Y-%m-%d %H:%M:%S")

        data_series =  self.profile_df.iloc[0, :]

        if start_step not in data_series.index:
            raise ValueError("Der Startschritt ist nicht im DataFrame enthalten.")

        start_index = data_series.index.get_loc(start_step)
        end_index = min(start_index + i, len(data_series))
        counter=0
        for n in range(start_index, end_index):
            if data_series.iloc[n]==3:
                if data_series.iloc[n+1] in (1,4):
                    counter+=1
            if counter==2:
                break
        n=min(i,n, len(data_series))
        expected_soc=pd.DataFrame(index=range(n-1),columns=["SOC"])
        expected_soc.loc[n,"SOC"]=self.max_capacity*0.1
        step=n
        for outer_step in range(start_index, n+start_index):
                try:
                    current_value = data_series.iloc[ n+2*start_index-outer_step]
                except:
                    print()
                time=self.model.current_date+self.model.timestep*(n+start_index-outer_step)                
            
                if current_value == 1:   
                    #self.temperature_efficiency_charging(self.model.temperature)
                    energy_loading=self.home_base_loading_power*0.7*self.model.timestep.seconds/(60*60)
                    expected_soc.loc[step-1,"SOC"]=max(expected_soc.loc[step,"SOC"]-energy_loading,0)

                elif current_value == 2:
                    expected_soc.loc[step-1,"SOC"]=expected_soc.loc[step,"SOC"]
    
                elif current_value == 3:
                    km = self.profile_df[str(time)].values[1]
                    #self.temperature_efficiency_driving(self.model.temperature)
                    energy_consumption = self.consumption_1km * km * 1.35
                    expected_soc.loc[step-1,"SOC"]=expected_soc.loc[step,"SOC"]+energy_consumption 
                    
                elif current_value == 4:
                    #self.temperature_efficiency_charging(self.model.temperature)
                    energy_loading=(self.work_loading_power * 0.7 * self.model.timestep.seconds/(60*60))
                    expected_soc.loc[step-1,"SOC"]=max(expected_soc.loc[step,"SOC"]-energy_loading,0)
                if  expected_soc.loc[step-1,"SOC"] is np.nan:
                    print("")
                step-=1 
 
        self.min_loading_power= max((expected_soc.loc[0,"SOC"]-self.SOC)/0.7/self.model.timestep.seconds*(60*60),0)


    def forecast_max(self):
        if self.status == 1:   
             self.max_loading_power = self.home_base_loading_power*self.temp_coef_charge
        elif self.status == 4:   
             self.max_loading_power = self.work_loading_power*self.temp_coef_charge
        else:
              self.max_loading_power = 0  
              
        if (self.max_capacity - self.SOC) < (self.max_loading_power)*(self.model.timestep.seconds/(60*60)):
                 self.max_loading_power= ((self.max_capacity - self.SOC))/(self.model.timestep.seconds/(60*60))
        
    def bid_function_max(self, a_power):
         a=1.5*a_power*a_power*(self.model.timestep.seconds/(60*60))
         return a
    
    def step(self):
        self.bid_function, self.coefficients_bid = fit_function_buy(self, 0, 0, 0)
        self.bid_ext=[0,0,0,"lin"]
        self.bid=[0,0,self.bid_function,"lin"]
        self.forecast_min(self.forecast_steps)
        self.forecast_max() 
        if self.LEC_participation==True:
            if self.method=="optimisation":
                #if self.max_prognosis<= self.updated:
                        #optimize(self)
                if self.status in [1,4]:
                    buy_price_1=abs(self.max_buy_price[self.updated]*self.risk_aversion[0])*(self.model.timestep.seconds/(60*60))
                    buy_price_2=abs(self.max_buy_price[self.updated]*self.risk_aversion[1])*(self.model.timestep.seconds/(60*60))
                    self.bid_function,  self.coefficients_bid=fit_function_buy(self, self.max_loading_power/self.model.sref,buy_price_1, buy_price_2)
                    self.forecast_min(self.forecast_steps)
                    self.forecast_max() 
                    if self.min_loading_power != 0:
                        self.bid_function=self.bid_function_max
                        if not np.isnan(self.bus):
                            self.bid=[0, self.max_loading_power/self.model.sref, self.bid_function, "quad"]
                            self.coefficients_bid=[0,0,1.5**2*(self.model.timestep.seconds/(60*60))]
                        if np.isnan(self.bus):
                            self.bid_ext=[self.max_loading_power/self.model.sref, self.max_loading_power/self.model.sref, self.bid_function, "lin"] 
                    elif not np.isnan(self.bus):
                        if self.bid_function(1)==0:
                                self.bid=[0,0,self.bid_function,"quad"]
                        else:
                            self.bid=[0, self.max_loading_power/self.model.sref, self.bid_function, "quad"]
                    elif np.isnan(self.bus):
                        self.bid_ext=[self.power_out[self.updated]/self.model.sref, self.power_out[self.updated]/self.model.sref, self.price_out[self.updated], "lin"]
               
            if self.method=="learning":
                self.bid_function,  self.coefficients_bid=fit_function_buy(self, 0,0, 0)
                self.bid=[0,0,self.bid_function, "quad"]
                self.bid_ext=[0,0,0,"lin"]
        if self.LEC_participation==False:
            self.optimal_power_buy_current = np.clip(self.optimal_power_buy[self.updated], self.min_loading_power/self.model.sref, self.max_loading_power/self.model.sref)
            self.optimal_power_out_current=np.clip(self.optimal_power_out[self.updated], self.min_loading_power, self.max_loading_power)
            if self.optimal_power_buy_current>2:
                 print("extremly high loading pwoer")
        self.updated+=1