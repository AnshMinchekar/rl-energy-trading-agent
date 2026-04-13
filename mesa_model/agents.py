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
import json
import os

def initialize(self, typ, factor, cosphi, bus):
    self.typ=typ
    curve=self.model.grid.load.profile[self.model.grid.load.agent_id==self.unique_id].values[0]+"_pload"
    self.a_power_profile=pd.DataFrame({"time":pd.to_datetime(config.load_profile["time"],format='%d.%m.%Y %H:%M',  errors='coerce'),"power":config.load_profile[curve].astype(float)*factor/self.model.sref})
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
        power=abs(self.a_power_profile.loc[self.a_power_profile["time"]==self.model.current_date,"power"])
        if power.empty:
            power=0
        else:
            power=power.values[0]
        a_power_min=power
        a_power_max=power
        return [a_power_min, a_power_max]    
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
            self.price=0
            self.energy=0
        else:
            self.price=price.values[0]
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
            # print(f"Looking for: {self.model.current_date + self.model.timestep}")
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
            if len(self.model.results)!=0:
                try:
                    result=self.model.results[int(self.model.stepcount-1)]["agents"]
                except Exception:
                    result={}
                if isinstance(result, pd.DataFrame):
                    result=result[result["Agent ID"]==self.unique_id]   
                    energy=np.abs(result["Energy bought [kWh]"]).values[0]
            else:
                energy=0
                
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
    
    def __init__(self, model, capacity, power, node, efficiency, discharge, method):
        super().__init__(model)
        self.capacity = capacity
        # discharge is a per-day retention factor (e.g. 0.87 → lose 13% per day).
        # Scale to per-step so update_status applies the right fraction each timestep.
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
        
        # Initialize bid/ask
        self.ask = 0
        self.bid = 0
        self.coefficients_ask = [0, 0]
        self.coefficients_bid = [0, 0]
        
        if self.method == "learning":
            self.soc = config.storage["SOC_start"]
            
            # 8-neuron hidden layer stored as numpy arrays.
            # Input order: [soc, price_norm, price_diff_1h, price_diff_4h,
            #               sin_hour, cos_hour, sin_dow, cos_dow, price_percentile]
            # (max_discharge and max_charge removed — fully determined by soc)
            # (hour/dow replaced by sin/cos pairs for continuous circular encoding)
            # 8 neurons gives more capacity to learn the joint (soc × price × time) arbitrage
            # signal that 4 neurons were too constrained to represent simultaneously.
            n_hidden, n_inputs = 8, 9
            # Small random initialization so gradients can flow from the start.
            # Scale 0.1 keeps the initial policy output near-neutral (±0.1 range)
            # so the soc/price biases drive early exploration, but non-zero h1 and
            # W2 ensure ΔW = lr × δ × h1 is non-zero and learning can begin.
            rng = np.random.RandomState(seed=42)
            scale = 0.1
            W1 = rng.randn(n_hidden, n_inputs).astype(float) * scale
            b1 = np.zeros(n_hidden, dtype=float)
            W2 = rng.randn(n_hidden).astype(float) * scale
            b2 = np.array([0.0], dtype=float)

            self.theta = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

            # Critic value function weights (linear approximation).
            # Warm-started with sensible priors so the critic isn't fully blind early on.
            self.value_theta = {
                "soc_w":      0.0,
                "soc_sq_w":  -1.0,   # penalise SOC extremes from the start
                "price_w":    0.3,   # higher price → better state to sell from
                "price_diff_w": 0.0,
                "hour_w":     0.0,
                "bias":       0.0,
            }
            
            self.actor_lr = 0.03       # Policy learning rate
            self.critic_lr = 0.05      # Value function learning rate 
            self.gamma = 0.98          # Discount factor
            self.memory = deque(maxlen=4000)
            self.batch_size = 32
            
            # Tracking
            self.last_state = None
            self.last_action = None
            self.last_raw_action = None   # raw tanh(z) before bias/noise — used for gradients
            self.last_hidden = None       # post-activation h1 (shape: n_hidden)
            self.last_h1_in = None        # pre-activation h1_in — needed for LeakyReLU derivative
            self.episode_profits = []
            self.cumulative_reward = 0
            self.cumulative_profit = 0
            self.cumulative_bought = 0
            self.cumulative_sold = 0
            self.update_frequency = 48
            self.episode_counter = 0
            
            # Exploration
            self.exploration_rate = 0.35
            self.exploration_decay = 0.997
            self.min_exploration = 0.08
            
            # Performance tracking
            self.best_profit = -np.inf
            self.best_theta = None
            self.soc_history = deque(maxlen=2000)
            
            # SOC operating range
            self.soc_floor = 0.10
            self.soc_ceiling = 0.85
            self.soc_target = 0.50
            
            # Price cache (for fast point lookups and forecasts)
            self._price_index = None
            self._price_array = None
            self._price_cache_ready = False

            # Rolling 24-hour observed price history (96 steps × 15 min = 24 h).
            # Used for average/percentile so we only use prices the agent has seen.
            self.price_history = deque(maxlen=96)

            # TD learning tracking
            self.td_errors = []  # Track TD errors for debugging

    
    def provide_a_power(self):
        """Calculate available charge/discharge power."""
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
        """Build price lookup cache for O(1) access."""
        try:
            price_df = self.model.market_price
            self._price_index = dict(zip(price_df["time"], price_df["price"]))
            self._price_array = price_df["price"].values
            self._price_cache_ready = True
        except (AttributeError, TypeError, KeyError):
            self._price_cache_ready = False
    
    def get_current_price(self):
        """Get current market price."""
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

    def get_price_forecast(self, hours_ahead):
        """Get forecasted price N hours ahead."""
        if not self._price_cache_ready:
            self._initialize_price_cache()
        
        try:
            steps_ahead = int(hours_ahead * 3600 / self.model.timestep.seconds)
            forecast_time = self.model.current_date + self.model.timestep * steps_ahead
            
            if self._price_cache_ready and self._price_index:
                price = self._price_index.get(forecast_time)
                if price is not None:
                    return float(price)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        return self.get_current_price()

    def get_average_price(self, hours_back=24):
        """Get average price over the last N observed hours.

        Uses the rolling price_history deque (populated in update_status) so only
        prices the agent has actually seen are included — no future leakage.
        """
        if len(self.price_history) > 0:
            return float(np.mean(self.price_history))
        return self.get_current_price()

    def get_price_percentile(self, hours_back=24):
        """Get current price percentile relative to recent observed prices."""
        if len(self.price_history) > 1:
            current_price = self.get_current_price()
            arr = np.array(self.price_history)
            return float(np.sum(arr < current_price) / len(arr))
        return 0.5

    
    def build_state(self):
        """Build state representation for RL policy.

        State (9-D):
          [soc, price_norm, price_diff_1h, price_diff_4h,
           sin_hour, cos_hour, sin_dow, cos_dow, price_percentile]

        Time is encoded as sin/cos pairs for continuity across midnight and week boundaries.
        max_discharge/max_charge are omitted — they are deterministic functions of soc.
        """
        soc = float(self.soc)

        price_now = self.get_current_price()
        price_forecast_1h = self.get_price_forecast(1)
        price_forecast_4h = self.get_price_forecast(4)
        avg_price = self.get_average_price(24)
        price_percentile = self.get_price_percentile(24)

        price_norm    = (price_now - avg_price) / (avg_price + 1e-6)
        price_diff_1h = (price_forecast_1h - price_now) / (price_now + 1e-6)
        price_diff_4h = (price_forecast_4h - price_now) / (price_now + 1e-6)

        current_time = self.model.current_date
        hour_rad = 2 * np.pi * current_time.hour / 24.0
        dow_rad  = 2 * np.pi * current_time.weekday() / 7.0
        sin_hour, cos_hour = float(np.sin(hour_rad)), float(np.cos(hour_rad))
        sin_dow,  cos_dow  = float(np.sin(dow_rad)),  float(np.cos(dow_rad))

        return [soc, price_norm, price_diff_1h, price_diff_4h,
                sin_hour, cos_hour, sin_dow, cos_dow, price_percentile]

    
    def estimate_value(self, state):
        # State layout: [soc, price_norm, price_diff_1h, price_diff_4h,
        #                sin_hour, cos_hour, sin_dow, cos_dow, price_percentile]
        soc        = state[0]
        price_norm = state[1]
        price_diff = state[2]
        sin_hour   = state[4]   # hour_w now tracks sin(hour) — continuous proxy for time-of-day

        value = (
            self.value_theta["soc_w"]      * (soc - self.soc_target) +
            self.value_theta["soc_sq_w"]   * ((soc - self.soc_target) ** 2) +
            self.value_theta["price_w"]    * price_norm +
            self.value_theta["price_diff_w"] * price_diff +
            self.value_theta["hour_w"]     * sin_hour +
            self.value_theta["bias"]
        )

        return value

    def policy(self, state):
        """Two-layer neural network policy (4 hidden neurons).

        Returns
        -------
        action      : float in [-1, 1]  — final action sent to the market
        raw_action  : float             — tanh(z) before any bias/noise/clip (used for gradients)
        h1          : ndarray (n_hidden,) — post-activation hidden values
        h1_in       : ndarray (n_hidden,) — pre-activation hidden values (for LeakyReLU derivative)
        """
        x = np.array(state, dtype=float)   # shape (9,)
        soc, _, _, _, _, _, _, _, price_percentile = state

        # --- Layer 1: hidden (LeakyReLU) ---
        h1_in = self.theta["W1"] @ x + self.theta["b1"]          # (n_hidden,)
        h1 = np.where(h1_in > 0, h1_in, 0.1 * h1_in)            # LeakyReLU
        h1 = np.clip(h1, -5.0, 5.0)

        # --- Layer 2: output (Tanh) ---
        z = float(np.dot(self.theta["W2"], h1) + self.theta["b2"][0])
        raw_action = float(np.tanh(z))                            # kept clean for gradient use
        action = raw_action

        # --- Soft biases (10% influence — reduced from 40% to limit credit-assignment noise) ---
        # Soft SOC biases — only activate at true extremes so the learned policy dominates
        # in the normal trading range (10–85%). Biases match the updated soc_ceiling of 0.85.
        soc_bias = 0.0
        if soc < 0.12:
            soc_bias = 0.8        # Very low SOC: push to charge hard
        elif soc < self.soc_floor:
            soc_bias = 0.4
        elif soc > 0.93:
            soc_bias = -0.8       # Very high SOC: push to discharge hard
        elif soc > self.soc_ceiling:
            soc_bias = -0.5 * (soc - self.soc_ceiling) / 0.10

        price_bias = 0.0
        if price_percentile < 0.15:
            price_bias = 0.4
        elif price_percentile < 0.25:
            price_bias = 0.2
        elif price_percentile > 0.85:
            price_bias = -0.4
        elif price_percentile > 0.75:
            price_bias = -0.2

        # 90% network / 10% bias so the learned policy dominates
        action = 0.9 * action + 0.1 * (soc_bias + price_bias)

        # --- Exploration noise ---
        if 0.20 < soc < 0.90:
            noise = np.random.normal(0, max(self.min_exploration, self.exploration_rate) * 0.4)
            action += noise

        action = float(np.clip(action, -1.0, 1.0))

        # --- Hard override: true emergencies only ---
        if soc < 0.10:
            action = 1.0
        elif soc > 0.95:
            action = -1.0

        return action, raw_action, h1, h1_in

    
    def action_to_bid(self, action):
        """Convert RL action to market bid/ask.

        Override policy:
          soc < 0.10          → emergency charge (hard override, return early)
          soc < soc_floor     → clip action so the agent can only charge (RL magnitude kept)
          otherwise           → full RL control
        """
        self.ask = [0, 0, self.offer_function(0), "lin"]
        self.bid = [0, 0, self.offer_function(0), "lin"]
        self.coefficients_ask = [0, 0]
        self.coefficients_bid = [0, 0]

        max_discharge, max_charge = self.provide_a_power()
        eps = 1e-6
        price_now = self.get_current_price()

        # --- True emergency: charge at any cost ---
        if self.soc < 0.10:
            if max_charge > eps:
                bid_price = 1000.0
                bid_fun = self.offer_function(bid_price)
                self.bid = [max_charge, max_charge, bid_fun, "lin"]
                self.coefficients_bid = [0, bid_price * (self.model.timestep.seconds / 3600)]
            return

        # --- Below soc_floor: allow charge-only (keep RL magnitude, forbid discharge) ---
        if self.soc < self.soc_floor:
            action = max(action, 0.0)

        # --- Safe discharge ceiling ---
        if self.soc > self.soc_floor:
            max_safe_discharge_kwh = (self.soc - self.soc_floor) * self.capacity
            max_safe_discharge_power = (
                max_safe_discharge_kwh * (3600 / self.model.timestep.seconds)
                / self.model.sref * self.efficiency
            )
            max_safe_discharge = min(max_discharge, max_safe_discharge_power)
        else:
            max_safe_discharge = 0

        price_percentile = self.get_price_percentile()

        if action > eps:   # Charge
            desired_power = min(action * max_charge, max_charge)
            if desired_power > eps:
                # Bid premium scales with price context:
                # At low percentiles (genuinely cheap) → small premium to avoid overpaying.
                # At mid/high percentiles → larger premium to ensure fill despite competition.
                if price_percentile < 0.15:
                    bid_premium = 0.5   # very cheap: don't overpay
                elif price_percentile < 0.35:
                    bid_premium = 0.8
                else:
                    bid_premium = 1.2   # default: need premium to beat ext_grid margin
                lec_min = self.model.gridfee_LEC + self.model.levies_LEC
                bid_price = max(price_now + bid_premium, lec_min + bid_premium)
                bid_fun = self.offer_function(bid_price)
                self.bid = [0, desired_power, bid_fun, "lin"]
                self.coefficients_bid = [0, bid_price * (self.model.timestep.seconds / 3600)]

        elif action < -eps:   # Discharge
            desired_power = min(-action * max_discharge, max_safe_discharge)
            if desired_power > eps:
                # Ask discount scales with price context:
                # At high percentiles (genuinely expensive) → small discount to maximise revenue.
                # At mid/low percentiles → larger discount to ensure sell is filled.
                if price_percentile > 0.85:
                    ask_discount = 0.1   # very expensive: stay near market to capture full price
                elif price_percentile > 0.65:
                    ask_discount = 0.25
                else:
                    ask_discount = 0.4   # default
                ask_price = max(price_now - ask_discount, 0.01)
                ask_fun = self.offer_function(ask_price)
                self.ask = [0, desired_power, ask_fun, "lin"]
                self.coefficients_ask = [0, ask_price * (self.model.timestep.seconds / 3600)]

    
    def compute_reward(self, bought, sold, price):
        """Reward function with three components.

        Components are scaled to roughly the same order of magnitude so the
        agent can learn price-timing without being overwhelmed by SOC alarms.

        1. profit_reward   — financially accurate primary signal
        2. arbitrage_bonus — additive-only timing nudge (no double-penalising)
        3. soc_penalty     — safety guardrail, only at true extremes (soc<0.10 or >0.92)
        """
        avg_price = self.get_average_price(24)
        price_ratio = price / (avg_price + 1e-6)

        # 1. Primary: actual financial cost/profit (scaled ×5 for magnitude)
        energy_cost_eur = price * (bought - sold) / 100
        profit_reward = -energy_cost_eur * 5.0

        # 2. Timing bonus — broader price thresholds so good-but-not-extreme trades are rewarded.
        # Scaled by trade size so the agent is rewarded proportionally to commitment: a full-power
        # trade at a good price earns the full bonus; a tiny hedge earns only a fraction.
        max_energy_per_step = self.max_power * (self.model.timestep.seconds / 3600.0)  # kWh
        arbitrage_bonus = 0.0
        if bought > 0.01:
            if price_ratio < 0.70:
                arbitrage_bonus += 2.0
            elif price_ratio < 0.85:
                arbitrage_bonus += 1.0
            elif price_ratio < 0.95:
                arbitrage_bonus += 0.3
            trade_size_norm = min(bought / (max_energy_per_step + 1e-6), 1.0)
            arbitrage_bonus *= (0.3 + 0.7 * trade_size_norm)
        if sold > 0.01:
            sell_bonus = 0.0
            if price_ratio > 1.30:
                sell_bonus += 2.0
            elif price_ratio > 1.15:
                sell_bonus += 1.0
            elif price_ratio > 1.05:
                sell_bonus += 0.3
            trade_size_norm = min(sold / (max_energy_per_step + 1e-6), 1.0)
            arbitrage_bonus += sell_bonus * (0.3 + 0.7 * trade_size_norm)

        # 3. SOC safety penalty — only at true extremes, does not overlap trading range
        soc_penalty = 0.0
        if self.soc < 0.10:
            soc_penalty = -2.0 * (0.10 - self.soc) / 0.10   # max -2.0 at soc=0
        elif self.soc > 0.92:
            soc_penalty = -2.0 * (self.soc - 0.92) / 0.08   # max -2.0 at soc=1

        # NOTE: soc_bonus removed — it rewarded inaction (maintaining soc≈0.5 without trading)
        # and overwhelmed the profit signal. The profit_reward + arbitrage_bonus are sufficient.

        return profit_reward + arbitrage_bonus + soc_penalty

    
    def update_status(self):
        """Update SOC and perform RL learning."""
        if len(self.model.results) == 0:
            return
        
        try:
            result = self.model.results[int(self.model.stepcount - 1)]["agents"]
            result = result[result["Agent ID"] == self.unique_id]
        except (KeyError, IndexError):
            return
        
        if len(result) == 0:
            return
        
        bought = result["Energy bought [kWh]"].to_numpy()[0]
        sold = result["Energy sold [kWh]"].to_numpy()[0]
        # DEBUG — remove once trading is confirmed
        if self.method == "learning" and self.model.stepcount <= 4:
            print(f"[DBG update step={int(self.model.stepcount)}] bought={bought:.4f} kWh sold={sold:.4f} kWh", flush=True)

        old_soc = self.soc
        energy_delta = bought * self.efficiency - sold / self.efficiency
        soc_delta = energy_delta / self.capacity
        self.soc = min(max((old_soc * self.discharge) + soc_delta, 0), 1)
        
        if self.method == "learning":
            self.cumulative_bought += bought
            self.cumulative_sold += sold

            self.soc_history.append(self.soc)

            # Add observed price to rolling window (used by get_average_price / get_price_percentile)
            price = self.get_current_price()
            self.price_history.append(price)

            if self.last_state is not None and self.last_action is not None:

                actual_cost = price * (bought - sold) / 100
                self.cumulative_profit -= actual_cost

                reward = self.compute_reward(bought, sold, price)
                self.cumulative_reward += reward

                next_state = self.build_state()

                self.memory.append((
                    self.last_state,
                    self.last_action,
                    self.last_raw_action,   # raw tanh(z) — used for correct gradient computation
                    reward,
                    next_state,
                    self.last_hidden,       # post-activation h1 (ndarray)
                    self.last_h1_in,        # pre-activation h1_in — needed for LeakyReLU derivative
                    self.soc,
                ))

                self.last_state = next_state

                if self.model.stepcount % self.update_frequency == 0:
                    self.episode_counter += 1
                    self.update_parameters()

                    self._print_episode_summary()
                    self._log_episode()
                    self._reset_episode_counters()

    def _print_episode_summary(self):
        recent_soc = list(self.soc_history)[-96:] if len(self.soc_history) >= 96 else list(self.soc_history)
        avg_soc = np.mean(recent_soc) if recent_soc else self.soc
        min_soc = np.min(recent_soc) if recent_soc else self.soc
        max_soc = np.max(recent_soc) if recent_soc else self.soc
        
        avg_td_error = np.mean(self.td_errors[-100:]) if self.td_errors else 0
        
        print(f"\n{'='*60}")
        print(f"[Storage {self.unique_id}] Episode {self.episode_counter} Summary")
        print(f"{'='*60}")
        print(f"  Cumulative Reward:   {self.cumulative_reward:>12.2f}")
        print(f"  Actual Profit (€):   {self.cumulative_profit:>12.4f}")
        print(f"  Energy Bought (kWh): {self.cumulative_bought:>12.2f}")
        print(f"  Energy Sold (kWh):   {self.cumulative_sold:>12.2f}")
        print(f"  Net Energy (kWh):    {self.cumulative_bought - self.cumulative_sold:>12.2f}")
        print(f"  Exploration Rate:    {self.exploration_rate:>12.4f}")
        print(f"  Memory Size:         {len(self.memory):>12d}")
        print(f"  Current SOC:         {self.soc*100:>12.1f}%")
        print(f"  SOC Range (24h):     {min_soc*100:>6.1f}% - {max_soc*100:.1f}%")
        print(f"  Avg SOC (24h):       {avg_soc*100:>12.1f}%")
        print(f"  Avg TD Error:        {avg_td_error:>12.4f}")
        print(f"{'='*60}", flush=True)

        if self.cumulative_profit > self.best_profit:
            self.best_profit = self.cumulative_profit
            self.best_theta = {
                "W1": self.theta["W1"].copy(),
                "b1": self.theta["b1"].copy(),
                "W2": self.theta["W2"].copy(),
                "b2": self.theta["b2"].copy(),
            }
            print(f"  *** New best profit: €{self.best_profit:.4f} ***", flush=True)

    def _log_episode(self):
        """Log episode to JSON file."""
        recent_soc = list(self.soc_history)[-96:] if len(self.soc_history) >= 96 else list(self.soc_history)
        soc_arr = np.array(recent_soc) if recent_soc else np.array([self.soc])

        episode_log = {
            "agent_id": int(self.unique_id),
            "episode": int(self.episode_counter),
            "timestamp": str(self.model.current_date),
            "cumulative_reward": round(float(self.cumulative_reward), 4),
            "actual_profit_eur": round(float(self.cumulative_profit), 4),
            "energy_bought_kwh": round(float(self.cumulative_bought), 4),
            "energy_sold_kwh": round(float(self.cumulative_sold), 4),
            "exploration_rate": round(float(self.exploration_rate), 6),
            "avg_td_error": round(float(np.mean(self.td_errors[-100:])) if self.td_errors else 0, 4),
            "soc_avg": round(float(np.mean(soc_arr)), 4),
            "soc_min": round(float(np.min(soc_arr)), 4),
            "soc_max": round(float(np.max(soc_arr)), 4),
            "theta": {
                "W1": [[round(float(v), 6) for v in row] for row in self.theta["W1"]],
                "b1": [round(float(v), 6) for v in self.theta["b1"]],
                "W2": [round(float(v), 6) for v in self.theta["W2"]],
                "b2": [round(float(v), 6) for v in self.theta["b2"]],
            },
            "value_theta": {k: round(float(v), 6) for k, v in self.value_theta.items()},
        }
        
        try:
            log_path = os.path.join("output", "episode_logs.jsonl")
            os.makedirs("output", exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(episode_log) + "\n")
        except Exception as e:
            print(f"[Warning] Failed to write episode log: {e}")

    def _reset_episode_counters(self):
        """Reset episode counters."""
        self.episode_profits.append(self.cumulative_profit)
        
        self.cumulative_reward = 0
        self.cumulative_profit = 0
        self.cumulative_bought = 0
        self.cumulative_sold = 0
        
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration)
    
    def _sample_batch(self):
        """Sample a random mini-batch from memory."""
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        return [self.memory[i] for i in indices]

    def _critic_gradient_step(self, batch):
        """Compute and apply one critic gradient update. Returns list of TD errors."""
        grad_value = {k: 0.0 for k in self.value_theta}
        td_errors = []

        for transition in batch:
            state, action, raw_action, reward, next_state, h1, h1_in, soc = transition

            # State layout: [soc, price_norm, price_diff_1h, price_diff_4h,
            #                sin_hour, cos_hour, sin_dow, cos_dow, price_percentile]
            soc_state  = state[0]
            price_norm = state[1]
            price_diff = state[2]
            sin_hour   = state[4]

            V_current = self.estimate_value(state)
            V_next    = self.estimate_value(next_state)
            td_error  = reward + self.gamma * V_next - V_current
            td_errors.append(td_error)

            grad_value["soc_w"]        += td_error * (soc_state - self.soc_target)
            grad_value["soc_sq_w"]     += td_error * ((soc_state - self.soc_target) ** 2)
            grad_value["price_w"]      += td_error * price_norm
            grad_value["price_diff_w"] += td_error * price_diff
            grad_value["hour_w"]       += td_error * sin_hour
            grad_value["bias"]         += td_error

        max_grad = 0.5
        for key in self.value_theta:
            grad = np.clip(grad_value[key] / self.batch_size, -max_grad, max_grad)
            self.value_theta[key] += self.critic_lr * grad

        return td_errors

    def update_parameters(self):
        if len(self.memory) < self.batch_size:
            return

        max_grad = 0.5

        # ── Critic: 5 independent updates before the actor step ───────────
        # Running more critic updates ensures the value function is a good
        # baseline before the actor gradient uses it as an advantage signal.
        all_td_errors = []
        for _ in range(5):
            td_errors = self._critic_gradient_step(self._sample_batch())
            all_td_errors.extend(td_errors)

        #Actor: 1 update using the now-improved critic 
        batch = self._sample_batch()

        grad_W1 = np.zeros_like(self.theta["W1"])
        grad_b1 = np.zeros_like(self.theta["b1"])
        grad_W2 = np.zeros_like(self.theta["W2"])
        grad_b2 = np.zeros_like(self.theta["b2"])

        actor_td_errors = []
        raw_advantages = []
        actor_transitions = []
        for transition in batch:
            state, action, raw_action, reward, next_state, h1, h1_in, soc = transition
            V_current = self.estimate_value(state)
            V_next    = self.estimate_value(next_state)
            td_error  = reward + self.gamma * V_next - V_current
            actor_td_errors.append(td_error)
            raw_advantages.append(td_error)
            actor_transitions.append(transition)

        # Normalize advantages across the batch so large TD spikes don't dominate the gradient.
        adv_arr = np.array(raw_advantages)
        adv_mean = float(np.mean(adv_arr))
        adv_std  = float(np.std(adv_arr)) + 1e-8
        normalized_advantages = (adv_arr - adv_mean) / adv_std

        for i, transition in enumerate(actor_transitions):
            state, action, raw_action, reward, next_state, h1, h1_in, soc = transition
            advantage = float(normalized_advantages[i])

            # Tanh derivative evaluated at raw network output (before bias/noise)
            dz = advantage * (1.0 - raw_action ** 2)

            grad_W2 += dz * h1
            grad_b2 += dz

            leaky_deriv = np.where(h1_in > 0, 1.0, 0.1)
            delta_h = dz * self.theta["W2"] * leaky_deriv
            x = np.array(state, dtype=float)
            grad_W1 += np.outer(delta_h, x)
            grad_b1 += delta_h

        # Adaptive actor learning rate
        actor_lr = self.actor_lr
        if len(self.episode_profits) >= 4:
            recent = np.mean(self.episode_profits[-2:])
            older  = np.mean(self.episode_profits[-4:-2])
            actor_lr *= 1.1 if recent > older else 0.9
            actor_lr = float(np.clip(actor_lr, 0.01, 0.08))

        self.theta["W1"] += actor_lr * np.clip(grad_W1 / self.batch_size, -max_grad, max_grad)
        self.theta["b1"] += actor_lr * np.clip(grad_b1 / self.batch_size, -max_grad, max_grad)
        self.theta["W2"] += actor_lr * np.clip(grad_W2 / self.batch_size, -max_grad, max_grad)
        self.theta["b2"] += actor_lr * np.clip(grad_b2 / self.batch_size, -max_grad, max_grad)

        # ── Weight decay: prevent W1/W2 from growing into tanh saturation ─
        self.theta["W1"] *= 0.999
        self.theta["W2"] *= 0.999

        # ── Store TD errors for monitoring ────────────────────────────────
        self.td_errors.extend(all_td_errors)
        if len(self.td_errors) > 1000:
            self.td_errors = self.td_errors[-1000:]

        avg_td = float(np.mean(all_td_errors))
        std_td = float(np.std(all_td_errors))
        print(f"[Storage {self.unique_id}] Updated (actor_lr={actor_lr:.4f}, critic_lr={self.critic_lr:.4f})")
        print(f"  TD Error: mean={avg_td:.3f}, std={std_td:.3f}")
        print(f"  W2 (output weights): {np.round(self.theta['W2'], 3)}")
        print(f"  Critic: soc={self.value_theta['soc_w']:.3f}, price={self.value_theta['price_w']:.3f}, bias={self.value_theta['bias']:.3f}", flush=True)

    
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
        
        self.coefficients_ask = [0, 0]
        self.coefficients_bid = [0, 0]
        self.ask = [0, 0, self.offer_function(0), "lin"]
        self.bid = [0, 0, self.offer_function(0), "lin"]
        
        if self.LEC_participation:
            if self.method == "optimisation":
                # Optimization-based bidding (unchanged)
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
                # RL-based bidding
                state = self.build_state()
                action, raw_action, hidden, h1_in = self.policy(state)

                self.last_state      = state
                self.last_action     = action
                self.last_raw_action = raw_action
                self.last_hidden     = hidden
                self.last_h1_in      = h1_in

                self.action_to_bid(action)
                # DEBUG — remove once trading is confirmed
                if self.model.stepcount <= 3:
                    price_now = self.get_current_price()
                    print(f"[DBG step={int(self.model.stepcount)}] action={action:.3f} soc={self.soc:.3f} "
                          f"price={price_now:.2f} bid_pmax={self.bid[1]:.4f} ask_pmax={self.ask[1]:.4f} "
                          f"bid_coef={self.coefficients_bid} ask_coef={self.coefficients_ask}", flush=True)

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