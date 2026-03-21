from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from dataclasses import field
# --- Market-level constants ---
@dataclass(frozen=True)
class Market_StaticData:
    n_bids: int
    n_asks: int
    slack_bus: int
    n_nodes: int
    gridfee_LEC: float
    levies_LEC: float
    gridfee_ext: float
    levies_ext: float
    slack_agent_id: int
    s_ref:int
    timestep:float
    margin_buy: float
    margin_sell: float
    solver: str
    
@dataclass()
class MarketResults:
    demand: pd.DataFrame = field(default_factory=pd.DataFrame)
    supply: pd.DataFrame = field(default_factory=pd.DataFrame)
    Node_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    Input_Grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    agents: pd.DataFrame = field(default_factory=pd.DataFrame)

    
    def copy(self):
        """Deep copy all DataFrames (safe for logging between rounds)."""
        import copy
        return MarketResults(
            demand=self.demand.copy(),
            supply=self.supply.copy(),
            Node_results=self.Node_results.copy(),
            Input_Grid=self.Input_Grid.copy(),
            agents=self.agents.copy(),

        )
@dataclass
class Market_VariableData:
    date: str
    timestep: int
    temperature: float
    ext_price: float


# --- Agents (static) ---
@dataclass(frozen=True)
class Agent_Bid_StaticData:
    agent_id: int
    bus: int
    cosphi: float
    bidding_type: str   # "lin" or "quad"
    agent_type: str

@dataclass(frozen=True)
class Agent_Ask_StaticData:
    agent_id: int
    bus: int
    cosphi: float
    asking_type: str 
    agent_type: str   # "lin" or "quad"

# --- Agents (dynamic) ---
@dataclass
class Agent_Bid_RoundData:
    agent_id: int
    p_min: float
    p_max: float
    f_coef: tuple[float, float, float]




@dataclass
class Agent_Ask_RoundData:
    agent_id: int
    p_min: float
    p_max: float
    f_coef: tuple[float, float, float]
