from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import pandas as pd
import numpy as np
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Reals, Binary, Constraint,
    Objective, minimize, value, Piecewise, summation, Expression, RangeSet
)
from pyomo.opt import SolverFactory
import pandapower as pp
from data.data_handler import (
    Market_StaticData,
    Agent_Bid_StaticData,
    Agent_Ask_StaticData,
    Agent_Bid_RoundData,
    Agent_Ask_RoundData,
    MarketResults,
    Market_VariableData,
)
import copy
class MarketOptimizer:
    def __init__(self,
                 market_static: Market_StaticData,
                 bid_static: list[Agent_Bid_StaticData],
                 ask_static: list[Agent_Ask_StaticData],
                 grid_model,
                 solver):
        self.market_static = market_static
        self.bid_static = bid_static
        self.ask_static = ask_static
        self.grid=grid_model
        self.model = self._create_model()
        self.results_template = self._initialize_result_template()
        self.solver_name = solver
        self._solver_instance = SolverFactory(solver)

       

    def _create_model(self):
        m = ConcreteModel(name="market_clearing")
        m.BIDS = RangeSet(0, len(self.bid_static)-1)  # number of bids
        m.ASKS = RangeSet(0, len(self.ask_static)-1)  # number of asks
        m.N= RangeSet(0, self.market_static.n_nodes-1)  # number of nodes
        # --- Static parameters ---
        m.id_bid = Param(m.BIDS, initialize={
            i: self.bid_static[i].agent_id for i in range(len(self.bid_static))
        })
        m.bus_bid = Param(m.BIDS, initialize={
            i: self.bid_static[i].bus for i in range(len(self.bid_static))
        })
        m.cosphi_bid = Param(m.BIDS, initialize={
            i: self.bid_static[i].cosphi for i in range(len(self.bid_static))
        })
        m.id_ask = Param(m.ASKS, initialize={
            j: self.ask_static[j].agent_id for j in range(len(self.ask_static))
        })
        m.bus_ask = Param(m.ASKS, initialize={
            j: self.ask_static[j].bus for j in range(len(self.ask_static))
        })
        m.cosphi_ask = Param(m.ASKS, initialize={
            j: self.ask_static[j].cosphi for j in range(len(self.ask_static))
        })

        m.f_type_bid = Param(m.BIDS, initialize={
            i: self.bid_static[i].bidding_type for i in range(len(self.bid_static))
        })
        m.f_type_ask = Param(m.ASKS, initialize={
            j: self.ask_static[j].asking_type for j in range(len(self.ask_static))
        })

        m.slack_bus = Param(initialize=self.market_static.slack_bus)  # Adjust for 0-based indexing
        m.gridfee_LEC = Param(initialize=self.market_static.gridfee_LEC)
        m.levies_LEC = Param(initialize=self.market_static.levies_LEC)
        m.gridfee_ext = Param(initialize=self.market_static.gridfee_ext)
        m.levies_ext = Param(initialize=self.market_static.levies_ext)
        m.margin_buy = Param(initialize=self.market_static.margin_buy)
        m.margin_sell = Param(initialize=self.market_static.margin_sell)
        
        # --- Dynamic coefficients (mutable) ---
        m.a_bid = Param(m.BIDS, initialize=0.0, mutable=True)
        m.b_bid = Param(m.BIDS, initialize=0.0, mutable=True)
        m.c_bid = Param(m.BIDS, initialize=0.0, mutable=True)

        m.a_ask = Param(m.ASKS, initialize=0.0, mutable=True)
        m.b_ask = Param(m.ASKS, initialize=0.0, mutable=True)
        m.c_ask = Param(m.ASKS, initialize=0.0, mutable=True)

        m.pbid_min = Param(m.BIDS, within=NonNegativeReals, initialize=0.0, mutable=True)
        m.pbid_max = Param(m.BIDS, within=NonNegativeReals, initialize=0.0, mutable=True)
        m.pask_min = Param(m.ASKS, within=NonNegativeReals, initialize=0.0, mutable=True)
        m.pask_max = Param(m.ASKS, within=NonNegativeReals, initialize=0.0, mutable=True)

        
        
        
        # Decision variables
        # ---------------------------
        # Active and reactive quantities for bids (consumption / utility)
        m.p_bid = Var(m.BIDS, within=NonNegativeReals,
                      bounds=lambda m, j: (m.pbid_min[j], m.pbid_max[j]))
        m.q_bid = Var(m.BIDS, within=Reals)
        # Active and reactive quantities for asks (supply / cost)
        m.p_ask = Var(m.ASKS, within=NonNegativeReals,
                      bounds=lambda m, j: (m.pask_min[j], m.pask_max[j]))
        m.q_ask = Var(m.ASKS, within=Reals)

        # Totals (expressions filled later)
        m.P_ext_buy = Var(within=NonNegativeReals)
        m.gridfee_levies_lec = Var(within=NonNegativeReals)
        m.gridfee_levies_ext = Var(within=NonNegativeReals)
        m.netnod_pw_vector = Var(m.N, within=Reals, bounds=(-1000, 1000))
        m.netnod_qw_vector = Var(m.N, within=Reals, bounds=(-1000, 1000))
        m.netnod_pw_vector_pos = Var(m.N, within=NonNegativeReals, bounds=(0, 1000))
        m.netnod_pw_vector_neg = Var(m.N, within=NonNegativeReals, bounds=(0, 1000))
        m.netnod_pw_vector_sc  = Var(m.N, within=NonNegativeReals, bounds=(0, 1000))

        # Optional if used later
        m.P_sc  = Var(within=NonNegativeReals)
        m.P_lec = Var(within=NonNegativeReals)

        # If you need binary indicators (like z_sc, z_lec)
        m.z_sc  = Var(m.N, within=Binary)
        m.z_lec = Var(within=Binary)

        # --- Parameters (placeholders for Gurobi data) ---
        M = 1e6  # big-M constant

                
        
        # Bounds and basic constraints
        # ---------------------------
        def pbid_lb_rule(_m, i):
            return _m.p_bid[i] >= _m.pbid_min[i]
        m.pbid_lb = Constraint(m.BIDS, rule=pbid_lb_rule)

        def pbid_ub_rule(_m, i):
                return _m.p_bid[i] <= _m.pbid_max[i]
        m.pbid_ub = Constraint(m.BIDS, rule=pbid_ub_rule)

        def pask_lb_rule(_m, j):
                return _m.p_ask[j] >= _m.pask_min[j]
        m.pask_lb = Constraint(m.ASKS, rule=pask_lb_rule)

        def pask_ub_rule(_m, j):
                return _m.p_ask[j] <= _m.pask_max[j]
        m.pask_ub = Constraint(m.ASKS, rule=pask_ub_rule)

        # Reactive power modeling :
        #  - q = p * tan(acos(cosphi))   (power factor based)
        def qbid_rule(_m, i):
                if self.bid_static[i].agent_id==self.market_static.slack_agent_id:
                    return Constraint.Skip
                return _m.q_bid[i] == (
                    _m.p_bid[i] * math.tan(math.acos(_m.cosphi_bid[i]))
                    
                )
        m.qbid_def = Constraint(m.BIDS, rule=qbid_rule)

        def qask_rule(_m, j):
                if self.ask_static[j].agent_id==self.market_static.slack_agent_id:
                    return Constraint.Skip
                return _m.q_ask[j] == (
                    _m.p_ask[j] * math.tan(math.acos(_m.cosphi_ask[j]))
                    
                )
        m.qask_def = Constraint(m.ASKS, rule=qask_rule)

        # Nodal Power Equations
        def energy_passive_rule(m, n):
            return m.netnod_qw_vector[n] == (
                -sum(m.q_bid[i] for i in m.BIDS if m.bus_bid[i] == n)
                + sum(m.q_ask[j] for j in m.ASKS if m.bus_ask[j] == n)
            )
        m.nodal_energy_passive_con = Constraint(m.N, rule=energy_passive_rule)


        def energy_active_rule(m, n):
            return m.netnod_pw_vector[n] == (
                -sum(m.p_bid[i] for i in m.BIDS if m.bus_bid[i] == n)
                + sum(m.p_ask[j] for j in m.ASKS if m.bus_ask[j] == n)
            )
        m.nodal_energy_active_con = Constraint(m.N, rule=energy_active_rule)
        
        def reactive_balance_rule(m):
            return sum(m.netnod_qw_vector[n] for n in m.N) == 0
        m.reactive_balance_con = Constraint(rule=reactive_balance_rule)

        def active_balance_rule(m):
            return sum(m.netnod_pw_vector[n] for n in m.N) == 0
        m.active_balance_con = Constraint(rule=active_balance_rule)

        #Self-consumption and LEC modeling
        def sc_upper_cost_rule(m, n):
            return m.netnod_pw_vector_sc[n] <= sum(m.p_ask[j] for j in m.ASKS if m.bus_ask[j] == n)
        m.sc_upper_cost_con = Constraint(m.N, rule=sc_upper_cost_rule)

        def sc_upper_util_rule(m, n):
            return m.netnod_pw_vector_sc[n] <= sum(m.p_bid[i] for i in m.BIDS if m.bus_bid[i] == n)
        m.sc_upper_util_con = Constraint(m.N, rule=sc_upper_util_rule)

        def sc_lower_cost_rule(m, n):
            return (
                m.netnod_pw_vector_sc[n]
                >= sum(m.p_ask[j] for j in m.ASKS if m.bus_ask[j] == n) - M * (1 - m.z_sc[n])
            )
        m.sc_lower_cost_con = Constraint(m.N, rule=sc_lower_cost_rule)

        def sc_lower_util_rule(m, n):
            return (
                m.netnod_pw_vector_sc[n]
                >= sum(m.p_bid[i] for i in m.BIDS if m.bus_bid[i] == n) - M * m.z_sc[n]
            )
        m.sc_lower_util_con = Constraint(m.N, rule=sc_lower_util_rule)

        def lec_pos_rule(m, n):
            return m.netnod_pw_vector_pos[n] == (
                sum(m.p_ask[j] for j in m.ASKS if m.bus_ask[j] == n)
                - m.netnod_pw_vector_sc[n]
            )
        m.lec_pos_con = Constraint(m.N, rule=lec_pos_rule)

        def lec_neg_rule(m, n):
            return m.netnod_pw_vector_neg[n] == (
                sum(m.p_bid[i] for i in m.BIDS if m.bus_bid[i] == n)
                - m.netnod_pw_vector_sc[n]
            )
        m.lec_neg_con = Constraint(m.N, rule=lec_neg_rule)
        
        def self_consumption_total_rule(m):
            return m.P_sc == sum(m.netnod_pw_vector_sc[n] for n in m.N if n != m.slack_bus)
        m.self_consumption_con = Constraint(rule=self_consumption_total_rule)

        def lec_upper_rule_1(m):
            return m.P_lec <= sum(m.netnod_pw_vector_pos[n] for n in m.N if n != m.slack_bus)
        m.lec_upper_1 = Constraint(rule=lec_upper_rule_1)

        def lec_upper_rule_2(m):
            return m.P_lec <= sum(m.netnod_pw_vector_neg[n] for n in m.N if n != m.slack_bus)
        m.lec_upper_2 = Constraint(rule=lec_upper_rule_2)

        def lec_lower_rule_1(m):
            return (
                m.P_lec
                >= sum(m.netnod_pw_vector_neg[n] for n in m.N if n != m.slack_bus)
                - M * (1 - m.z_lec)
            )
        m.lec_lower_1 = Constraint(rule=lec_lower_rule_1)

        def lec_lower_rule_2(m):
            return (
                m.P_lec
                >= sum(m.netnod_pw_vector_pos[n] for n in m.N if n != m.slack_bus)
                - M * m.z_lec
            )
        m.lec_lower_2 = Constraint(rule=lec_lower_rule_2)
        
        # Balance and fee-related expressions

        def pext_rule(_m):
                # external buy = ask at slack bus
                return _m.P_ext_buy == sum(_m.p_ask[j] for j in _m.ASKS if _m.bus_ask[j] == _m.slack_bus)
        m.ext_total = Constraint(rule=pext_rule)

        def fee_lec_rule(_m):
                return _m.gridfee_levies_lec == _m.P_lec * (_m.gridfee_LEC + _m.levies_LEC) / 4.0
        m.fee_lec = Constraint(rule=fee_lec_rule)

        def fee_ext_rule(_m):
                return _m.gridfee_levies_ext == _m.P_ext_buy * (_m.gridfee_ext + _m.levies_ext) / 4.0
        m.fee_ext = Constraint(rule=fee_ext_rule)

        # System active power balance (market clearing): total demand equals total supply
        def power_balance_rule(_m):
                return sum(_m.p_bid[i] for i in _m.BIDS) == sum(_m.p_ask[j] for j in _m.ASKS)
        m.power_balance = Constraint(rule=power_balance_rule)

        # ---------------------------
        # Objective: welfare minimization form
        #  minimize   -sum(V_bid(p)) + sum(C_ask(p)) + fees
        # ---------------------------
        # Linear value/cost forms: V = a + b*p  
        def bid_value_rule(m, i):
            if m.f_type_bid[i] == "lin":
                return m.a_bid[i] + m.b_bid[i] * m.p_bid[i]
            elif m.f_type_bid[i] == "quad":
                return m.c_bid[i] + m.b_bid[i] * m.p_bid[i] + m.a_bid[i] * m.p_bid[i] ** 2
            else:
                return 0
        m.bid_value = Expression(m.BIDS, rule=bid_value_rule)

        def ask_cost_rule(m, j):
            if m.f_type_ask[j] == "lin":
                return m.a_ask[j] + m.b_ask[j] * m.p_ask[j]
            elif m.f_type_ask[j] == "quad":
                return m.c_ask[j] + m.b_ask[j] * m.p_ask[j] + m.a_ask[j] * m.p_ask[j] ** 2
            else:
                return 0
        m.ask_cost = Expression(m.ASKS, rule=ask_cost_rule)

        m.total_value = Expression(expr=sum(m.bid_value[i] for i in m.BIDS))
        m.total_cost = Expression(expr=sum(m.ask_cost[j] for j in m.ASKS))

        m.obj = Objective(
            expr= - m.total_value + m.total_cost + m.gridfee_levies_lec + m.gridfee_levies_ext,
            sense=minimize,
            )
        return m
    def _initialize_result_template(self):
          # --- Build unique agent table ---
        unique_agents = {}
        for a in self.bid_static + self.ask_static:
            if a.agent_id not in unique_agents:
                unique_agents[a.agent_id] = {
                    "Agent ID": a.agent_id,
                    "Node": a.bus,
                    "Agent Type": getattr(a, "agent_type"),
                }
        unique_agents_df = pd.DataFrame(unique_agents.values())

        demand_vector = pd.DataFrame({
            "Agent ID": [a.agent_id for a in self.bid_static],
            "Node": [a.bus for a in self.bid_static],
            "Agent Type": [a.agent_type for a in self.bid_static],
            "Energy [kWh]": 0.0,
            "Price [€]": 0.0,
            "relative Price [€/kWh]": 0.0
        })
        supply_vector = pd.DataFrame({
            "Agent ID": [a.agent_id for a in self.ask_static],
            "Node": [a.bus for a in self.ask_static],
            "Agent Type": [a.agent_type for a in self.ask_static],
            "Energy [kWh]": 0.0,
            "Price [€]": 0.0,
            "relative Price [€/kWh]": 0.0
        })
     
        node_results = pd.DataFrame({
            "Node": [n for n in range(self.market_static.n_nodes)],
            "HN Self-Consumption [kWh]": 0.0,
            "HN Energy LEC [kWh]": 0.0,
            "HN Energy External [kWh]": 0.0,
            "Revenue energy [€]": 0.0,
            "Gridfees and Levies [€]": 0.0,
        })
        agent_result_vector = unique_agents_df.assign(
        **{
            "Energy sold [kWh]": 0.0,
            "ask price [€/kWh]": 0.0,
            "Energy bought [kWh]": 0.0,
            "bid price [€/kWh]": 0.0,
            "HN Self-Consumption [kWh]":0.0,
            "Energy LEC [kWh]":0.0,
            "Energy External [kWh]":0.0,
            "Revenue Energy LEC [€]": 0.0,
            "Revenue Energy External [€]": 0.0,
            "Fees and Levies LEC [€]": 0.0,
            "Fees and Levies External [€]": 0.0,
            "LEC_Participation": True  # Default to True, will be updated for non-LEC agents                                   
        }
        )

        input_grid_columns = [
        "time",
        "Temperature [°C]",
        "Market Price [€/kWh]",
        "DA Price [€/kWh]",
        "Social Welfare [€]",
        "Voltage deviation max [p.u.]",
        "Current loading max [%]",
        "Transformer loading [kVA]"
        ]
        input_grid = pd.DataFrame(columns=input_grid_columns)
        input_grid.loc[0] = [None] * len(input_grid_columns)
        

        return MarketResults(
        demand=demand_vector,
        supply=supply_vector,
        Node_results=node_results,
        Input_Grid=input_grid,
        agents=agent_result_vector,
        )
    
    def compute_allocation(self,
                           bid_round_data: list[Agent_Bid_RoundData],
                           ask_round_data: list[Agent_Ask_RoundData],
                           ):
        m=self.model
        
        # Get agent IDs from round data
        bid_agent_ids = {bid.agent_id for bid in bid_round_data}
        ask_agent_ids = {ask.agent_id for ask in ask_round_data}
        
        # Update bids that have data
        for bid in bid_round_data:
            i = next(
                    (i for i in m.BIDS if m.id_bid[i] == bid.agent_id),
                    None
                )
            m.a_bid[i].set_value(bid.f_coef[0])
            m.b_bid[i].set_value(bid.f_coef[1] if len(bid.f_coef) > 1 else 0.0)
            m.c_bid[i].set_value(bid.f_coef[2] if len(bid.f_coef) > 2 else 0.0)
            m.pbid_min[i].set_value(bid.p_min)
            m.pbid_max[i].set_value(bid.p_max)
            # Explicitly update variable bounds
            m.p_bid[i].setlb(bid.p_min)
            m.p_bid[i].setub(bid.p_max)
        
        # Zero out bids not in round data
        for i in m.BIDS:
            if m.id_bid[i] not in bid_agent_ids:
                m.a_bid[i].set_value(0.0)
                m.b_bid[i].set_value(0.0)
                m.c_bid[i].set_value(0.0)
                m.pbid_min[i].set_value(0.0)
                m.pbid_max[i].set_value(0.0)
                # Explicitly update variable bounds to zero
                m.p_bid[i].setlb(0.0)
                m.p_bid[i].setub(0.0)

        # Update asks that have data
        for ask in ask_round_data:
            j = next(
                    (j for j in m.ASKS if m.id_ask[j] == ask.agent_id),
                    None
                )
            if j is None:
                print(f"⚠️ Warning: No index for agent_id={ask.agent_id}")
                continue
            m.a_ask[j].set_value(ask.f_coef[0])
            m.b_ask[j].set_value(ask.f_coef[1] if len(ask.f_coef) > 1 else 0.0)
            m.c_ask[j].set_value(ask.f_coef[2] if len(ask.f_coef) > 2 else 0.0)
            m.pask_min[j].set_value(ask.p_min)
            m.pask_max[j].set_value(ask.p_max)
            m.p_ask[j].setlb(ask.p_min)
            m.p_ask[j].setub(ask.p_max)
        for j in m.ASKS:
            if m.id_ask[j] not in ask_agent_ids:
                m.a_ask[j].set_value(0.0)
                m.b_ask[j].set_value(0.0)
                m.c_ask[j].set_value(0.0)
                m.pask_min[j].set_value(0.0)
                m.pask_max[j].set_value(0.0)
                m.p_ask[j].setlb(0.0)
                m.p_ask[j].setub(0.0)
                   

        solver = self._solver_instance
        self.result = solver.solve(m, tee=False)

        if hasattr(solver, '_solver_model') and solver._solver_model is not None:
            try:
                solver._solver_model.dispose()
            except Exception:
                pass
            solver._solver_model = None
    def process_pyomo_results(self, market_variable:Market_VariableData, new_non_LEC_data):
        res=copy.deepcopy(self.results_template) 
        m = self.model
        sref = self.market_static.s_ref
        timestep_h = self.market_static.timestep / 3600.0
        self.market_variable=market_variable
        self.new_non_LEC_data = new_non_LEC_data

        bids = np.array([value(m.p_bid[i]) for i in m.BIDS])
        bid_prices = np.array([value(m.bid_value[i]) for i in m.BIDS])
        res.demand.loc[:, "Energy [kWh]"] = bids * sref * timestep_h
        res.demand.loc[:, "Price [€]"] = bid_prices
        res.demand.loc[:, "relative Price [€/kWh]"] = bid_prices / np.maximum(res.demand["Energy [kWh]"], 1e-9)

        asks = np.array([value(m.p_ask[j]) for j in m.ASKS])
        ask_prices = np.array([value(m.ask_cost[j]) for j in m.ASKS])
        res.supply.loc[:, "Energy [kWh]"] = asks * sref * timestep_h
        res.supply.loc[:, "Price [€]"] = ask_prices
        res.supply.loc[:, "relative Price [€/kWh]"] = ask_prices / np.maximum(res.supply["Energy [kWh]"], 1e-9)

        # -------------------------------------------------------------------------
        #extract network-level data (if modeled in Pyomo)
        # -------------------------------------------------------------------------

        netnod_pw_vector = np.array([value(self.model.netnod_pw_vector[n]) for n in self.model.netnod_pw_vector])
        netnod_qw_vector = np.array([value(self.model.netnod_qw_vector[n]) for n in self.model.netnod_qw_vector])
        netnod_pw_vector_pos = np.array([value(self.model.netnod_pw_vector_pos[n]) for n in self.model.netnod_pw_vector])
        netnod_pw_vector_neg = np.array([value(self.model.netnod_pw_vector_neg[n]) for n in self.model.netnod_pw_vector])
        neg_sum = float(np.sum([netnod_pw_vector_neg[i] for i in range(self.market_static.n_nodes) if i != self.market_static.slack_bus]))
        pos_sum = float(np.sum([netnod_pw_vector_pos[i] for i in range(self.market_static.n_nodes) if i != self.market_static.slack_bus]))
        P_lec   = float(value(self.model.P_lec))
        pw_sc   = np.array([value(self.model.netnod_pw_vector_sc[n])   for n in range(self.market_static.n_nodes)])
        # --- 2) Split pw into LEC part and external part (vector) ---
        # Equivalent to your MVar logic
        if neg_sum != 0 and pos_sum != 0:
            pw_LEC = []
            for n in range(self.market_static.n_nodes):
                if n == self.market_static.slack_bus:
                    pw_LEC.append(0.0)
                elif netnod_pw_vector[n] <= 0:
                    pw_LEC.append(P_lec / neg_sum * netnod_pw_vector[n])  # pw[n] is negative/zero
                else:  # pw[n] >= 0
                    pw_LEC.append(P_lec / pos_sum * netnod_pw_vector[n])  # pw[n] is positive/zero
            pw_LEC = np.array(pw_LEC, dtype=float)
        else:
            # shape like netnod_pw_vector_neg; zeros
            pw_LEC = np.zeros_like(netnod_pw_vector_neg, dtype=float)
        pw_ext = netnod_pw_vector - pw_LEC

        # Totals (divide by 2 like in your code)
        pw_ext_total = float(np.sum(np.abs(pw_ext)) / 2.0)
        pw_LEC_total = float(np.sum(np.abs(pw_LEC)) / 2.0)

        # Shares
        pw_ext_share = (pw_ext / pw_ext_total) if pw_ext_total != 0 else np.zeros_like(pw_ext)
        pw_LEC_share = (pw_LEC / pw_LEC_total) if pw_LEC_total != 0 else np.zeros_like(pw_LEC)

        # --- 3) Grid fees / levies allocation (node level) ---
        # Expect these as scalar Vars or Params in Pyomo (adjust if Param):
        gridfee_levies_lec = float(value(self.model.gridfee_levies_lec))
        gridfee_levies_ext = float(value(self.model.gridfee_levies_ext))
        # Allocate only to importing nodes (negative share)
        gridfee_levies_lec_vec = gridfee_levies_lec * np.where(pw_LEC_share < 0, -pw_LEC_share, 0.0) / 100.0
        gridfee_levies_ext_vec = gridfee_levies_ext * np.where(pw_ext_share < 0, -pw_ext_share, 0.0) / 100.0
        gridfee_levies_total   = gridfee_levies_lec_vec + gridfee_levies_ext_vec

        # €/kWh per node (protect divide-by-zero; 1/4 came from your 15-min to hours)
        gridfee_levies_total_kWh = []
        for n in range(self.market_static.n_nodes):
            if netnod_pw_vector[n] != 0:
                gridfee_levies_total_kWh.append(abs(round(gridfee_levies_total[n] / (netnod_pw_vector[n] / 4.0), 3)))
            else:
                gridfee_levies_total_kWh.append(0.0)

        # --- 4) Agent-level split (SC / LEC / External), Mesa-free ---
        # We distribute each node’s SC/LEC/EXT to agents by nodal share of energy.
        # Build outputs:
        agent_pw_sc  = pd.DataFrame(columns=["Agent ID", "HN Self-Consumption [kWh]"])
        agent_pw_lec = pd.DataFrame(columns=["Agent ID", "Energy LEC [kWh]"])
        agent_pw_ext = pd.DataFrame(columns=["Agent ID", "Energy External [kWh]"])

        # Pre-sum nodal totals from demand/supply vectors
        # (They already contain Node & Energy [kWh] after your scaling)
        demand_node_sum = res.demand.groupby("Node")["Energy [kWh]"].sum().to_dict()
        supply_node_sum = res.supply.groupby("Node")["Energy [kWh]"].sum().to_dict()

        # Convert pw_* (which are in model units) to kWh over the timestep, like you did:
        # your old code: /4 * 100 → that combined your timestep and sref; keep that mapping:
        # Here: kWh_node = pw_node * (timestep_hours) * sref  (adjust to match your scaling exactly)
        # To mimic the old values closely, we’ll keep /4*100 formulation:
        node_SC_kWh  = pw_sc  * (self.market_static.timestep/(60*60)) * (self.market_static.s_ref)
        node_EXT_kWh = pw_ext * (self.market_static.timestep/(60*60)) * (self.market_static.s_ref)
        node_LEC_kWh = pw_LEC * (self.market_static.timestep/(60*60)) * (self.market_static.s_ref)

        # Reset all numeric energy and price columns to 0 before writing new values
        res.agents["Energy bought [kWh]"] = 0.0
        res.agents["Energy sold [kWh]"] = 0.0
        res.agents["bid price [€/kWh]"] = 0.0
        res.agents["ask price [€/kWh]"] = 0.0
        res.agents["Energy LEC [kWh]"] = 0.0
        res.agents["Energy External [kWh]"] = 0.0
        res.agents["HN Self-Consumption [kWh]"] = 0.0
        
        # Bidders (consumers): split by demand shares at same node
        for _, row in res.demand.iterrows():
            aid  = int(row["Agent ID"])
            node = int(row["Node"])
            e_kWh = float(row["Energy [kWh]"])
            if abs(e_kWh)>1e-6:
                if demand_node_sum.get(node, 0.0) > 0:
                    share = e_kWh / demand_node_sum[node]
                    a_sc  = node_SC_kWh[node]  * share
                    a_ext = max(-node_EXT_kWh[node], 0.0) * share
                    a_lec = max(-node_LEC_kWh[node], 0.0) * share
                else:
                    a_sc = a_ext = a_lec = 0.0

                agent_pw_sc.loc[len(agent_pw_sc)]   = [aid,  a_sc]
                agent_pw_ext.loc[len(agent_pw_ext)] = [aid,  a_ext]
                agent_pw_lec.loc[len(agent_pw_lec)] = [aid,  a_lec]
                if aid in res.agents["Agent ID"].values:
                    res.agents.loc[res.agents["Agent ID"] == aid, "Energy bought [kWh]"] = e_kWh
                    res.agents.loc[res.agents["Agent ID"] == aid, "bid price [€/kWh]"] = row["relative Price [€/kWh]"]
                    res.agents.loc[res.agents["Agent ID"] == aid, "Energy LEC [kWh]"] = a_lec
                    res.agents.loc[res.agents["Agent ID"] == aid, "Energy External [kWh]"] = a_ext
                    res.agents.loc[res.agents["Agent ID"] == aid, "HN Self-Consumption [kWh]"] = a_sc


        # Suppliers: negative sign (export perspective)
        for _, row in res.supply.iterrows():
            aid  = int(row["Agent ID"])
            node = int(row["Node"])
            e_kWh = float(row["Energy [kWh]"])
            if abs(e_kWh)>1e-6:
                if supply_node_sum.get(node) > 0:
                    share = e_kWh / supply_node_sum[node]
                    a_sc  = -node_SC_kWh[node]  * share
                    a_ext = -np.maximum(node_EXT_kWh[node], 0.0) * share
                    a_lec = -np.maximum(node_LEC_kWh[node], 0.0) * share
                else:
                    a_sc = a_ext = a_lec = 0.0
                
                agent_pw_sc.loc[len(agent_pw_sc)]   = [aid,  a_sc]
                agent_pw_ext.loc[len(agent_pw_ext)] = [aid,  a_ext]
                agent_pw_lec.loc[len(agent_pw_lec)] = [aid,  a_lec]
                if aid in res.agents["Agent ID"].values:
                    res.agents.loc[res.agents["Agent ID"] == aid, "Energy sold [kWh]"] = e_kWh
                    res.agents.loc[res.agents["Agent ID"] == aid, "ask price [€/kWh]"] = row["relative Price [€/kWh]"]
                    res.agents.loc[res.agents["Agent ID"] == aid, "Energy LEC [kWh]"] = a_lec
                    res.agents.loc[res.agents["Agent ID"] == aid, "Energy External [kWh]"] = a_ext
                    res.agents.loc[res.agents["Agent ID"] == aid, "HN Self-Consumption [kWh]"] = a_sc

        # Calculate slack price following Gurobi logic
        slack_bus = self.market_static.slack_bus
        slack_agent_id = self.market_static.slack_agent_id
        
        # Find slack agent in bids and asks
        slack_bid_idx = next((i for i in m.BIDS if m.id_bid[i] == slack_agent_id), None)
        slack_ask_idx = next((j for j in m.ASKS if m.id_ask[j] == slack_agent_id), None)
        
        if slack_bid_idx is not None and slack_ask_idx is not None:
            # Energy and price from both bid and ask
            energy = value(m.p_ask[slack_ask_idx]) / 4 + value(m.p_bid[slack_bid_idx]) / 4
            price = value(m.ask_cost[slack_ask_idx]) + value(m.bid_value[slack_bid_idx])
            
            if energy > 0:
                slack_price = price / energy / 100
            else:
                # Fallback: use ask price coefficient
                slack_price = value(m.b_ask[slack_ask_idx]) * 4 / 100- self.market_static.margin_sell/100
        else:
            # Fallback if slack agent not found
            slack_price = 0.0

        # Calculate revenue and fees for LEC agents
        res.agents["Revenue Energy LEC [€]"] = slack_price * res.agents["Energy LEC [kWh]"]
        res.agents["Revenue Energy External [€]"] = slack_price * res.agents["Energy External [kWh]"]
        res.agents["Fees and Levies LEC [€]"] = (self.market_static.gridfee_LEC + self.market_static.levies_LEC) / 100 * res.agents["Energy LEC [kWh]"].clip(lower=0)
        res.agents["Fees and Levies External [€]"] = (self.market_static.gridfee_ext + self.market_static.levies_ext) / 100 * res.agents["Energy External [kWh]"].clip(lower=0)
        
        # Add non LEC agents
        for nd in self.new_non_LEC_data:
            agent_id = nd['agent_id']
            bus = nd['bus']
            p_buy = nd.get('p_buy', 0)* (self.market_static.timestep/(60*60)) * (self.market_static.s_ref)
            p_sell = nd.get('p_sell', 0)* (self.market_static.timestep/(60*60)) * (self.market_static.s_ref)
            p_out = nd.get('p_out', 0)* (self.market_static.timestep/(60*60)) * (self.market_static.s_ref)
            HN_SC = nd.get('HN_self_consumption', 0)* (self.market_static.timestep/(60*60)) * (self.market_static.s_ref)
            typ=nd.get('type','non_LEC')
            
            # Calculate revenue and fees for non-LEC agents
            revenue_external = max((self.market_variable[0].ext_price - self.market_static.margin_sell)/100 * p_sell, 
                                   (self.market_variable[0].ext_price + self.market_static.margin_buy)/100 * p_buy)
            fees_external = (self.market_static.gridfee_ext + self.market_static.levies_ext) / 100 * p_buy

            new_row = {
                    "Agent ID": agent_id,
                    "Node": bus,
                    "Agent Type": typ,
                    "Energy sold [kWh]": p_sell,
                    "ask price [€/kWh]": 0.0,
                    "Energy bought [kWh]": p_buy,
                    "bid price [€/kWh]": 0.0,
                    "HN Self-Consumption [kWh]": HN_SC,
                    "Energy LEC [kWh]": 0.0,
                    "Energy External [kWh]": p_sell if p_sell > 0 else p_buy,
                    "Revenue Energy LEC [€]": 0.0,
                    "Revenue Energy External [€]": revenue_external,
                    "Fees and Levies LEC [€]": 0.0,
                    "Fees and Levies External [€]": fees_external,
                    "LEC_Participation": False
                }
            res.agents = pd.concat([res.agents, pd.DataFrame([new_row])], ignore_index=True)

        # Sort agents by Node
        res.agents = res.agents.sort_values(by="Node").reset_index(drop=True)
        
        # -------------------------------------------------------------------------
        # 4. Run power flow for grid results
        # -------------------------------------------------------------------------
        # Create node-summed power vector from agent results (sell - buy)
        agent_pw_by_node = res.agents.groupby("Node").apply(
            lambda x: (x["Energy sold [kWh]"].sum() - x["Energy bought [kWh]"].sum()) / (timestep_h * sref)
        ).reindex(range(self.market_static.n_nodes), fill_value=0.0).values
        
        net2 = copy.deepcopy(self.grid)
        for n, (p, q) in enumerate(zip(agent_pw_by_node, netnod_qw_vector)):
            if n!= slack_bus:
                if p < 0:
                    pp.create_load(net2, n, p_mw=-p * self.market_static.s_ref / 1000, q_mvar=-q * self.market_static.s_ref / 1000)
                elif p > 0:
                    pp.create_sgen(net2, n, p_mw=p * self.market_static.s_ref / 1000, q_mvar=q * self.market_static.s_ref / 1000)
        try:
            pp.runpp(net2)
        except:
            print("error on powerflow")
        v_d_max = abs(net2.res_bus.vm_pu - 1).max() * 100
        I_d_max = net2.res_line.loading_percent.max()
        trafo_load = float(np.hypot(net2.res_ext_grid.p_mw, net2.res_ext_grid.q_mvar).max() * 1000)

        # -------------------------------------------------------------------------
        # 5. Node results
        # -------------------------------------------------------------------------
        # Overwrite numeric values efficiently (vectorized)
        node_df = res.Node_results.copy()
        node_df.loc[:, "HN Self-Consumption [kWh]"] = node_SC_kWh
        node_df.loc[:, "HN Energy LEC [kWh]"] = node_LEC_kWh
        node_df.loc[:, "HN Energy External [kWh]"] = node_EXT_kWh
        gridfees = np.array(gridfee_levies_total)*sref if "gridfee_levies_total" in locals() else np.zeros(len(node_df))
        node_df.loc[:, "Revenue energy [€]"] = slack_price * (
            node_df["HN Energy LEC [kWh]"] + node_df["HN Energy External [kWh]"]
        )
        node_df.loc[:, "Gridfees and Levies [€]"] = gridfees
        
        # Set LEC_Participation: False if node has non-LEC agents, True otherwise
        non_lec_buses = set(nd['bus'] for nd in self.new_non_LEC_data)
        node_df.loc[:, "LEC_Participation"] = node_df["Node"].apply(lambda x: False if x in non_lec_buses else True)
        
        # Update non-LEC bus rows with aggregated agent values
        for bus in non_lec_buses:
            # Get all agents on this bus from res.agents
            bus_agents = res.agents[res.agents["Node"] == bus]
            
            if len(bus_agents) > 0:
                # Sum values and divide by 2 for HN Self-Consumption
                total_hn_sc = abs(bus_agents["HN Self-Consumption [kWh]"]).sum() / 2.0
                total_energy_ext = bus_agents["Energy External [kWh]"].sum() 
                total_revenue = bus_agents["Revenue Energy External [€]"].sum() 
                
                # Calculate fees: only for demand (Energy bought > 0)
                gridfee_ext = self.market_static.gridfee_ext
                levies_ext = self.market_static.levies_ext
                total_fees = ((gridfee_ext + levies_ext) / 100 * bus_agents["Energy bought [kWh]"]).sum()
                
                # Update node_df for this bus
                node_df.loc[node_df["Node"] == bus, "HN Self-Consumption [kWh]"] = total_hn_sc
                node_df.loc[node_df["Node"] == bus, "HN Energy External [kWh]"] = total_energy_ext
                node_df.loc[node_df["Node"] == bus, "Revenue energy [€]"] = total_revenue
                node_df.loc[node_df["Node"] == bus, "Gridfees and Levies [€]"] = total_fees
        
        res.Node_results = node_df

        # -------------------------------------------------------------------------
        # 6. External results summary
        # -------------------------------------------------------------------------
    

        input = res.Input_Grid.copy()
        input.loc[0, "Social Welfare [€]"] = -value(self.model.obj)
        input.loc[0, "Voltage deviation max [p.u.]"] = v_d_max
        input.loc[0, "Current loading max [%]"] = I_d_max
        input.loc[0, "Transformer loading [kVA]"] = trafo_load
        input.loc[0, "time"] = self.market_variable[0].date
        input.loc[0, "Temperature [°C]"] = self.market_variable[0].temperature
        input.loc[0, "Market Price [€/kWh]"] = slack_price
        input.loc[0, "DA Price [€/kWh]"] = self.market_variable[0].ext_price/100
        res.Input_Grid = input

        # -------------------------------------------------------------------------
        # 7. EV, Storage, and Heatpump Results
        # -------------------------------------------------------------------------
        # Extract EV results from agent data
        ev_agents = res.agents[res.agents["Agent Type"] == "EV"].copy()
        if len(ev_agents) > 0:
            EV_result_vector = pd.DataFrame({
                "Agent ID": ev_agents["Agent ID"],
                "SOC [kWh]": 0.0,  # To be filled by model.agents if available
                "SOC_rel": 0.0,
                "Status": 0,
                "Distance [km]": 0.0,
                "Acc. Energy outside LEC [kWh]": 0.0,
                "Acc. Cost outside LEC [€]": 0.0,
            })
            # If agent state data is available in new_non_LEC_data or market_variable, populate here
            for nd in self.new_non_LEC_data:
                if nd.get('type') == 'EV' and nd['agent_id'] in EV_result_vector["Agent ID"].values:
                    idx = EV_result_vector[EV_result_vector["Agent ID"] == nd['agent_id']].index[0]
                    EV_result_vector.loc[idx, "SOC [kWh]"] = nd.get('soc_kwh', 0.0)
                    EV_result_vector.loc[idx, "SOC_rel"] = nd.get('soc_rel', 0.0)
                    EV_result_vector.loc[idx, "Status"] = nd.get('status', 0)
                    EV_result_vector.loc[idx, "Distance [km]"] = nd.get('distance_km', 0.0)
                    EV_result_vector.loc[idx, "Acc. Energy outside LEC [kWh]"] = nd.get('acc_energy_outside', 0.0)
                    EV_result_vector.loc[idx, "Acc. Cost outside LEC [€]"] = nd.get('acc_cost_outside', 0.0)
        else:
            EV_result_vector = pd.DataFrame(columns=[
                "Agent ID", "SOC [kWh]", "SOC_rel", "Status", "Distance [km]",
                "Acc. Energy outside LEC [kWh]", "Acc. Cost outside LEC [€]"
            ])

        # Extract Heatpump results
        hp_agents = res.agents[res.agents["Agent Type"] == "heatpump"].copy()
        if len(hp_agents) > 0:
            heatpump_result_vector = pd.DataFrame({
                "Agent ID": hp_agents["Agent ID"],
                "Temperature Indoor [°C]": 0.0,
                "SOC_rel": 0.0
            })
            for nd in self.new_non_LEC_data:
                if nd.get('type') == 'heatpump' and nd['agent_id'] in heatpump_result_vector["Agent ID"].values:
                    idx = heatpump_result_vector[heatpump_result_vector["Agent ID"] == nd['agent_id']].index[0]
                    heatpump_result_vector.loc[idx, "Temperature Indoor [°C]"] = nd.get('temperature', 0.0)
                    heatpump_result_vector.loc[idx, "SOC_rel"] = nd.get('soc_rel', 0.0)
        else:
            heatpump_result_vector = pd.DataFrame(columns=["Agent ID", "Temperature Indoor [°C]", "SOC_rel"])

        # Extract Storage results
        storage_agents = res.agents[res.agents["Agent Type"] == "storage"].copy()
        if len(storage_agents) > 0:
            storage_result_vector = pd.DataFrame({
                "Agent ID": storage_agents["Agent ID"],
                "SOC_rel": 0.0
            })
            for nd in self.new_non_LEC_data:
                if nd.get('type') == 'storage' and nd['agent_id'] in storage_result_vector["Agent ID"].values:
                    idx = storage_result_vector[storage_result_vector["Agent ID"] == nd['agent_id']].index[0]
                    storage_result_vector.loc[idx, "SOC_rel"] = nd.get('soc_rel', 0.0)
        else:
            storage_result_vector = pd.DataFrame(columns=["Agent ID", "SOC_rel"])

        # Add to results
        res.EV = EV_result_vector
        res.Heatpump = heatpump_result_vector
        res.storage = storage_result_vector

        res = {
            k: (v.round(4) if isinstance(v, pd.DataFrame) else v)
            for k, v in res.__dict__.items()
        }

        # -------------------------------------------------------------------------
        # 8. Package results
        # -------------------------------------------------------------------------
        
        return res.copy()
     


# def compute_allocation(model):
#     #try:
#         M=1e6     
#         model.agents_ask= [a.ask for a in model.agents if a.ask!=0]  # Vector(min energy, max energy, price/kWh)
#         model.agents_bid= [a.bid for a in model.agents if a.bid!=0]

#         nn=len(model.grid.bus)
#         na=bus_bid.shape[0]
#         nb=bus_ask.shape[0]
#         z_sc = opt_all.addMVar(nn,vtype=GRB.BINARY, name="z_sc")
#         z_lec=opt_all.addVar(vtype=GRB.BINARY, name="z_lec")

#         "Set up utility vector as optimizaion input"
#         lb2=np.array([[0]*n_bids, [0]*n_bids, [model.agents_bid[xx][0] for xx in range(0,n_bids)],[-10]*n_bids ]).T #lower bounds for agent  id, bid_price, bid quantity
#         ub2=np.array([[len(model.agents)]*n_bids, [200]*n_bids, [model.agents_bid[xx][1] for xx in range(0,n_bids)],[10]*n_bids ]).T #upper bounds for agent  id, bid_price, bid quantity
#         utility_vector= opt_all.addMVar((n_bids,4),lb=lb2, ub=ub2) # Vector(agent_id, bided Price,bided Active Power, Reactive Power)
#         opt_all.addConstr(utility_vector[:,0]==id_bid)
#         cosphi=[a.cosphi for a in model.agents if a.bid!=0]
#         for xx in range(0, n_bids):
#             if model.agents_bid[xx][3]=="lin":
#                 opt_all.addConstr(utility_vector[xx,1]==model.agents_bid[xx][2](utility_vector[xx,2]),name= "bid Value %d" % xx)
#             if model.agents_bid[xx][3]=="quad":               
#                 opt_all.addQConstr(utility_vector[xx,1].item()<=model.agents_bid[xx][2](utility_vector[xx,2].item()),name= "bid Value %d" % xx)
#             if np.isnan(cosphi[xx]):
#                     opt_all.addConstr(utility_vector[xx,3]==0, name= "Reactive Energy bid %d" % xx)
#             else:
#                 opt_all.addConstr(utility_vector[xx,3]==utility_vector[xx,2]* math.tan(math.acos(cosphi[xx])), name= "Reactive Energy bid %d" % xx)
#         opt_all.addConstr([utility_vector[x] for x in range(n_bids) if id_bid[x]==model.id_ext_grid][0][3]==0)
        
        
#         "Set up cost vector as optimization input"
#         lb1=np.array([[0]*n_asks, [-200]*n_asks, [model.agents_ask[xx][0] for xx in range(0,n_asks)],[-100]*n_asks ]).T #lower bounds for agent  id, ask_price, ask quantity
#         ub1=np.array([[100]*n_asks, [200]*n_asks, [model.agents_ask[xx][1] for xx in range(0,n_asks)],[100]*n_asks ]).T #upper bounds for agent  id, ask_price, ask quantity
#         cost_vector= opt_all.addMVar((n_asks,4),lb=lb1, ub=ub1)  # Vector(agent_id, ask Price, ask Active Power, Reactive Power)
#         opt_all.addConstr(cost_vector[:,0]==id_ask)
#         cosphi=[a.cosphi for a in model.agents if a.ask!=0]
#         for xx in range(0, n_asks):
#             if model.agents_ask[xx][3]=="lin":
#                 opt_all.addConstr(cost_vector[xx,1]==model.agents_ask[xx][2](cost_vector[xx,2]), name= "ask value %d" % xx)
#             if model.agents_ask[xx][3]=="quad":               
#                 opt_all.addQConstr(cost_vector[xx,1].item()>=model.agents_ask[xx][2](cost_vector[xx,2].item()),name= "ask value %d" % xx)
#             if id_ask[xx]!=model.id_ext_grid:
#                 if np.isnan(cosphi[xx]):
#                     opt_all.addConstr(cost_vector[xx,3]==0, name= "Reactive Energy ask %d" % xx)
#                 else:
#                     opt_all.addConstr(cost_vector[xx,3]==cost_vector[xx,2]* math.tan(math.acos(cosphi[xx])), name= "Reactive Energy ask %d" % xx)
    
        
#         "Load Flow Calculation and Constraints"
        
#         netnod_pw_vector=opt_all.addMVar((nn), lb=-1000, ub=1000) #Net Nodal Active Power Vector
#         netnod_qw_vector=opt_all.addMVar((nn), lb=-1000, ub=1000) #Net Nodal Reactive Power Vector (==0 except for slack node)
#         netnod_pw_vector_pos = opt_all.addMVar((nn), lb=0, ub=1000)
#         netnod_pw_vector_neg = opt_all.addMVar((nn), lb=0, ub=1000)
#         netnod_pw_vector_sc = opt_all.addMVar((nn), lb=0, ub=1000)

        
#         for n in range(0,nn):          
#             opt_all.addConstr(netnod_qw_vector[n]==-sum(utility_vector[i,3] for i in range(0,na) if bus_bid[i]==n)+sum(cost_vector[i,3] for i in range(0,nb) if bus_ask[i]==n), name="Energy_passive %d" % n)            
#             opt_all.addConstr(netnod_pw_vector[n]==(-sum(utility_vector[i,2] for i in range(0,na) if bus_bid[i]==n)+sum(cost_vector[i,2] for i in range(0,nb) if bus_ask[i]==n)),name="Energy_active %d" % n)
#         #Energy Self-Supply (p_sc=min(utility_vector, cost_vector))
#             opt_all.addConstr(netnod_pw_vector_sc[n] <= sum(cost_vector[i,2] for i in range(0,nb) if bus_ask[i]==n))
#             opt_all.addConstr(netnod_pw_vector_sc[n] <= sum(utility_vector[i,2] for i in range(0,na) if bus_bid[i]==n))
#             opt_all.addConstr(netnod_pw_vector_sc[n] >= sum(cost_vector[i,2] for i in range(0,nb) if bus_ask[i]==n)-M*(1-z_sc[n]))
#             opt_all.addConstr(netnod_pw_vector_sc[n] >= sum(utility_vector[i,2] for i in range(0,na) if bus_bid[i]==n)-M*z_sc[n])
#         #Energy Traded Within LEC()        
#             opt_all.addConstr(netnod_pw_vector_pos[n] ==sum(cost_vector[i,2] for i in range(0,nb) if bus_ask[i]==n)-   netnod_pw_vector_sc[n])
#             opt_all.addConstr(netnod_pw_vector_neg[n] ==sum(utility_vector[i,2] for i in range(0,na) if bus_bid[i]==n)-   netnod_pw_vector_sc[n])
 
#         P_sc = opt_all.addVar(name="P Self_Consumption")
#         opt_all.addConstr(P_sc== sum(netnod_pw_vector_sc[n] for n in range(0,nn) if n!= model.slack))
 
#         P_lec = opt_all.addVar(name="P_lec")      
#         opt_all.addConstr(P_lec<=sum([netnod_pw_vector_pos[n] for n in range(nn) if n!=model.slack]))
#         opt_all.addConstr(P_lec<=sum([netnod_pw_vector_neg[n] for n in range(nn) if n!=model.slack]))
#         opt_all.addConstr(P_lec>=sum([netnod_pw_vector_neg[n] for n in range(nn) if n!=model.slack])-M*(1-z_lec))
#         opt_all.addConstr(P_lec>=sum([netnod_pw_vector_pos[n] for n in range(nn) if n!=model.slack])-M*z_lec)

#         opt_all.addConstr(gp.quicksum(netnod_qw_vector)==0)
#         opt_all.addConstr(gp.quicksum(netnod_pw_vector)==0)

        
#         "Allocation rule, welfare maximisation"
#         gridfee_levies_lec = opt_all.addVar()
#         gridfee_levies_ext = opt_all.addVar()
#         P_ext_buy= opt_all.addVar()
#         opt_all.addConstr(P_ext_buy==sum([cost_vector[n,2] for n  in range(0,nb) if bus_ask[n]==model.slack]))
#         opt_all.addConstr(gridfee_levies_lec==P_lec *(model.gridfee_LEC+model.levies_LEC)/4)
#         opt_all.addConstr(gridfee_levies_ext==P_ext_buy*(model.gridfee_ext+model.levies_ext)/4)
#         opt_all.setObjective(((utility_vector[:,1].sum())*(-1)+cost_vector[:,1].sum() + gridfee_levies_lec + gridfee_levies_ext),GRB.MINIMIZE)
             
        
#         opt_all.optimize()
#         if opt_all.Status == GRB.INFEASIBLE:
#             opt_all.computeIIS()
#             print('\nThe following constraints and variables are in the IIS:')
#             for c in opt_all.getConstrs():
#                 if c.IISConstr: print(f'\t{c.constrname}: {opt_all.getRow(c)} {c.Sense} {c.RHS}')

#             for v in opt_all.getVars():
#                 if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
#                 if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
#             print(f"Error, Market Dispatch not feasible, date {model.current_date}")
#             sys.exit()
        
#         "Give back results in a readable way"
#         # Aux_arr
#         ID=[a.unique_id for a in model.agents[:]]
#         node=[a.bus if not np.isnan(a.bus) else "Outside LEM" for a in model.agents[:]]
#         type1=[a.__class__.__name__ for a in model.agents[:]]
#         aux_arr=pd.DataFrame({
#         'Agent ID': ID,
#         'Node': node,
#         'Agent Type': type1})
        
#           # Demand Vector   
#         demand_vector=np.insert(utility_vector.X,1,0, axis=1) 
#         for a in range(0, len(demand_vector)):
#             demand_vector[a,3]=demand_vector[a,3]*(model.timestep.seconds/(60*60))*model.sref
#             demand_vector[a,4]=demand_vector[a,4]*(model.timestep.seconds/(60*60))*model.sref
#             if demand_vector[a,3]!=0: #if no energy is brought - all row elemnts are 0
#                 demand_vector[a,1]=demand_vector[a,2]/demand_vector[a,3]  #calculate price/kWh
#         demand_vector=demand_vector[demand_vector[:,1].argsort()[::-1]] #sort by price/kWh in ascending order
#         demand_vector=pd.DataFrame(demand_vector, columns=["Agent ID", "relative Price [€/kWh]", "Price [€]", "Energy [kWh]", "reactive Energy [kvar]"])
#         demand_vector=pd.merge(demand_vector, aux_arr, on="Agent ID", how="left")
        
        
#         # Supply Vector
#         supply_vector=np.insert(cost_vector.X,1,0,axis=1)
#         for a in range(0, len(supply_vector)):
#             supply_vector[a,3]=supply_vector[a,3]*(model.timestep.seconds/(60*60))*model.sref
#             supply_vector[a,4]=supply_vector[a,4]*(model.timestep.seconds/(60*60))*model.sref
#             if supply_vector[a,3]!=0:
#                supply_vector[a,1]=supply_vector[a,2]/supply_vector[a,3]
#         supply_vector=supply_vector[supply_vector[:,1].argsort()]
#         supply_vector=pd.DataFrame(supply_vector, columns=["Agent ID", "relative Price [€/kWh]", "Price [€]", "Energy [kWh]", "reactive Energy [kvar]"])
#         supply_vector=pd.merge(supply_vector, aux_arr, on="Agent ID", how="left")
        
#         netnod_pw_vector_neg_sum=sum([netnod_pw_vector_neg.X[i] for i in range(0, nn) if i!=model.slack])
#         netnod_pw_vector_pos_sum=sum([netnod_pw_vector_pos.X[i] for i in range(0, nn) if i!=model.slack])
#         netnod_pw_vector_LEC=[]
#         if netnod_pw_vector_neg_sum!=0 and netnod_pw_vector_pos_sum!=0:
#             for n in range(nn):
#                 if n==model.slack:
#                     netnod_pw_vector_LEC.append(0)
#                 elif netnod_pw_vector.X[n]<=0:
#                     netnod_pw_vector_LEC.append(P_lec.X/netnod_pw_vector_neg_sum*netnod_pw_vector.X[n])
#                 elif netnod_pw_vector.X[n]>=0:
#                     netnod_pw_vector_LEC.append(P_lec.X/netnod_pw_vector_pos_sum*netnod_pw_vector.X[n])
#         else:
#             netnod_pw_vector_LEC=0*netnod_pw_vector_neg.X
#         netnod_pw_vector_LEC=np.array(netnod_pw_vector_LEC)
#         netnod_pw_vector_ext=netnod_pw_vector.X-netnod_pw_vector_LEC
#         netnod_pw_vector_ext_total=sum([abs(netnod_pw_vector_ext[i]) for i in range(0, nn)])/2
#         netnod_pw_vector_LEC_total=sum([abs(netnod_pw_vector_LEC[i]) for i in range(0, nn)])/2
        
#         if netnod_pw_vector_ext_total!=0:
#             netnod_pw_vector_ext_share=netnod_pw_vector_ext/netnod_pw_vector_ext_total
#         else:
#             netnod_pw_vector_ext_share=netnod_pw_vector_ext*0
#         if netnod_pw_vector_LEC_total!=0:
#             netnod_pw_vector_LEC_share=netnod_pw_vector_LEC/netnod_pw_vector_LEC_total
#         else:
#             netnod_pw_vector_LEC_share=netnod_pw_vector_LEC*0
       
#         gridfee_levies_lec=gridfee_levies_lec.X*np.where(netnod_pw_vector_LEC_share<0,-netnod_pw_vector_LEC_share,0)/100
#         gridfee_levies_ext=gridfee_levies_ext.X*np.where(netnod_pw_vector_ext_share<0,-netnod_pw_vector_ext_share,0)/100
#         gridfee_levies_total= gridfee_levies_lec+  gridfee_levies_ext
#         gridfee_levies_total_kWh=[abs(round(gridfee_levies_total[n]/(netnod_pw_vector.X[n]/4),3)) if netnod_pw_vector.X[n]!=0 else 0  for n in range(nn)]

#         agent_pw_sc=pd.DataFrame(columns=['Agent ID', 'HN Self-Consumption [kWh]'])
#         agent_pw_lec=pd.DataFrame(columns=['Agent ID', 'Energy LEC [kWh]'])
#         agent_pw_ext=pd.DataFrame(columns=['Agent ID', 'Energy External [kWh]'])
        
        
#         for agent in model.agents:
#             agent_sc=0
#             agent_ext=0
#             agent_lec=0
#             if agent.bid!=0:
#                 if sum(demand_vector[demand_vector["Node"]==int(agent.bus)]["Energy [kWh]"])!=0:
#                     nodal_agent_share=(demand_vector[demand_vector["Agent ID"]==agent.unique_id]["Energy [kWh]"]/sum(demand_vector[demand_vector["Node"]==int(agent.bus)]["Energy [kWh]"])).values[0]                    
#                     agent_sc=netnod_pw_vector_sc.X[int(agent.bus)]*nodal_agent_share/4*100
#                     agent_ext = max(-netnod_pw_vector_ext[int(agent.bus)]*nodal_agent_share, 0.0) * 4 * 100
#                     agent_lec = max(-netnod_pw_vector_LEC[int(agent.bus)]*nodal_agent_share, 0.0) * 4 * 100
#             if agent.ask!=0:
#                 if sum(supply_vector[supply_vector["Node"]==int(agent.bus)]["Energy [kWh]"])!=0:
#                     nodal_agent_share=(supply_vector[supply_vector["Agent ID"]==agent.unique_id]["Energy [kWh]"]/sum(supply_vector[supply_vector["Node"]==int(agent.bus)]["Energy [kWh]"])).values[0]                    
#                     agent_sc=-netnod_pw_vector_sc.X[int(agent.bus)]*nodal_agent_share/4*100
#                     agent_ext=-max(netnod_pw_vector_ext[int(agent.bus)]*nodal_agent_share, 0.0) * 4 * 100
#                     agent_lec=-max(netnod_pw_vector_LEC[int(agent.bus)]*nodal_agent_share, 0.0) * 4 * 100
#             agent_pw_sc.loc[len(agent_pw_sc)] = [agent.unique_id, agent_sc]
#             agent_pw_lec.loc[len(agent_pw_lec)] = [agent.unique_id, agent_lec]
#             agent_pw_ext.loc[len(agent_pw_ext)] = [agent.unique_id, agent_ext]
              
#         net2=copy.deepcopy(model.grid)
#         counter=0
#         for a in range(len(netnod_pw_vector.X)):
#                 pw=netnod_pw_vector.X[a]*model.sref/1000
#                 qw=netnod_qw_vector.X[a]*model.sref/1000
#                 if pw<0:
#                     pp.create_load(net2,counter,p_mw=-pw, q_mvar=-qw)
#                 if pw>0:
#                     pp.create_sgen(net2,counter,p_mw=pw, q_mvar=qw)
#                 counter+=1
#         counter=0

#         pp.runpp(net2)
#         v_r=np.real(net2.res_bus["vm_pu"]*np.exp(1j * net2.res_bus["va_degree"]))
#         v_i=np.imag(net2.res_bus["vm_pu"]*np.exp(1j * net2.res_bus["va_degree"]))
#         v_d_max=max(abs(net2.res_bus["vm_pu"]-1))*100
#         I=net2.res_line["i_ka"]
#         I_d_max=max(net2.res_line["loading_percent"])
#         Trafo=max((net2.res_ext_grid["p_mw"]**2+net2.res_ext_grid["q_mvar"]**2)**(0.5)*1000)      
        
#         "Agent Results"
#         demand=demand_vector[["Agent ID", "Energy [kWh]", "relative Price [€/kWh]"]].sort_values(by="Agent ID")
#         supply=supply_vector[["Agent ID", "Energy [kWh]", "relative Price [€/kWh]"]].sort_values(by="Agent ID")
#         agent_result_vector=pd.merge(supply, demand, on="Agent ID", how="outer")
#         agent_result_vector=pd.merge( aux_arr,agent_result_vector, on="Agent ID", how="outer").sort_values(by="Agent ID")
#         agent_result_vector.columns=["Agent ID", "Node", "Agent Type", "Energy sold [kWh]", "ask price [€/kWh]",  "Energy bought [kWh]", "bid price [€/kWh]"]
#         agent_result_vector["Energy sold [kWh]"]=agent_result_vector["Energy sold [kWh]"].fillna(0)
#         agent_result_vector["Energy bought [kWh]"]=agent_result_vector["Energy bought [kWh]"].fillna(0)

#         #price
#         slack_id=[a for a in model.agents if a.flex == 999][0].unique_id
#         energy=cost_vector.X[cost_vector.X[:,0]==slack_id][0][2]/4+utility_vector.X[utility_vector.X[:,0]==slack_id][0][2]/4
#         price=cost_vector.X[cost_vector.X[:,0]==slack_id][0][1]+utility_vector.X[utility_vector.X[:,0]==slack_id][0][1]
#         if energy>0:
#             slack_price=price/energy/100
#         else:
#             slack_price=[a.ask for a in model.agents if a.flex == 999][0][2](1)*4/100-[a.margin_buy for a in model.agents if a.flex == 999][0]/100

#         agent_result_vector=pd.merge(agent_result_vector, agent_pw_ext, on="Agent ID", how="outer")
#         agent_result_vector=pd.merge(agent_result_vector, agent_pw_lec, on="Agent ID", how="outer")   
#         agent_result_vector=pd.merge(agent_result_vector, agent_pw_sc, on="Agent ID", how="outer")   
        
        
#         agent_result_vector["Revenue Energy LEC [€]"]= (slack_price* agent_result_vector["Energy LEC [kWh]"]).fillna(0)
#         agent_result_vector["Revenue Energy External [€]"]= (slack_price* agent_result_vector["Energy External [kWh]"]).fillna(0)
#         agent_result_vector["Fees and Levies LEC [€]"]= (model.gridfee_LEC+model.levies_LEC)/100* agent_result_vector["Energy LEC [kWh]"].clip(lower=0)
#         agent_result_vector["Fees and Levies External [€]"]=  (model.gridfee_ext+model.levies_ext)/100* agent_result_vector["Energy External [kWh]"].clip(lower=0)
#         #agent_result_vector["Profit [€]"]=[agent_result_vector["Nodal Price [€/kWh]"][a]-max(agent_result_vector["ask price [€/kWh]"].fillna(0)[a],agent_result_vector["bid price [€/kWh]"].fillna(0)[a]) for a in range(len(ID))]* agent_result_vector["Energy total [kWh]"]
        
        
#         "Nodal Results"
#         nodes = list(range(0, nn))

#         # Build each column as a flat list/array of length nn
#         nodal_result_vector = pd.DataFrame({
#             "Node": nodes,
#             "HN Self-Consumption [kWh]": netnod_pw_vector_sc.X / 4 * 100,
#             "HN Energy LEC [kWh]": P_lec.X / 4 * 100 *  netnod_pw_vector_LEC_share,
#             "HN Energy External [kWh]": netnod_pw_vector.X[model.slack] / 4 * 100 *  netnod_pw_vector_ext_share,
#             "Revenue energy [€]": slack_price* ((P_lec.X*netnod_pw_vector_LEC_share) + (netnod_pw_vector.X[model.slack] *netnod_pw_vector_ext_share))/ 4 * 100 ,
#             "Gridfees and Levies [€]": gridfee_levies_total * 100
#         })

#         # Ensure 'Node' is integer-typed (safe handling)
#         nodal_result_vector["Node"] = pd.to_numeric(nodal_result_vector["Node"], errors="coerce").astype("Int64")
      
              
#         "Line Results"
#         line=pd.concat([model.grid.line.name,I ], axis=1)
#         line.columns=["Line name", "I real [kA]"]
       
#         "EV Results"
#         if len(model.results)>0:
#             EV_result_vector=model.results[model.stepcount-1]["EV"].copy()
#         else:
#             columns = [
#                 "Agent ID",
#                 "SOC [kWh]",
#                 "SOC_rel",
#                 "Status",
#                 "Distance [km]",
#                 "Acc. Energy outside LEC [kWh]",
#                 "Acc. Cost outside LEC [€]",
#             ]

#             # Initialize the empty DataFrame
#             EV_result_vector = pd.DataFrame(columns=columns)
#             EV_result_vector["Agent ID"]=pd.DataFrame(agent_result_vector.loc[agent_result_vector["Agent Type"]=="EV"].loc[:,"Agent ID"].reset_index(drop=True))
#             EV_result_vector.fillna(0, inplace=True)
#         EV_result_vector["SOC [kWh]"]=[model.agents[xx-1].SOC for xx in EV_result_vector["Agent ID"].values]
#         EV_result_vector["SOC_rel"]=[model.agents[xx-1].SOC_rel for xx in EV_result_vector["Agent ID"].values]             
        
#         for xx in range(len(EV_result_vector)):
#             agent_id=int(EV_result_vector.loc[xx]["Agent ID"])
#             EV_result_vector.loc[xx, "Acc. Energy outside LEC [kWh]"]=[model.agents[agent_id-1].energy_outside_LEC]
#             EV_result_vector.loc[xx,"Acc. Cost outside LEC [€]"]=[model.agents[agent_id-1].capital_outside_LEC]
#             EV_result_vector.loc[xx,"Status"]=[model.agents[agent_id-1].status]
#             EV_result_vector.loc[xx,"Distance [km]"]=[model.agents[agent_id-1].current_km]
            

#         #Heatpump Results
#         if len(model.results)>0:
#             heatpump_result_vector=model.results[model.stepcount-1]["Heatpump"].copy()
#         else:
#             columns = [
#                 "Agent ID",
#                 "Temperature Indoor [°C]",
#                 "SOC_rel"
#               ]
#             heatpump_result_vector = pd.DataFrame(columns=columns)
#             heatpump_result_vector["Agent ID"]=pd.DataFrame(agent_result_vector.loc[agent_result_vector["Agent Type"]=="heatpump"].loc[:,"Agent ID"].reset_index(drop=True))
#             heatpump_result_vector.fillna(0, inplace=True)
#         heatpump_result_vector["Temperature Indoor [°C]"]=[model.agents[xx-1].T_in for xx in heatpump_result_vector["Agent ID"].values]
#         heatpump_result_vector["SOC_rel"]=[model.agents[xx-1].soc for xx in heatpump_result_vector["Agent ID"].values]
        
    
         
#         # Storage Results
#         storage_result_vector=pd.DataFrame(agent_result_vector.loc[agent_result_vector["Agent Type"]=="storage"].loc[:,"Agent ID"].reset_index(drop=True))
#         storage_result_vector.columns=["Agent ID"]
#         storage_result_vector["SOC_rel"]=[model.agents[xx-1].soc for xx in storage_result_vector["Agent ID"].values] 
                
#         # External
#         columns = [
#                 "Date",
#                 "T out [°C]",
#                 "DA Price [€/kWh]",
#                 "Slack Price [€/kWh]",
#                 "Total Social Welfare [€]",
#                 "Max. Voltage Deviation [p.u.]",
#                 "Max. Line Loading [%]",
#                 "Transformer Usage [kVA]"
#                 ]
#         social_welfare=-opt_all.ObjVal if opt_all.status == GRB.OPTIMAL else 0
#         External=[model.current_date, model.temperature_df[model.temperature_df.loc[:,"time"]==model.current_date].values[0][1], [agents.price for agents in model.agents if agents.flex==999][0], slack_price, social_welfare,v_d_max,I_d_max,Trafo]
#         External_vector = pd.DataFrame([External], columns=columns)
        
        
        
#         data = {
#             'demand': demand_vector,
#             'supply': supply_vector,
#             'Line_results': line,
#             "Node_results": nodal_result_vector,
#             "Input_Grid": External_vector,
#             "agents": agent_result_vector,
#             "storage":storage_result_vector,
#             "EV":EV_result_vector,
#             "Heatpump":heatpump_result_vector,
           
            
#         }

#         for a, value in data.items():
#             numeric_cols=value.select_dtypes(include='number').columns
#             value[numeric_cols] =value[numeric_cols].round(4)
#         opt_all.remove(opt_all.getVars())  # Remove all variables
#         opt_all.remove(opt_all.getConstrs())  # Remove all constraints
#         opt_all.dispose()
#         del opt_all

            
#         return data