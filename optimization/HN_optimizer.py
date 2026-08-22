import gc
import os
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Reals, Binary, 
    Constraint, Objective, minimize, value, SolverFactory, RangeSet, inequality
)
from datetime import timedelta, datetime
import numpy as np
import sys


class HNOptimizer:
    """
    Simplified version of HEM Optimizer that builds the model once during initialization
    and updates only parameters/bounds in each optimization round.
    """

    # Usable SOC band for batteries. Must match the limits the storage agent enforces
    # physically in agents.py provide_a_power; if the LP band is narrower it stops
    # planning (and stops posting an ask) as soon as the real SOC leaves the band.
    SOC_MIN = 0.05
    SOC_MAX = 0.95

    def __init__(self, agent_data_list, model_params, external_data):
        """
        Initialize the optimizer with static data and build the optimization model once.
        """
        # Store data
        self.agent_data = [agent for agent in agent_data_list 
                          if agent['bus'] == model_params['bus'] and agent['flex'] != 999]
        self.bus = model_params['bus']
        self.LEC_participation = model_params.get('LEC_participation')
        
        # Model parameters
        self.total_steps = model_params['total_steps']
        self.timestep = model_params['timestep']
        self.sref = model_params['sref']
        self.gridfee = model_params['gridfee_ext']
        self.levies = model_params['levies_ext']
        self.gridfee_LEC = model_params['gridfee_LEC']
        self.levies_LEC = model_params['levies_LEC']
        
        # Agent counts
        self.num_agents = len(self.agent_data)
        self.num_evs = sum(1 for a in self.agent_data if a['flex'] == 2)
        self.num_batteries = sum(1 for a in self.agent_data if a['flex'] == 1)
        self.num_heatpumps = sum(1 for a in self.agent_data if a['flex'] == 3)

        # Agent indices of the batteries, needed by the fee-base decomposition in
        # _build_base_constraints. Computed here because that runs before
        # _build_agent_constraints, which is what populates self.battery_indices.
        self.battery_agent_idx = [ii for ii, a in enumerate(self.agent_data) if a['flex'] == 1]
        
        # External data
        self.margin_buy = external_data['margin_buy']
        self.margin_sell = external_data['margin_sell']
        self.margin_charge = external_data['margin_charge']
        self.energy_price_df = external_data['energy_price_df'].copy()
        self.temperature_df = external_data['temperature_df'].copy()
        
        # Solver configuration - configurable solver
        solver_name = external_data.get('solver', 'gurobi')  # Default to gurobi
        self.solver = SolverFactory(solver_name)
        
        # Calculate optimal thread count based on MPI setup
        num_workers = external_data.get('num_mpi_workers', 1)  # Number of MPI worker processes
        total_cpus = external_data.get('total_cpus', None)  # Total available CPUs
        
        if total_cpus is not None and num_workers > 0:
            # Distribute CPUs among workers (reserve 1 for master if needed)
            threads_per_worker = max(1, (total_cpus - 1) // num_workers)
        else:
            # Default: use 2 threads per worker to avoid oversubscription
            threads_per_worker = 2
        
        # Set solver options based on solver type.
        #
        # mip_gap was 0.15 with time_limit 60. On a node energy cost of EUR 10-50 that
        # permits EUR 1.5-7.5 of optimality slack, while the entire battery arbitrage
        # opportunity is a 1-3 ct/kWh spread (a full 146.7 kWh cycle ~ EUR 2.93) -- the
        # tolerance was wider than the signal being optimised. 0.01 puts the relative
        # slack an order of magnitude below the spread.
        #
        # mip_gap_abs matters at least as much here, because a *relative* gap is
        # meaningless on a node whose objective sits near zero. Measured on bus 14:
        # the solver burned the full 300 s time limit and still reported a 28.6% gap
        # while the absolute slack was EUR 0.010 -- one cent, grinding to certify a
        # quantity with no economic content.
        #
        # 0.05 was tried first and left buses 6 and 12 (slack EUR 0.063 / 0.051) still
        # timing out just above the floor. 0.15 clears the whole measured spread with
        # room, and is still negligible against what a solve actually decides: this is
        # a receding-horizon plan where only the FIRST of 83 steps is executed, so
        # EUR 0.15 spread over the horizon is well under a cent on the action taken,
        # versus a per-step trade worth EUR 0.19-0.58.
        #
        # time_limit drops 300 -> 60 for the same reason. Once the buy_flag binaries
        # were gone the incumbent was always within pennies at the point of timeout;
        # the extra 240 s bought certification, never a better schedule. 60 s bounds
        # the worst case per bus at a fifth of the cost.
        self._configure_solver_options(solver_name,
                                       time_limit=60,
                                       mip_gap=0.01,
                                       mip_gap_abs=0.15,
                                       threads=threads_per_worker)

        # Per-solve solver diagnostics (termination condition, gap, wall time).
        # Written by _log_solve_diagnostics; see self.solve_log_path.
        # SOLVE_LOG_DIR redirects the whole directory without touching model.py,
        # so two runs (e.g. an A/B of a formulation change) can be kept apart.
        self.solve_log_path = external_data.get('solve_log_path')
        self.solve_log_dir = os.environ.get(
            'SOLVE_LOG_DIR', os.path.join('output', 'solver_diagnostics'))
        self._solve_seq = 0
        
        # Build model once
        self.max_horizon= model_params.get('horizon')
        self._build_model()



        
        print(f"HEM Optimizer initialized with {self.num_agents} agents")
    
    def _configure_solver_options(self, solver_name, time_limit=60, mip_gap=0.15,
                                  mip_gap_abs=None, threads=4):
        """
        Configure solver options in a solver-agnostic way.

        mip_gap_abs stops the solve once the incumbent is within that many EUR of the
        bound, regardless of the ratio. It is the meaningful criterion for a node whose
        objective is near zero, where a relative gap can stay huge over pennies.
        Solvers that expose no absolute-gap parameter (GLPK, CBC here) simply ignore it.
        """
        solver_name = solver_name.lower()

        if solver_name in ['gurobi', 'gurobi_direct', 'gurobi_persistent']:
            # Gurobi options - optimized for fast feasible solutions
            # (gurobi_direct/_persistent run in-process via gurobipy but accept
            #  the same native parameter names as the legacy shell interface)
            self.solver.options['TimeLimit'] = time_limit
            self.solver.options['MIPGap'] = mip_gap
            if mip_gap_abs is not None:
                self.solver.options['MIPGapAbs'] = mip_gap_abs
            self.solver.options['MIPFocus'] = 1  # 1=feasibility, 2=optimality, 3=bound
            self.solver.options['Heuristics'] = 0.3  # Spend 30% time finding good solutions fast
            self.solver.options['Cuts'] = 1  # Moderate cuts (faster than aggressive)
            self.solver.options['Presolve'] = 2  # Aggressive presolve
            self.solver.options['NoRelHeurTime'] = 10  # Stop expensive heuristics after 10s
            self.solver.options['ImproveStartTime'] = 5  # Start improving initial solution after 5s
            self.solver.options['Threads'] = threads
            self.solver.options['OutputFlag'] = 0
            
        elif solver_name in ['cplex', 'cplex_direct']:
            # IBM CPLEX options - optimized for fast feasible solutions
            self.solver.options['timelimit'] = time_limit
            self.solver.options['mip_tolerances_mipgap'] = mip_gap
            if mip_gap_abs is not None:
                self.solver.options['mip_tolerances_absmipgap'] = mip_gap_abs
            self.solver.options['emphasis_mip'] = 1  # 1=feasibility, 2=optimality, 3=balanced, 4=hidden
            self.solver.options['mip_strategy_heuristicfreq'] = 10  # Run heuristics more frequently
            self.solver.options['mip_limits_cutpasses'] = 1  # Fewer cut passes (faster)
            self.solver.options['preprocessing_presolve'] = 'y'  # Enable presolve
            self.solver.options['mip_strategy_rinsheur'] = 10  # RINS heuristic frequency
            self.solver.options['mip_strategy_fpheur'] = 1  # Feasibility pump heuristic
            self.solver.options['threads'] = threads
            # Suppress verbose parameter change logging
            self.solver.options['output_clonelog'] = -1  # Disable clone log messages
            self.solver.options['simplex_display'] = 0  # Minimize simplex output
            self.solver.options['mip_display'] = 0  # Suppress solution output logging
            
        elif solver_name == 'cbc':
            # CBC (open-source) options
            self.solver.options['seconds'] = time_limit
            self.solver.options['ratio'] = mip_gap
            self.solver.options['threads'] = threads
            
        elif solver_name == 'glpk':
            # GLPK (open-source, limited options)
            self.solver.options['tmlim'] = time_limit
            self.solver.options['mipgap'] = mip_gap

        elif solver_name == 'appsi_highs':
            # HiGHS (open-source) via APPSI interface
            self.solver.options['time_limit'] = time_limit
            self.solver.options['mip_rel_gap'] = mip_gap
            if mip_gap_abs is not None:
                self.solver.options['mip_abs_gap'] = mip_gap_abs
            self.solver.options['threads'] = threads
            self.solver.options['output_flag'] = False

        else:
            # Generic options (may not work for all solvers)
            print(f"Warning: Solver '{solver_name}' not specifically configured. Using generic options.")
            self.solver.options['timelimit'] = time_limit
            self.solver.options['mipgap'] = mip_gap
    
    def _build_model(self):
        """Build the optimization model structure once"""
        self.model = ConcreteModel("HEM_Optimization")
        
        # Sets
        self.model.AGENTS = RangeSet(0, self.num_agents - 1)
        self.model.TIME = RangeSet(0, self.max_horizon - 1)
        
        if self.num_evs > 0:
            self.model.EVS = RangeSet(0, self.num_evs - 1)
        if self.num_batteries > 0:
            self.model.BATTERIES = RangeSet(0, self.num_batteries - 1)
        if self.num_heatpumps > 0:
            self.model.HEATPUMPS = RangeSet(0, self.num_heatpumps - 1)
        
        # Dynamic parameters (updated each round)
        self.model.horizon = Param(mutable=True, default=self.max_horizon)
        self.model.prices = Param(self.model.TIME, mutable=True, default=30.0)
        
        # Mutable parameters for agent states
        if self.num_batteries > 0:
            self.model.battery_init_soc = Param(self.model.BATTERIES, mutable=True, default=0.5)
        if self.num_evs > 0:
            self.model.ev_init_soc = Param(self.model.EVS, mutable=True, default=0.2)
            # Remove plug-in/out time parameters as we'll use profile data
        if self.num_heatpumps > 0:
            self.model.hp_init_temp = Param(self.model.HEATPUMPS, mutable=True, default=20.0)
        
        # Add ambient temperature parameter for each time step (needed for EVs and heat pumps)
        if self.num_evs > 0 or self.num_heatpumps > 0:
            self.model.T_amb = Param(self.model.TIME, mutable=True, default=15.0)
        
        # Variables
        self.model.power_buy = Var(self.model.AGENTS, self.model.TIME, 
                                  within=NonNegativeReals, bounds=(0, 1000))
        self.model.power_sell = Var(self.model.AGENTS, self.model.TIME, 
                                   within=NonNegativeReals, bounds=(0, 1000))
        self.model.power_net = Var(self.model.TIME, within=Reals, bounds=(-1000, 1000))

        # Base that grid fees + levies are charged on: the node's external import
        # EXCLUDING battery charging, which settlement exempts (§118(6) EnWG,
        # market_optimizer.py is_storage rule). See fee_base_rule.
        self.model.fee_base = Var(self.model.TIME, within=NonNegativeReals, bounds=(0, 1000))
        
        if self.num_evs > 0:
            self.model.power_out_ev = Var(self.model.EVS, self.model.TIME, 
                                         within=NonNegativeReals, bounds=(0, 1000))
        
        self.model.cost_energy = Var(self.model.TIME, within=Reals, bounds=(-1e5, 1e3))
        self.model.cost_charge_ext = Var(self.model.TIME, within=Reals, bounds=(-100, 1000))
        
        self.model.stored_en = Var(within=Reals, bounds=(-1e5, 1e4))
        self.model.stored_en_price = Var(within=Reals, bounds=(-1e5, 1e4))
        
        # NOTE: Vars `A` (NonNegativeReals), `zz` (Binary) and `z_mode` (Binary per
        # agent per step) were declared here and deleted 2026-08-17. `A`/`zz` were
        # constrained by three Big-M rules (A_upper / A_zero / A_lower) but appeared in
        # no objective and no other constraint; `z_mode` was never referenced at all.
        # Together they added ~(2 + num_agents) * max_horizon binaries per optimizer,
        # all of them free to branch on and none of them affecting the solution.

        if self.num_batteries > 0:
            # Band matches the physical envelope the storage agent actually enforces in
            # agents.py provide_a_power (discharge down to 0.05, charge up to 0.95).
            self.model.soc_battery = Var(self.model.BATTERIES, RangeSet(0, self.max_horizon),
                                        within=NonNegativeReals, bounds=(self.SOC_MIN, self.SOC_MAX))
        
        if self.num_heatpumps > 0:
            self.model.T_hp = Var(self.model.HEATPUMPS, RangeSet(0, self.max_horizon), 
                                 within=Reals, bounds=(10, 30))
            self.model.Qdot_hp = Var(self.model.HEATPUMPS, RangeSet(0, self.max_horizon), 
                                    within=Reals, bounds=(-4000, 4000))
        
        if self.num_evs > 0:
            self.model.soc_ev = Var(self.model.EVS, RangeSet(0, self.max_horizon), 
                                   within=NonNegativeReals, bounds=(0, 10000))
        
        # NOTE: Var `buy_flag` (Binary per step) was deleted 2026-08-17, and with it
        # the Big-M constants bigM / bigMM. It selected between the buy-price and
        # sell-price expression for cost_energy through four Big-M constraints, but
        # that disjunction was never needed -- the cost function is convex, so its
        # upper envelope is reachable with two plain inequalities. See the cost_energy
        # constraints in _build_base_constraints for the argument.
        #
        # This was the dominant source of integrality: max_horizon binaries per
        # optimizer, every one of them with a Big-M so loose the LP relaxation could
        # satisfy the inactive branch with any fractional flag. Measured before the
        # change, bus 10's first solve ran the full 300 s time limit and terminated
        # at a 175% gap. What remains is battery_mode_{k}, num_batteries * horizon
        # binaries -- and a bus with no battery is now a pure LP.

        # Build base constraints
        self._build_base_constraints()
        
        # Build agent constraints once (using mutable parameters for variable data)
        self._build_agent_constraints()
        
        # Objective (will be updated with stored energy in agent constraints)
        def objective_rule(m):
            n = int(value(m.horizon))
            return (sum(m.cost_energy[t] + m.cost_charge_ext[t] for t in range(n)) - m.stored_en_price)
        self.model.objective = Objective(rule=objective_rule, sense=minimize)
        
        print("Full model structure built successfully")
    
    def _build_agent_constraints(self):
        """Build agent constraints once using mutable parameters for variable data"""
        # Helper functions (matching original)
        def clamp(val, min_val, max_val):
            return max(min(val, max_val), min_val)
        
        def provide_a_power_range(agent_data, time):
            try:
                power_profile = agent_data['power_profile']
                power = power_profile.loc[power_profile["time"] == time, "power"]
                if power.empty:
                    power = 0
                else:
                    power = power.values[0]
                return power   
            except Exception as e:
                return 0
        
        def get_profile_series(agent_data, year):
            # Get profile data for specific year - agent_data should contain profile info
            return agent_data.get(f"profile_df_{year}", agent_data.get('profile_df', {}))
        
        # Initialize counters
        battery_idx = 0
        hp_idx = 0
        ev_idx = 0
        
        # Store agent indices for stored energy calculation
        self.battery_indices = []  # List of (batt_idx, agent_idx, capacity)
        self.hp_indices = []  # List of (hp_idx, agent_idx, C_adapted, cop, timestep_hours)
        self.ev_indices = []  # List of (ev_idx, agent_idx)
        
        for ii, agent_data in enumerate(self.agent_data):
            if agent_data['flex'] == 0:  # Unflexible Load/RES
                # For unflexible agents, we'll handle power profiles in parameter updates
                def unflexible_power_rule_buy(m, t, agent_idx=ii, agent_info=agent_data):
                    if t >= value(m.horizon):
                        return Constraint.Skip
                    return m.power_buy[agent_idx, t] >= 0  # Will be fixed in update method
                self.model.add_component(f'unflexible_buy_{ii}', Constraint(self.model.TIME, rule=unflexible_power_rule_buy))
                
                def unflexible_power_rule_sell(m, t, agent_idx=ii, agent_info=agent_data):
                    if t >= value(m.horizon):
                        return Constraint.Skip
                    return m.power_sell[agent_idx, t] >= 0  # Will be fixed in update method
                self.model.add_component(f'unflexible_sell_{ii}', Constraint(self.model.TIME, rule=unflexible_power_rule_sell))
        
            elif agent_data['flex'] == 1:  # Battery (same as before)
                if self.num_batteries > 0:
                    current_batt_idx = battery_idx
                    current_agent_idx = ii
                    current_agent_data = agent_data
                    
                    # Initial SOC constraint using mutable parameter
                    def battery_initial_soc_rule(m):
                        return m.soc_battery[current_batt_idx, 0] == m.battery_init_soc[current_batt_idx]
                    self.model.add_component(f'battery_init_soc_{current_batt_idx}', Constraint(rule=battery_initial_soc_rule))
                    
                    # Power limits
                    def battery_power_sell_limit_rule(m, t):
                        if t >= value(m.horizon):
                            return Constraint.Skip
                        return m.power_sell[current_agent_idx, t] <= current_agent_data['max_power'] / self.sref
                    self.model.add_component(f'battery_sell_limit_{current_batt_idx}', Constraint(self.model.TIME, rule=battery_power_sell_limit_rule))
                    
                    def battery_power_buy_limit_rule(m, t):
                        if t >= value(m.horizon):
                            return Constraint.Skip
                        return m.power_buy[current_agent_idx, t] <= current_agent_data['max_power'] / self.sref
                    self.model.add_component(f'battery_buy_limit_{current_batt_idx}', Constraint(self.model.TIME, rule=battery_power_buy_limit_rule))
                    
                    # Reformulate bilinear constraint using Big-M method for SCIP compatibility
                    # Add binary variable for charge/discharge mode
                    setattr(self.model, f'battery_mode_{current_batt_idx}', Var(self.model.TIME, within=Binary))
                    battery_mode_var = getattr(self.model, f'battery_mode_{current_batt_idx}')
                    
                    # If mode=1: can buy, cannot sell. If mode=0: can sell, cannot buy
                    def battery_buy_mode_rule(m, t):
                        if t >= value(m.horizon):
                            return Constraint.Skip
                        return m.power_buy[current_agent_idx, t] <= battery_mode_var[t] * current_agent_data['max_power'] / self.sref
                    self.model.add_component(f'battery_buy_mode_{current_batt_idx}', Constraint(self.model.TIME, rule=battery_buy_mode_rule))
                    
                    def battery_sell_mode_rule(m, t):
                        if t >= value(m.horizon):
                            return Constraint.Skip
                        return m.power_sell[current_agent_idx, t] <= (1 - battery_mode_var[t]) * current_agent_data['max_power'] / self.sref
                    self.model.add_component(f'battery_sell_mode_{current_batt_idx}', Constraint(self.model.TIME, rule=battery_sell_mode_rule))
                    
                    def battery_soc_dynamics_rule(m, t):
                        if t > value(m.horizon) - 1 or t > self.max_horizon - 1:
                            return Constraint.Skip
                        delta_soc = ((m.power_buy[current_agent_idx, t] * current_agent_data['efficiency'] - 
                                    m.power_sell[current_agent_idx, t] / current_agent_data['efficiency']) /
                                   ((3600 / self.timestep.seconds) * current_agent_data['capacity'] / self.sref))
                        return m.soc_battery[current_batt_idx, t + 1] == m.soc_battery[current_batt_idx, t] + delta_soc
                    self.model.add_component(f'battery_soc_dynamics_{current_batt_idx}', Constraint(self.model.TIME, rule=battery_soc_dynamics_rule))
                    
                    # Store battery info for stored energy calculation
                    self.battery_indices.append((current_batt_idx, current_agent_idx, current_agent_data['capacity']))
                    battery_idx += 1
            
            elif agent_data['flex'] == 3:  # Heat Pump (completely rewritten to match original)
                if self.num_heatpumps > 0:
                    current_hp_idx = hp_idx
                    current_agent_idx = ii
                    current_agent_data = agent_data
                    
                    # No selling for heat pumps
                    def hp_no_sell_rule(m, t):
                        if t >= value(m.horizon):
                            return Constraint.Skip
                        return m.power_sell[current_agent_idx, t] == 0
                    self.model.add_component(f'hp_no_sell_{current_hp_idx}', Constraint(self.model.TIME, rule=hp_no_sell_rule))
                    
                    # Initial temperature constraint using mutable parameter
                    def hp_initial_temp_rule(m):
                        return m.T_hp[current_hp_idx, 0] == m.hp_init_temp[current_hp_idx]
                    self.model.add_component(f'hp_init_temp_{current_hp_idx}', Constraint(rule=hp_initial_temp_rule))
                    
                    # Heat flow upper bound constraint (always active)
                    def hp_qdot_upper_rule(m, t):
                        if t > value(m.horizon):
                            return Constraint.Skip
                        return m.Qdot_hp[current_hp_idx, t] <= current_agent_data.get('Q_dot_max', 8000)
                    self.model.add_component(f'hp_qdot_upper_{current_hp_idx}', Constraint(self.model.TIME, rule=hp_qdot_upper_rule))
                    
                    # Heat flow lower bound (non-negative)
                    def hp_qdot_lower_rule(m, t):
                        if t > value(m.horizon):
                            return Constraint.Skip
                        return m.Qdot_hp[current_hp_idx, t] >= 0
                    self.model.add_component(f'hp_qdot_lower_{current_hp_idx}', Constraint(self.model.TIME, rule=hp_qdot_lower_rule))
                    
                    # Power to heat flow relationship: power_buy = Qdot / cop / sref (always active)
                    def hp_power_qdot_rule(m, t):
                        if t > value(m.horizon):
                            return Constraint.Skip
                        cop = current_agent_data.get('cop', 3.5)
                        return m.power_buy[current_agent_idx, t] == m.Qdot_hp[current_hp_idx, t] / cop / self.sref
                    self.model.add_component(f'hp_power_qdot_{current_hp_idx}', Constraint(self.model.TIME, rule=hp_power_qdot_rule))
                    
                    # Temperature dynamics will be rebuilt each round in _update_agent_constraints
                    # based on whether heating is needed (matching original conditional constraint building)
                    
                    # Temperature bounds
                    def hp_temp_upper_rule(m, t):
                        if t > value(m.horizon):
                            return Constraint.Skip
                        return m.T_hp[current_hp_idx, t] <= current_agent_data.get('T_max', 24.0)
                    
                    def hp_temp_lower_rule(m, t):
                        if t > value(m.horizon):
                            return Constraint.Skip  
                        return m.T_hp[current_hp_idx, t] >= current_agent_data.get('T_min', 18.0)
                    
                    self.model.add_component(f'hp_temp_upper_{current_hp_idx}', Constraint(self.model.TIME, rule=hp_temp_upper_rule))
                    self.model.add_component(f'hp_temp_lower_{current_hp_idx}', Constraint(self.model.TIME, rule=hp_temp_lower_rule))
                    
                    # Store heat pump info for stored energy calculation
                    C_adapted = current_agent_data.get('C_adapted', 50000)
                    cop = current_agent_data.get('cop', 3.5)
                    timestep_hours = self.timestep.seconds / 3600
                    self.hp_indices.append((current_hp_idx, current_agent_idx, C_adapted, cop, timestep_hours))
                    hp_idx += 1
            
            elif agent_data['flex'] == 2:  # EV (completely rewritten to match original)
                if self.num_evs > 0:
                    current_ev_idx = ev_idx
                    current_agent_idx = ii
                    current_agent_data = agent_data
                    
                    # No selling for EVs
                    def ev_no_sell_rule(m, t):
                        if t >= value(m.horizon):
                            return Constraint.Skip
                        return m.power_sell[current_agent_idx, t] == 0
                    self.model.add_component(f'ev_no_sell_{current_ev_idx}', Constraint(self.model.TIME, rule=ev_no_sell_rule))
                    
                    # Initial SOC constraint using mutable parameter
                    def ev_initial_soc_rule(m):
                        return m.soc_ev[current_ev_idx, 0] == m.ev_init_soc[current_ev_idx]
                    self.model.add_component(f'ev_init_soc_{current_ev_idx}', Constraint(rule=ev_initial_soc_rule))
                    
                    # SOC bounds (matching original)
                    def ev_soc_upper_rule(m, t):
                        if t > value(m.horizon):
                            return Constraint.Skip
                        return m.soc_ev[current_ev_idx, t] <= 0.98 * current_agent_data['max_capacity']
                    
                    def ev_soc_lower_rule(m, t):
                        if t > value(m.horizon):
                            return Constraint.Skip
                        return m.soc_ev[current_ev_idx, t] >= 0.02 * current_agent_data['max_capacity']
                    
                    self.model.add_component(f'ev_soc_upper_{current_ev_idx}', Constraint(self.model.TIME, rule=ev_soc_upper_rule))
                    self.model.add_component(f'ev_soc_lower_{current_ev_idx}', Constraint(self.model.TIME, rule=ev_soc_lower_rule))
                    
                    # SOC dynamics constraint - accounts for charging, external charging, and driving
                    # We'll use a mutable parameter for driving energy consumption
                    setattr(self.model, f'ev_driving_energy_{current_ev_idx}', Param(self.model.TIME, mutable=True, default=0.0))
                    ev_driving_energy_param = getattr(self.model, f'ev_driving_energy_{current_ev_idx}')
                    
                    def ev_soc_dynamics_rule(m, t):
                        if t > value(m.horizon) - 1 or t > self.max_horizon - 1:
                            return Constraint.Skip
                        # Home charging contribution (Pyomo expression, not Python calculation)
                        charging_energy = (m.power_buy[current_agent_idx, t] * current_agent_data.get('efficiency', 0.9) * 
                                         self.timestep.seconds / 3600 * self.sref)
                        # Work/external charging contribution (Pyomo expression)
                        external_charging = (m.power_out_ev[current_ev_idx, t] * current_agent_data.get('efficiency', 0.9) * 
                                           self.timestep.seconds / 3600 * self.sref)
                        # SOC next = SOC current + charging - driving energy
                        return m.soc_ev[current_ev_idx, t + 1] == m.soc_ev[current_ev_idx, t] + charging_energy + external_charging - ev_driving_energy_param[t]
                    self.model.add_component(f'ev_soc_dynamics_{current_ev_idx}', Constraint(self.model.TIME, rule=ev_soc_dynamics_rule))
                    
                    # The actual mode-based constraints (1=home, 2=parked, 3=driving, 4=work) 
                    # will be handled in the update method through variable fixing based on profile data
                    
                    # Store EV info for stored energy calculation
                    self.ev_indices.append((current_ev_idx, current_agent_idx))
                    ev_idx += 1
        
        # Define stored energy constraint using rule that evaluates at solve time
        # Create separate variables for each component to aid debugging
        self.model.battery_stored = Var(within=Reals, bounds=(-1e5, 1e4))
        self.model.hp_stored = Var(within=Reals, bounds=(-1e5, 1e4))
        self.model.ev_stored = Var(within=Reals, bounds=(-1e5, 1e4))
        
        def battery_stored_rule(m):
            battery_total = sum(
                (m.soc_battery[batt_idx, value(m.horizon)] - m.battery_init_soc[batt_idx]) * capacity
                for batt_idx, agent_idx, capacity in self.battery_indices
            ) if self.battery_indices else 0
            return m.battery_stored == battery_total
        self.model.battery_stored_constraint = Constraint(rule=battery_stored_rule)
        
        def hp_stored_rule(m):
            hp_total = sum(
                (m.T_hp[hp_idx, value(m.horizon)] - m.hp_init_temp[hp_idx]) * C_adapted * 60 / 15 / cop * timestep_hours
                for hp_idx, agent_idx, C_adapted, cop, timestep_hours in self.hp_indices
            ) if self.hp_indices else 0
            return m.hp_stored == hp_total
        self.model.hp_stored_constraint = Constraint(rule=hp_stored_rule)
        
        def ev_stored_rule(m):
            ev_total = sum(
                2 * m.soc_ev[ev_idx, value(m.horizon)] - m.ev_init_soc[ev_idx]
                for ev_idx, agent_idx in self.ev_indices
            ) if self.ev_indices else 0
            return m.ev_stored == ev_total
        self.model.ev_stored_constraint = Constraint(rule=ev_stored_rule)
        
        def stored_en_rule(m):
            return m.stored_en == m.battery_stored + m.hp_stored + m.ev_stored
        self.model.stored_en_constraint = Constraint(rule=stored_en_rule)
        
        # Terminal value of the inventory change over the horizon, in EUR.
        #
        # stored_en is a delta in kWh and avg_price is in ct/kWh, so the whole product
        # needs /100 to match cost_energy's units -- the old /10 on the fee block left the
        # credit ~100x too large (measured: 64.8 kWh credited EUR 90.1, i.e. EUR 1.39/kWh,
        # against EUR 11.55 of total node energy cost), which was the only thing overcoming
        # the phantom charging fee and also why the LP refused to discharge once full.
        #
        # Fees are dropped from the credit as well: they are a flow cost already paid on
        # the way in, so charging them again on terminal inventory double-counts.
        def stored_en_price_rule(m):
            n = int(value(m.horizon))
            avg_price = sum(m.prices[t] for t in range(n)) / n if n > 0 else 0
            return m.stored_en_price == avg_price * m.stored_en / 100
        self.model.stored_en_price_constraint = Constraint(rule=stored_en_price_rule)
        
        print(f"Agent constraints built for {len(self.agent_data)} agents")
    
    def _build_base_constraints(self):
        """Build the base constraints that don't depend on specific agent data"""
        # Power balance constraint
        def power_balance_rule(m, t):
            if t >= value(m.horizon):
                return Constraint.Skip
            return m.power_net[t] == sum(m.power_buy[i, t] for i in m.AGENTS) - sum(m.power_sell[i, t] for i in m.AGENTS)
        self.model.power_balance = Constraint(self.model.TIME, rule=power_balance_rule)

        # Fee base: max(0, power_net - battery charging).
        #
        # power_net bundles household load with battery charging, so charging fees to the
        # whole of it taxes battery arbitrage at gridfee+levies (~15.7 ct/kWh) that
        # settlement never actually charges storage. Subtracting battery charging leaves
        # ordinary household consumption paying full fees, which is correct.
        #
        # Only the >= arm of max() is stated: fee_base appears nowhere except the buy
        # branch of cost_energy, with a positive coefficient, and cost_energy is minimised
        # -- so the solver drives fee_base down onto max(0, .) on its own. No binary needed.
        # On sell steps fee_base is free above its lower bound and enters nothing.
        def fee_base_rule(m, t):
            if t >= value(m.horizon):
                return Constraint.Skip
            batt_charge = sum(m.power_buy[i, t] for i in self.battery_agent_idx)
            return m.fee_base[t] >= m.power_net[t] - batt_charge
        self.model.fee_base_con = Constraint(self.model.TIME, rule=fee_base_rule)

        # NOTE: constraints A_upper / A_zero / A_lower were deleted here 2026-08-17
        # along with the `A` and `zz` variables they governed -- see the note at the
        # variable declarations. They defined A[t] = max(0, power_net[t]) via Big-M,
        # but nothing ever read A.

        # Cost of the node's external exchange, as the upper envelope of two lines.
        #
        # This replaced a Big-M disjunction on a `buy_flag` binary (four constraints
        # pinning cost_energy to whichever branch the flag selected, plus two more
        # defining the flag from the sign of power_net). The binary was unnecessary:
        # buying costs spot + margin_buy + fees per kWh while selling pays only
        # spot - margin_sell, so the buy line is strictly the steeper of the two and
        # the cost function is convex. Under a minimised objective, two >= inequalities
        # drive cost_energy to exactly max(buy, sell), which is the true cost:
        #
        #   power_net > 0 -> buy - sell = conv*(power_net*(margin_buy + margin_sell)
        #                    + fee_base*(levies + gridfee))/100 > 0, so buy wins.
        #   power_net < 0 -> nothing pushes fee_base up (it enters only here, with a
        #                    positive coefficient, under minimisation), so it falls to
        #                    its floor of max(0, power_net - battery charge) = 0,
        #                    leaving buy - sell = conv*power_net*1.3/100 < 0: sell wins.
        #
        # Exact, not a relaxation -- and it makes the LP relaxation exact too, where
        # the Big-M version left it nearly vacuous.
        #
        # Energy price applies to the full node exchange; fees only to the fee base
        # (node import net of battery charging, which settlement exempts).
        def _formula_buy(m, t):
            conv = self.sref / (3600 / self.timestep.seconds)
            return conv * (m.power_net[t] * (m.prices[t] + self.margin_buy) / 100
                           + m.fee_base[t] * (self.levies + self.gridfee) / 100)

        def _formula_sell(m, t):
            conv = self.sref / (3600 / self.timestep.seconds)
            return conv * m.power_net[t] * (m.prices[t] - self.margin_sell) / 100

        def cost_buy_rule(m, t):
            if t >= value(m.horizon):
                return Constraint.Skip
            return m.cost_energy[t] >= _formula_buy(m, t)
        self.model.cost_buy = Constraint(self.model.TIME, rule=cost_buy_rule)

        def cost_sell_rule(m, t):
            if t >= value(m.horizon):
                return Constraint.Skip
            return m.cost_energy[t] >= _formula_sell(m, t)
        self.model.cost_sell = Constraint(self.model.TIME, rule=cost_sell_rule)

        # External charging cost constraint
        def cost_charge_ext_rule(m, t):
            if t >= value(m.horizon):
                return Constraint.Skip
            if self.num_evs > 0:
                ev_power_sum = sum(m.power_out_ev[v, t] for v in m.EVS)
            else:
                ev_power_sum = 0
            return (m.cost_charge_ext[t] == ev_power_sum / (3600 / self.timestep.seconds) * 
                    self.sref * (m.prices[t] + self.margin_charge + self.gridfee + self.levies) / 100)
        self.model.cost_charge_ext_con = Constraint(self.model.TIME, rule=cost_charge_ext_rule)

    def _update_variable_bounds(self, n):
        """
        Tighten fee_base and cost_energy to the range this step can actually reach.

        This began as Big-M tightening (M=1e7 / MM=1e6 stood against variables bounded
        +/-1000 pu, four orders of magnitude looser than the model can reach). The
        Big-M constants are gone now -- the buy_flag disjunction they served was
        replaced by the convex two-inequality form -- but the bound arithmetic is still
        worth doing: both variables were declared with boxes far wider than anything
        reachable, and cost_energy's asymmetric (-1e5, 1e3) box could actually bind.

        Every bound set here dominates the widest value the variable can take given the
        current step's power bounds, so no feasible schedule is cut off. SAFETY is
        slack against float error only.
        """
        SAFETY = 1.05
        conv = self.sref / (3600 / self.timestep.seconds)
        # Declared bound on power_buy / power_sell / power_net; nothing can exceed it.
        DECLARED_P_UB = 1000.0

        def _var_ub(v):
            """Largest value v can take right now: its fixed value, or its upper bound."""
            if v.is_fixed():
                return abs(value(v))
            return DECLARED_P_UB if v.ub is None else v.ub

        # Widest absolute price over the horizon, so the cost bound holds through
        # negative-price hours and Q1 spikes alike.
        price_abs_max = max((abs(value(self.model.prices[t])) for t in range(n)),
                            default=0.0)

        fee_rate = self.levies + self.gridfee

        for t in range(n):
            buy_ub = min(DECLARED_P_UB,
                         sum(_var_ub(self.model.power_buy[i, t]) for i in range(self.num_agents)))
            sell_ub = min(DECLARED_P_UB,
                          sum(_var_ub(self.model.power_sell[i, t]) for i in range(self.num_agents)))

            # fee_base[t] only ever needs to reach max(0, power_net[t] - battery charge),
            # and power_net[t] <= buy_ub. Its declared ub of 1000 left it free to float
            # far above that.
            self.model.fee_base[t].setub(buy_ub)

            # cost_energy[t] settles on max(buy, sell); both formulas take power_net,
            # which lies in [-sell_ub, buy_ub], so each uses the wider of the two --
            # not buy_ub, which would under-bound a node whose PV export exceeds its
            # peak import. Their sum dominates either one.
            net_abs = max(buy_ub, sell_ub)
            f_buy = conv * (net_abs * (price_abs_max + self.margin_buy)
                            + buy_ub * fee_rate) / 100
            f_sell = conv * net_abs * (price_abs_max + self.margin_sell) / 100
            cost_bound = (f_buy + f_sell) * SAFETY

            # The declared (-1e5, 1e3) box was asymmetric, and the +1e3 ub could
            # actually bind on a high-price import step, silently truncating the cost
            # it is meant to equal.
            self.model.cost_energy[t].setlb(-cost_bound)
            self.model.cost_energy[t].setub(cost_bound)

        # Steps past the horizon are skipped by every constraint and absent from the
        # objective, but reset them so a later longer horizon does not inherit a bound.
        for t in range(n, self.max_horizon):
            self.model.fee_base[t].setub(DECLARED_P_UB)

    def _log_solve_diagnostics(self, result, n, solve_seconds):
        """
        Append one JSON line per solve: termination condition, objective, best bound,
        realised MIP gap, wall time.

        Solver output is suppressed (OutputFlag 0), so before this the only evidence
        about solve quality was wall-clock inference across a whole run. The realised
        gap is the number that decides whether the schedule is trustworthy: a solve
        that stops at the time limit with a gap of the same order as the arbitrage
        spread has returned a near-arbitrary schedule, not an optimal one.
        """
        import json, os
        from pyomo.opt import TerminationCondition

        self._solve_seq += 1

        def _num(x):
            try:
                x = float(x)
            except (TypeError, ValueError):
                return None
            return x if abs(x) < 1e29 else None  # Pyomo's +/-inf placeholders

        def fmt_num(x):
            return "n/a" if x is None else f"{x:.3f}"

        tc = str(result.solver.termination_condition)
        lb = _num(getattr(result.problem, 'lower_bound', None))
        ub = _num(getattr(result.problem, 'upper_bound', None))

        gap = None
        if lb is not None and ub is not None:
            gap = abs(ub - lb) / max(abs(ub), 1e-9)

        record = {
            'seq': self._solve_seq,
            # self.bus comes from the network dataframe as a numpy int64, which
            # json.dumps refuses; int() it here rather than reaching for a custom
            # encoder, since every other field is already a plain scalar or str.
            'bus': int(self.bus),
            'date': str(self.current_date),
            'horizon': n,
            'termination': tc,
            'status': str(result.solver.status),
            'objective': ub,
            'best_bound': lb,
            'mip_gap': gap,
            'wall_seconds': round(solve_seconds, 3),
            # Binaries left in the model. buy_flag is gone, so this is
            # num_batteries * horizon -- 0 for a bus with no battery, which is a
            # pure LP. Recorded to correlate solve time against integrality.
            'n_binaries': self.num_batteries * n,
        }

        # Loud on stdout only when the solve is suspect -- otherwise this fires on
        # every step of a multi-thousand-step run and buries the sim's own output.
        # "Suspect" is judged on absolute slack, not the ratio: a node with a near-zero
        # objective can sit at a 28% relative gap over one cent, which is not worth a
        # line of output, while EUR 1 of slack matters even at a tiny ratio.
        target_gap, target_gap_abs = 0.01, 0.15
        slack = None if (ub is None or lb is None) else abs(ub - lb)
        suspect = (result.solver.termination_condition != TerminationCondition.optimal
                   and (slack is None or slack > target_gap_abs * 1.5))
        suspect = suspect or (gap is not None and gap > target_gap * 1.5
                              and slack is not None and slack > target_gap_abs * 1.5)
        if suspect and tc != str(TerminationCondition.infeasible):
            gap_str = f"{gap:.4f}" if gap is not None else "n/a"
            # Print the absolute objective and bound alongside the ratio: a relative
            # gap is meaningless when the objective sits near zero, and what decides
            # whether the schedule is trustworthy is the slack in EUR against a
            # ~EUR 1.5-4.4 arbitrage cycle.
            print(f"[solver] bus {int(self.bus)} step {self._solve_seq} {self.current_date}: "
                  f"{tc}, gap={gap_str}, obj={fmt_num(ub)}, bound={fmt_num(lb)}, "
                  f"slack={fmt_num(slack)} EUR, {solve_seconds:.1f}s")

        path = self.solve_log_path
        if path is None:
            # One file per bus: buses may be solved by separate MPI workers, and
            # interleaved appends from several processes would corrupt the log.
            path = os.path.join(self.solve_log_dir, f'bus_{int(self.bus)}.jsonl')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(record, default=str) + '\n')
        except Exception as exc:
            # Diagnostics must never take down a multi-hour run. Deliberately broad:
            # an OSError-only guard let a numpy int64 in `bus` kill the first run
            # through json.dumps, 5 minutes of solve time after the solve succeeded.
            if self._solve_seq == 1:
                print(f"[solver] could not write solve log to {path}: {exc!r}")

    def optimize(self, variable_data):
        """
        Optimize with current variable data by updating model parameters
        """
        self.current_date = variable_data['current_date']
        stepcount = variable_data['stepcount']
        agent_states = variable_data['agent_states']
        
        # Calculate horizon
        n = int(min(self.max_horizon, self.total_steps - stepcount))
        
        # Update horizon parameter
        self.model.horizon.set_value(n)
        
        # Update price parameters
        price_data = self.energy_price_df[
            (self.energy_price_df['time'] >= self.current_date) & 
            (self.energy_price_df['time'] < self.current_date + n * self.timestep)
        ]["price"]
        
        for t in range(n):
                self.model.prices[t].set_value(float(price_data.iloc[t]))

        
        # Clear and rebuild agent constraints (simplified for now)
        self._update_agent_constraints(self.current_date, n, agent_states)
        
        # Handle unflexible agent power profiles (these need to be updated each round)
        self._update_unflexible_profiles(self.current_date, n)

        # Tighten fee_base / cost_energy against the bounds this step actually has.
        # Must run last: it reads the power bounds the two updates above just set.
        self._update_variable_bounds(n)

        # Solve
        import time as _time
        _t0 = _time.perf_counter()
        result = self.solver.solve(self.model, tee=False)
        solve_seconds = _time.perf_counter() - _t0
        self._log_solve_diagnostics(result, n, solve_seconds)

        # Cleanup Gurobi resources to prevent file handle leak
        if hasattr(self.solver, '_solver_model') and self.solver._solver_model is not None:
            try:
                self.solver._solver_model.dispose()
            except:
                pass
            self.solver._solver_model = None
        import gc
        gc.collect()
        
        from pyomo.opt import TerminationCondition
        tc = result.solver.termination_condition

        if tc == TerminationCondition.infeasible:
            print("\n" + "="*80)
            print("OPTIMIZATION INFEASIBLE - DIAGNOSTIC INFORMATION")
            print("="*80)
            print(f"Current date: {self.current_date}")
            print(f"Horizon: {n} timesteps")
            print(f"Bus: {self.bus}")
            print(f"Number of agents: {self.num_agents}")
            print(f"  - EVs: {self.num_evs}")
            print(f"  - Batteries: {self.num_batteries}")
            print(f"  - Heat Pumps: {self.num_heatpumps}")
            
            # Print EV states and constraints with SOC trajectory
            if self.num_evs > 0:
                print("\n--- EV DIAGNOSTICS WITH SOC TRAJECTORY ---")
                for ev_idx in range(self.num_evs):
                    agent_data = [a for a in self.agent_data if a['flex'] == 2][ev_idx]
                    agent_idx = next(i for i, a in enumerate(self.agent_data) if a['unique_id'] == agent_data['unique_id'])
                    
                    print(f"\nEV {ev_idx} (ID: {agent_data['unique_id']}):")
                    initial_soc = value(self.model.ev_init_soc[ev_idx])
                    max_cap = agent_data['max_capacity']
                    soc_min = 0.02 * max_cap
                    soc_max = 0.98 * max_cap
                    efficiency = agent_data.get('efficiency', 0.9)
                    
                    print(f"  Initial SOC: {initial_soc:.2f} Wh ({initial_soc/max_cap*100:.1f}%)")
                    print(f"  Max capacity: {max_cap:.2f} Wh")
                    print(f"  SOC bounds: [{soc_min:.2f}, {soc_max:.2f}] Wh")
                    print(f"  Efficiency: {efficiency:.2f}")
                    
                    # Simulate SOC trajectory
                    print(f"\n  {'t':>3} {'Mode':>8} {'Power_W':>10} {'Fixed':>6} {'Driving':>10} {'SOC_Wh':>10} {'SOC_%':>7} {'Status':>10}")
                    print(f"  {'-'*88}")
                    
                    soc = initial_soc
                    for t in range(min(10, n)):
                        power_buy_var = self.model.power_buy[agent_idx, t]
                        is_fixed = power_buy_var.is_fixed()
                        power_ub = power_buy_var.ub if power_buy_var.ub is not None else 1000
                        
                        # Get actual power value
                        if is_fixed:
                            actual_power = power_buy_var.value if power_buy_var.value is not None else 0
                        else:
                            actual_power = power_ub  # Assume max charging if not fixed
                        
                        # Get driving energy
                        driving_param = getattr(self.model, f'ev_driving_energy_{ev_idx}')
                        driving_energy = value(driving_param[t])
                        
                        # Determine mode
                        if is_fixed and actual_power == 0:
                            mode = "Driving" if driving_energy > 0 else "Parked"
                        elif power_ub < 100:
                            mode = "Home"
                        else:
                            mode = "Work"
                        
                        # Calculate actual charging energy
                        charging_energy = actual_power * efficiency * self.timestep.seconds / 3600 * self.sref
                        soc = soc + charging_energy - driving_energy
                        
                        # Check feasibility
                        status = "OK" if soc_min <= soc <= soc_max else "INFEAS!"
                        
                        print(f"  {t:3d} {mode:>8} {actual_power:10.2f} {str(is_fixed):>6} {driving_energy:10.2f} {soc:10.2f} {soc/max_cap*100:6.1f}% {status:>10}")
                    
                    # Summary
                    if soc < soc_min:
                        print(f"\n  *** SOC BELOW MINIMUM: {soc:.2f} < {soc_min:.2f} Wh ***")
                    elif soc > soc_max:
                        print(f"\n  *** SOC ABOVE MAXIMUM: {soc:.2f} > {soc_max:.2f} Wh ***")
            
            # Print Heat Pump states
            if self.num_heatpumps > 0:
                print("\n--- HEAT PUMP DIAGNOSTICS ---")
                for hp_idx in range(self.num_heatpumps):
                    agent_data = [a for a in self.agent_data if a['flex'] == 3][hp_idx]
                    print(f"\nHP {hp_idx} (ID: {agent_data['unique_id']}):")
                    print(f"  Initial temp: {value(self.model.hp_init_temp[hp_idx]):.2f} °C")
                    print(f"  Temp bounds: [{agent_data.get('T_min', 18.0):.1f}, {agent_data.get('T_max', 24.0):.1f}] °C")
                    print(f"  Q_dot_max: {agent_data.get('Q_dot_max', 8000):.0f} W")
                    print(f"  COP: {agent_data.get('cop', 3.5):.2f}")
            
            # Print Battery states
            if self.num_batteries > 0:
                print("\n--- BATTERY DIAGNOSTICS ---")
                for batt_idx in range(self.num_batteries):
                    agent_data = [a for a in self.agent_data if a['flex'] == 1][batt_idx]
                    print(f"\nBattery {batt_idx} (ID: {agent_data['unique_id']}):")
                    print(f"  Initial SOC: {value(self.model.battery_init_soc[batt_idx]):.2f}")
                    print(f"  SOC bounds: [{self.SOC_MIN}, {self.SOC_MAX}]")
                    print(f"  Max power: {agent_data['max_power']:.0f} W")
                    print(f"  Capacity: {agent_data['capacity']:.0f} Wh")
            
            print("\n--- ATTEMPTING TO COMPUTE IIS (Irreducible Inconsistent Subsystem) ---")
            try:
                grb = self.solver._solver_model
                grb.computeIIS()
                grb.write("model.ilp")
                print("IIS saved to model.ilp")
            except Exception as e:
                print(f"Could not compute IIS: {e}")
            
            print("="*80 + "\n")
            raise Exception("Optimization problem is infeasible. See diagnostics above.")
            exit()
        
        # Return simplified results
        if self.LEC_participation:
            return self._extract_results_LEC(n)
        else:
            return self._extract_results_standard(n)
    
    def _update_agent_constraints(self, current_date, n, agent_states):
        """Update only the mutable parameters for agent states - constraints stay constant"""
        from datetime import datetime
        
        # CRITICAL: Reset all fixed variables and bounds from previous round
        # Unfix all power variables and reset bounds to original values
        for agent_idx in range(self.num_agents):
            for t in range(self.max_horizon):
                # Unfix power_buy and power_sell
                if self.model.power_buy[agent_idx, t].is_fixed():
                    self.model.power_buy[agent_idx, t].unfix()
                if self.model.power_sell[agent_idx, t].is_fixed():
                    self.model.power_sell[agent_idx, t].unfix()
                
                # Reset bounds to original (0, 1000) from variable definition
                self.model.power_buy[agent_idx, t].setlb(0)
                self.model.power_buy[agent_idx, t].setub(1000)
                self.model.power_sell[agent_idx, t].setlb(0)
                self.model.power_sell[agent_idx, t].setub(1000)
        
        # Unfix and reset EV external charging variables and driving energy parameters
        if self.num_evs > 0:
            for ev_idx in range(self.num_evs):
                for t in range(self.max_horizon):
                    if self.model.power_out_ev[ev_idx, t].is_fixed():
                        self.model.power_out_ev[ev_idx, t].unfix()
                    self.model.power_out_ev[ev_idx, t].setlb(0)
                    self.model.power_out_ev[ev_idx, t].setub(1000)
                    
                    # Reset driving energy parameter to 0 for all timesteps
                    ev_driving_param = getattr(self.model, f'ev_driving_energy_{ev_idx}')
                    ev_driving_param[t].set_value(0.0)
                
                # Reset SOC bounds to original constraint bounds
                for t in range(self.max_horizon + 1):
                    self.model.soc_ev[ev_idx, t].setlb(0)
                    self.model.soc_ev[ev_idx, t].setub(10000)
        
        # Unfix heat pump temperature variables
        if self.num_heatpumps > 0:
            for hp_idx in range(self.num_heatpumps):
                for t in range(self.max_horizon + 1):
                    if self.model.T_hp[hp_idx, t].is_fixed():
                        self.model.T_hp[hp_idx, t].unfix()
        
        # Helper functions
        def clamp(val, min_val, max_val):
            return max(min(val, max_val), min_val)
        
        def get_profile_series(agent_data, year):
            # Get profile data for specific year
            return agent_data.get(f"profile_df_{year}", agent_data.get('profile_df', {}))
        
        # Create mappings from unique_id to required indices (built once per optimization)
        unique_id_to_agent_idx = {}  # Maps to position in self.agent_data
        unique_id_to_battery_idx = {}  # Maps to battery index in Pyomo model
        unique_id_to_ev_idx = {}      # Maps to EV index in Pyomo model  
        unique_id_to_hp_idx = {}      # Maps to heat pump index in Pyomo model
        
        battery_idx = 0
        ev_idx = 0
        hp_idx = 0
        
        for agent_idx, agent_data in enumerate(self.agent_data):
            unique_id = agent_data['unique_id']
            unique_id_to_agent_idx[unique_id] = agent_idx
            
            if agent_data['flex'] == 1:  # Battery
                unique_id_to_battery_idx[unique_id] = battery_idx
                battery_idx += 1
            elif agent_data['flex'] == 2:  # EV
                unique_id_to_ev_idx[unique_id] = ev_idx
                ev_idx += 1
            elif agent_data['flex'] == 3:  # Heat pump
                unique_id_to_hp_idx[unique_id] = hp_idx
                hp_idx += 1
        
        # Load ambient temperature data once for all time steps
        # Update ambient temperature values (needed for EVs and heat pumps)
        if self.num_evs > 0 or self.num_heatpumps > 0:
            T_amb_values = {}
            for t in range(n):
                time_point = current_date + t * self.timestep
                T_amb = self.temperature_df.loc[
                    self.temperature_df['time'] == time_point, 'Temperatur-Dortmund'
                ]
                if T_amb.empty:
                    T_amb_values[t] = 15.0  # Default ambient temperature
                else:
                    T_amb_values[t] = float(T_amb.values[0])
                self.model.T_amb[t].set_value(T_amb_values[t])

        # Update parameters using unique_id-based approach
        for unique_id, state in agent_states.items():
            if unique_id not in unique_id_to_agent_idx:
                continue  # Skip agents not in this optimizer
            
            agent_data = next(a for a in self.agent_data if a['unique_id'] == unique_id)
            agent_idx = unique_id_to_agent_idx[unique_id]
            
            if agent_data['flex'] == 1:  # Battery
                battery_idx = unique_id_to_battery_idx[unique_id]
                current_soc = clamp(state['soc'], self.SOC_MIN, self.SOC_MAX)
                self.model.battery_init_soc[battery_idx].set_value(current_soc)
            
            elif agent_data['flex'] == 2:  # EV
                ev_idx = unique_id_to_ev_idx[unique_id]
                # SOC (matching original bounds)
                current_ev_soc = clamp(state['soc'], 
                                     0.1 * agent_data['max_capacity'], 
                                     0.9 * agent_data['max_capacity'])
                self.model.ev_init_soc[ev_idx].set_value(current_ev_soc)
                
                # Handle EV modes based on profile data (matching original)
                for t in range(n):
                    time_point = current_date + t * self.timestep
                    
                    # Use pre-loaded ambient temperature
                    T_amb_val = T_amb_values[t]
                    
                    # Get profile data
                    data_series = get_profile_series(agent_data, time_point.year)
                    
                    # Handle different data_series formats
                    if hasattr(data_series, 'columns') and time_point in data_series.columns:
                        # DataFrame with datetime columns
                        current_value = data_series[time_point].iloc[0]
                    elif hasattr(data_series, 'loc') and time_point in data_series.index:
                        # Series with datetime index
                        current_value = data_series.loc[time_point]
                        if hasattr(current_value, 'values'):
                            current_value = current_value.values[0] if len(current_value.values) > 0 else 2
                    else:
                        current_value = 2  # Default to parked
                    
                    # Mode-based constraints (matching original exactly)
                    if current_value == 1:  # Home charging
                        # Apply temperature efficiency for charging
                        temp_coef = self._temperature_efficiency_charging(T_amb_val)
                        p_loading = agent_data.get('home_base_loading_power') * temp_coef
                        
                        # Fix power limits and SOC dynamics
                        self.model.power_buy[agent_idx, t].setub(p_loading / self.sref)
                        
                        # Fix external charging to 0
                        if hasattr(self.model, 'power_out_ev'):
                            self.model.power_out_ev[ev_idx, t].fix(0)
                        
                        # No driving energy when charging at home
                        ev_driving_param = getattr(self.model, f'ev_driving_energy_{ev_idx}')
                        ev_driving_param[t].set_value(0.0)
                    
                    elif current_value == 2:  # Parked
                        # No charging or discharging
                        self.model.power_buy[agent_idx, t].fix(0)
                        # SOC remains constant
                        if hasattr(self.model, 'power_out_ev'):
                            self.model.power_out_ev[ev_idx, t].fix(0)
                        # Ensure driving energy is 0 when parked (already reset above, but explicit is better)
                        ev_driving_param = getattr(self.model, f'ev_driving_energy_{ev_idx}')
                        ev_driving_param[t].set_value(0.0)
                    
                    elif current_value == 3:  # Driving
                        # Energy consumption based on km driven
                        km_driven = 0
                        
                        # Handle different data_series formats for km data
                        if hasattr(data_series, 'columns') and time_point in data_series.columns:
                            # DataFrame with datetime columns - check if there's a second row for km
                            if len(data_series) > 1:
                                km_driven = data_series[time_point].iloc[1] * 0.8
                        elif hasattr(data_series, 'loc') and time_point in data_series.index:
                            # Series with datetime index
                            km_val = data_series.loc[time_point]
                            if hasattr(km_val, 'values') and len(km_val.values) > 1:
                                km_driven = km_val.values[1] * 0.8
                        
                        # Apply temperature efficiency for driving
                        temp_coef = self._temperature_efficiency_driving(T_amb_val)
                        energy_consumed = agent_data.get('consumption_1km', 0.2) * km_driven * temp_coef  # Convert kWh to Wh
                        
                        # Fix no charging
                        self.model.power_buy[agent_idx, t].fix(0)
                        if hasattr(self.model, 'power_out_ev'):
                            self.model.power_out_ev[ev_idx, t].fix(0)
                        
                        # Set the driving energy parameter for this timestep
                        ev_driving_param = getattr(self.model, f'ev_driving_energy_{ev_idx}')
                        ev_driving_param[t].set_value(energy_consumed)
                    
                    elif current_value == 4:  # Work charging (external)
                        # Apply temperature efficiency for charging
                        temp_coef = self._temperature_efficiency_charging(T_amb_val)
                        p_loading = agent_data.get('work_loading_power', 22000) * temp_coef
                        
                        # Fix home charging to 0
                        self.model.power_buy[agent_idx, t].fix(0)
                        
                        # Set external charging limit
                        if hasattr(self.model, 'power_out_ev'):
                            self.model.power_out_ev[ev_idx, t].setub(p_loading / self.sref)
                        
                        # No driving energy when charging at work
                        ev_driving_param = getattr(self.model, f'ev_driving_energy_{ev_idx}')
                        ev_driving_param[t].set_value(0.0)
            
            elif agent_data['flex'] == 3:  # Heat pump
                hp_idx = unique_id_to_hp_idx[unique_id]
                T_min = agent_data.get('T_min', 18.0)
                T_max = agent_data.get('T_max', 24.0)
                T_set = agent_data.get('T_set', 21.0)
                current_temp = state.get('temperature', T_set)
                current_temp = clamp(current_temp, T_min+0.5, T_max-0.5)
                self.model.hp_init_temp[hp_idx].set_value(current_temp)
                
                # Handle seasonal logic and complex control (matching original)
                for t in range(n):
                    time_point = current_date + t * self.timestep
                    
                    # Use pre-loaded ambient temperature
                    T_amb_val = T_amb_values[t]
                    
                    # Seasonal check (matching original)
                    year = time_point.year
                    is_heating_season = (time_point < datetime(year, 5, 15) or 
                                       time_point > datetime(year, 9, 15))
                    
                    # Check if heating is needed
                    heating_needed = (is_heating_season and current_temp < T_max and 
                                    T_amb_val < T_set and current_temp > T_amb_val)
                    
                    if heating_needed:
                        # Allow heating operation
                        P_max_cap = agent_data.get('P_max_cap', 8000)
                        self.model.power_buy[agent_idx, t].setub(P_max_cap / self.sref)
                    else:
                        # No heating: fix power_buy to 0
                        self.model.power_buy[agent_idx, t].fix(0)
        
        # Rebuild heat pump dynamics constraints based on heating needs
        if self.num_heatpumps > 0:
            self._rebuild_hp_dynamics_constraints(current_date, n, T_amb_values)
        
        # print(f"Updated parameters for {len(agent_states)} agents using unique_ids")
    
    def _rebuild_hp_dynamics_constraints(self, current_date, n, T_amb_values):
        """Rebuild heat pump temperature dynamics constraints each round based on heating needs"""
        from datetime import datetime
        
        # Remove old dynamics constraints if they exist
        for hp_idx in range(self.num_heatpumps):
            # Remove all timestep-specific constraints from previous round
            for t in range(self.max_horizon):
                constraint_name = f'hp_temp_dynamics_{hp_idx}_t{t}'
                if hasattr(self.model, constraint_name):
                    self.model.del_component(constraint_name)
        
        # Rebuild dynamics constraints based on current conditions
        for hp_idx, (hp_pyomo_idx, agent_idx, C_adapted, cop, timestep_hours) in enumerate(self.hp_indices):
            agent_data = self.agent_data[agent_idx]
            
            T_min = agent_data.get('T_min', 18.0)
            T_max = agent_data.get('T_max', 24.0)
            T_set = agent_data.get('T_set', 21.0)
            R = agent_data.get('R', 0.01)
            
            # Get current temperature from initial temp parameter
            current_temp = self.model.hp_init_temp[hp_pyomo_idx].value
            
            # Build constraint for each timestep
            def make_hp_dynamics_rule(hp_idx_val, agent_idx_val, t_step):
                time_point = current_date + t_step * self.timestep
                T_amb_val = T_amb_values[t_step]
                
                # Check heating conditions (matching original exactly)
                year = time_point.year
                is_heating_season = (time_point < datetime(year, 5, 15) or time_point > datetime(year, 9, 15))
                
                def dynamics_rule(m):
                    if is_heating_season:
                        # Check if heating is actually needed
                        if current_temp <= T_max and T_max > T_amb_val:
                            # Apply physics-based dynamics when heating
                            temp_change = (1 / (R * C_adapted) * (m.T_amb[t_step] - m.T_hp[hp_idx_val, t_step]) + 
                                          m.Qdot_hp[hp_idx_val, t_step] / C_adapted) * 0.25
                            return m.T_hp[hp_idx_val, t_step + 1] == m.T_hp[hp_idx_val, t_step] + temp_change
                        else:
                            # Heating season but conditions not met: T[t+1] = T_set
                            return m.T_hp[hp_idx_val, t_step + 1] == T_max
                    else:
                        # Not heating season: T[t+1] = T_set
                        return m.T_hp[hp_idx_val, t_step + 1] == T_set
                
                return dynamics_rule
            
            # Create constraint for each timestep - need n constraints to set T[1] through T[n]
            # Constraints at t=0,1,...,n-1 set T[1], T[2], ..., T[n]
            for t in range(n):
                rule = make_hp_dynamics_rule(hp_pyomo_idx, agent_idx, t)
                constraint_name = f'hp_temp_dynamics_{hp_pyomo_idx}_t{t}'
                self.model.add_component(constraint_name, Constraint(rule=rule))
    
    def _temperature_efficiency_charging(self, T_amb):
        """Calculate temperature efficiency coefficient for EV charging"""
        if T_amb < 0:
            return 0.5
        elif T_amb < 15:
            return 0.7
        elif T_amb > 35:
            return 0.8
        else:
            return 1.0
    
    def _temperature_efficiency_driving(self, T_amb):
        """Calculate temperature efficiency coefficient for EV driving (energy consumption multiplier)"""
        if T_amb < 0:
            return 1.3
        elif T_amb > 35:
            return 1.1
        else:
            return 1.0
    
    def _update_unflexible_profiles(self, current_date, n):
        """Update power values for unflexible agents by fixing their variables"""
        def provide_a_power_range(agent_data, time):
            try:
                power_profile = agent_data['power_profile']
                power = power_profile.loc[power_profile["time"] == time, "power"]
                if power.empty:
                    power = 0
                else:
                    power = power.values[0]
                return power / self.sref  # Scale by reference power
            except Exception as e:
                return 0
        
        # Update unflexible agents by fixing their power variables
        for agent_idx, agent_data in enumerate(self.agent_data):
            if agent_data['flex'] == 0:  # Unflexible
                for t in range(n):
                    time_point = current_date + t * self.timestep
                    power = provide_a_power_range(agent_data, time_point)
                    
                    if power >= 0:
                        # Positive power = consumption (buy)
                        self.model.power_buy[agent_idx, t].fix(power)
                        self.model.power_sell[agent_idx, t].fix(0)
                    else:
                        # Negative power = generation (sell)
                        self.model.power_buy[agent_idx, t].fix(0)
                        self.model.power_sell[agent_idx, t].fix(-power)
                
                # Unfix variables beyond horizon
                for t in range(n, self.max_horizon):
                    self.model.power_buy[agent_idx, t].unfix()
                    self.model.power_sell[agent_idx, t].unfix()
    
    def _extract_results_LEC(self, n):
        """Extract results from the solved optimization model following original algorithm"""
        import numpy as np
        
        # Extract model variables
        power_vector = np.array([value(self.model.power_net[t]) for t in range(n)])
        cost_energy = np.array([value(self.model.cost_energy[t]) for t in range(n)])
        cost_charge_ext = np.array([value(self.model.cost_charge_ext[t]) for t in range(n)])
        
        # Extract power_buy and power_sell for all agents
        power_vector_buy = np.zeros((self.num_agents, n))
        power_vector_sell = np.zeros((self.num_agents, n))
        for i in range(self.num_agents):
            for t in range(n):
                power_vector_buy[i, t] = value(self.model.power_buy[i, t])
                power_vector_sell[i, t] = value(self.model.power_sell[i, t])
        
        # Extract power_out for EVs
        if self.num_evs > 0:
            power_vector_out = np.zeros((self.num_evs, n))
            for ev_idx in range(self.num_evs):
                for t in range(n):
                    power_vector_out[ev_idx, t] = value(self.model.power_out_ev[ev_idx, t])
        
        # Get prices from energy_price_df
        price_data = self.energy_price_df[
            (self.energy_price_df['time'] >= self.current_date) & 
            (self.energy_price_df['time'] < self.current_date + n * self.timestep)
        ]["price"].reset_index(drop=True)
        
        
        # Initialize results dictionary
        agent_results = {}
        
        # Process each agent following the original algorithm
        i = 0
        ev_counter = 0
        
        for agent_data in self.agent_data:
            if agent_data['flex'] != 999 and agent_data['flex'] != 0:
                agent_id = agent_data['unique_id']
                
                # Initialize arrays
                self_supply_buy = np.zeros(int(min(n, self.max_horizon)))
                self_supply_sell = np.zeros(int(min(n, self.max_horizon)))
                buy_price = np.zeros(int(min(n, self.max_horizon)))
                sell_price = np.inf * np.ones(int(min(n, self.max_horizon)))
                power_out = np.zeros(int(min(n, self.max_horizon)))
                price_out = np.zeros(int(min(n, self.max_horizon)))
                
                # Calculate prices for each timestep
                for counter in range(0, int(min(n, self.max_horizon))):
                    if abs(power_vector[counter]) >= 0.00001: 
                        cost = (cost_energy[counter] / power_vector[counter]) / (100 / self.sref) * 4
                    else:
                        cost = price_data.iloc[counter] + self.gridfee_LEC + self.levies_LEC+0.1

                    if agent_data['flex'] == 1:  # Battery
                        if power_vector[counter] ==0:
                            if abs(power_vector_buy[i, counter]) > 0.00001:
                                    buy_price[counter] = price_data.iloc[counter]+self.gridfee+self.levies+self.margin_buy-0.3
                            if abs(power_vector_sell[i, counter]) > 0.00001:
                                    # Was spot+gridfee+levies+margin_buy-0.5 (= spot+16.2), the bid
                                    # line above with the variable name swapped. An ask is only a
                                    # willingness-to-sell threshold -- settlement pays every seller
                                    # the slack price -- so a fee-inflated ask never clears. Matches
                                    # the power_vector != 0 branch below.
                                    sell_price[counter] = price_data.iloc[counter] -self.margin_sell-0.1
                        if power_vector[counter]!=0:
                            if abs(power_vector_sell[i, counter]) > 0.00001:
                                    sell_price[counter] = price_data.iloc[counter] -self.margin_sell-0.1
                            if abs(power_vector_buy[i, counter]) > 0.00001:
                                    buy_price[counter] = price_data.iloc[counter]+self.gridfee+self.levies+self.margin_buy+0.1

                    else: 
                        if abs(power_vector_buy[i, counter]) > 0.00001:
                            if power_vector[counter] <= 0.00001:
                                self_supply_buy[counter] = 1
                                buy_price[counter] = price_data.iloc[counter]+self.gridfee+self.levies+self.margin_buy-0.3
                            else:
                                buy_price[counter] = cost
                    
                    if agent_data['flex'] == 2:  # EV
                        if power_vector_out[ev_counter, counter] > 0.00001:
                            power_out_sum = np.sum(power_vector_out[:, counter])
                            if power_out_sum > 0:
                                price_out[counter] = (cost_charge_ext[counter] / power_out_sum) / (100 / self.sref) * 4
                            power_out[counter] = power_vector_out[ev_counter, counter]
                
                # Find max buy_price where self_supply_buy == 0, then replace all buy_price values with that max
                if agent_data['flex'] != 1:
                    # Exclude the last 4 entries from consideration
                    mask = self_supply_buy == 0
                    if len(buy_price) > 4:
                        mask[-4:] = False
                    external_buy_prices = buy_price[mask]
                    if len(external_buy_prices) > 0:
                        max_external_buy_price = np.max(external_buy_prices)
                        buy_price[self_supply_buy==0] = max_external_buy_price
                

                    # Initialize result dictionary for this agent
                    agent_results[agent_id] = {
                        'max_buy_price': buy_price
                    }
                
                if agent_data['flex'] == 1:  # Battery
                    # No fallback bid/ask on steps where the LP planned neither charge nor
                    # discharge: buy_price stays 0 and sell_price stays inf, which agents.py
                    # reads as "post nothing" for the bid and the ask respectively.
                    #
                    # The removed fallback priced idle-step bids at gridfee_LEC+levies_LEC
                    # (spot+8.95), below the 15.7 ct/kWh the market welfare objective charges
                    # on external imports, so it could only ever clear against local PV.
                    # Repricing it to the external fees is NOT the fix -- it fires on every
                    # idle step, so the battery would bid above market continuously and charge
                    # to full regardless of price. The LP's plan is the signal; the plan-driven
                    # bids above already cross the fee hurdle deliberately.
                    agent_results[agent_id] = {
                        'max_buy_price': buy_price
                    }
                    agent_results[agent_id]['min_sell_price'] = sell_price
                
                if agent_data['flex'] == 2:  # EV
                    agent_results[agent_id]['price_out'] = price_out
                    agent_results[agent_id]['power_out'] = power_out
             
                

            
            i += 1
            if agent_data['flex'] == 2:
                ev_counter += 1
        
        return agent_results
    
    def _extract_results_standard(self, n):
        """Extract results from the solved optimization model - only power vectors"""
        import numpy as np
        
        # Extract power_buy and power_sell for all agents
        power_vector_buy = np.zeros((self.num_agents, n))
        power_vector_sell = np.zeros((self.num_agents, n))
        for i in range(self.num_agents):
            for t in range(n):
                power_vector_buy[i, t] = value(self.model.power_buy[i, t])
                power_vector_sell[i, t] = value(self.model.power_sell[i, t])
        
        # Extract power_out for EVs
        if self.num_evs > 0:
            power_vector_out = np.zeros((self.num_evs, n))
            for ev_idx in range(self.num_evs):
                for t in range(n):
                    power_vector_out[ev_idx, t] = value(self.model.power_out_ev[ev_idx, t])
        
        
        # Initialize results dictionary
        agent_results = {}
        
        # Process each agent - only store power vectors
        i = 0
        ev_counter = 0
        
        for agent_data in self.agent_data:
            if agent_data['flex'] != 999 and agent_data['flex'] != 0:
                agent_id = agent_data['unique_id']
                
                # Initialize result dictionary for this agent with power vectors only
                agent_results[agent_id] = {
                    'optimal_power_buy': power_vector_buy[i, :int(min(n, self.max_horizon))]
                }
                if max(power_vector_buy[i,:])>2:
                    print("unrealsitic power")
                
                if agent_data['flex'] == 1:  # Battery
                    agent_results[agent_id]['optimal_power_sell'] = power_vector_sell[i, :int(min(n, self.max_horizon))]
                
                if agent_data['flex'] == 2:  # EV
                    agent_results[agent_id]['optimal_power_out'] = power_vector_out[ev_counter, :int(min(n, self.max_horizon))]
                    ev_counter += 1
            
            i += 1
        
        return agent_results

