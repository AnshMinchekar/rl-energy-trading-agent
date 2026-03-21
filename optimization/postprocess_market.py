import numpy as np
import pandas as pd
import pandapower as pp
import copy
from pyomo.environ import value
from data.data_handler import Market_StaticData, Agent_Bid_StaticData, Agent_Ask_StaticData

def process_pyomo_results(
    model,
    market_static: Market_StaticData,
    bid_agents: list[Agent_Bid_StaticData],
    ask_agents: list[Agent_Ask_StaticData],
    grid_model
):
    """
    Convert Pyomo optimization results into structured DataFrames.
    Independent of Mesa and Gurobi.
    """

    # -------------------------------------------------------------------------
    # 1. Basic agent info (no Mesa dependency)
    # -------------------------------------------------------------------------
        # 1. Basic agent info
    # ----------------------------------------------------------------------
    bid_info = pd.DataFrame({
        "Agent ID": [a.agent_id for a in bid_agents],
        "Node": [a.bus for a in bid_agents],
        "Agent Type": ["Bid"] * len(bid_agents),
        "Type": [a.bidding_type for a in bid_agents],
        "cosphi": [a.cosphi for a in bid_agents],
    })

    ask_info = pd.DataFrame({
        "Agent ID": [a.agent_id for a in ask_agents],
        "Node": [a.bus for a in ask_agents],
        "Agent Type": ["Ask"] * len(ask_agents),
        "Type": [a.asking_type for a in ask_agents],
        "cosphi": [a.cosphi for a in ask_agents],
    })

    aux_arr = pd.concat([bid_info, ask_info], ignore_index=True)

    # -------------------------------------------------------------------------
    # 2. Extract bid and ask results from Pyomo variables
    # -------------------------------------------------------------------------
    # Example: m.p_bid[i], m.p_ask[j] are Var containers
    bids = [value(model.p_bid[i]) for i in model.BIDS]
    asks = [value(model.p_ask[j]) for j in model.ASKS]

    # Optional: if your model also has bid_value[i] and ask_cost[j]
    try:
        bid_prices = [value(model.bid_value[i]) for i in model.BIDS]
        ask_prices = [value(model.ask_cost[j]) for j in model.ASKS]
    except Exception:
        bid_prices = [0] * len(bids)
        ask_prices = [0] * len(asks)

    # Build demand (bids) DataFrame
    demand_vector = pd.DataFrame({
        "Agent ID": [a.agent_id for a in bid_agents],
        "Price [€]": bid_prices,
        "Energy [kWh]": bids,
    })
    demand_vector["relative Price [€/kWh]"] = demand_vector["Price [€]"] / (demand_vector["Energy [kWh]"].replace(0, np.nan))
    demand_vector["Energy [kWh]"] *= (market_static["timestep"]/(60*60)) * (market_static["s_ref"])
    demand_vector = pd.merge(demand_vector, aux_arr, on="Agent ID", how="left")
    demand_vector.sort_values(by="relative Price [€/kWh]", ascending=False, inplace=True)

    # Build supply (asks) DataFrame
    supply_vector = pd.DataFrame({
        "Agent ID": [a.agent_id for a in ask_agents],
        "Price [€]": ask_prices,
        "Energy [kWh]": asks,
    })
    supply_vector["relative Price [€/kWh]"] = supply_vector["Price [€]"] / (supply_vector["Energy [kWh]"].replace(0, np.nan))
    supply_vector["Energy [kWh]"] *= (market_static["timestep"]/(60*60)) * (market_static["s_ref"])
    supply_vector = pd.merge(supply_vector, aux_arr, on="Agent ID", how="left")
    supply_vector.sort_values(by="relative Price [€/kWh]", inplace=True)

    # -------------------------------------------------------------------------
    #extract network-level data (if modeled in Pyomo)
    # -------------------------------------------------------------------------


    netnod_pw_vector = np.array([value(model.netnod_pw_vector[n]) for n in model.netnod_pw_vector])
    netnod_qw_vector = np.array([value(model.netnod_qw_vector[n]) for n in model.netnod_qw_vector])
    netnod_pw_vector_pos = np.array([value(model.netnod_pw_vector_pos[n]) for n in model.netnod_pw_vector])
    netnod_pw_vector_neg = np.array([value(model.netnod_pw_vector_neg[n]) for n in model.netnod_pw_vector])
    neg_sum = float(np.sum([netnod_pw_vector_neg[i] for i in range(market_static["n_nodes"]) if i != market_static["slack_bus"]]))
    pos_sum = float(np.sum([netnod_pw_vector_pos[i] for i in range(market_static["n_nodes"]) if i != market_static["slack_bus"]]))
    P_lec   = float(value(model.P_lec))
    pw_sc   = np.array([value(model.netnod_pw_vector_sc[n])   for n in range(market_static["n_nodes"])])
    # --- 2) Split pw into LEC part and external part (vector) ---
    # Equivalent to your MVar logic
    if neg_sum != 0 and pos_sum != 0:
        pw_LEC = []
        for n in range(market_static["n_nodes"]):
            if n == market_static["slack_bus"]:
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
    gridfee_levies_lec = float(value(model.gridfee_levies_lec))
    gridfee_levies_ext = float(value(model.gridfee_levies_ext))
    # Allocate only to importing nodes (negative share)
    gridfee_levies_lec_vec = gridfee_levies_lec * np.where(pw_LEC_share < 0, -pw_LEC_share, 0.0) / 100.0
    gridfee_levies_ext_vec = gridfee_levies_ext * np.where(pw_ext_share < 0, -pw_ext_share, 0.0) / 100.0
    gridfee_levies_total   = gridfee_levies_lec_vec + gridfee_levies_ext_vec

    # €/kWh per node (protect divide-by-zero; 1/4 came from your 15-min to hours)
    gridfee_levies_total_kWh = []
    for n in range(market_static["n_nodes"]):
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
    demand_node_sum = demand_vector.groupby("Node")["Energy [kWh]"].sum().to_dict()
    supply_node_sum = supply_vector.groupby("Node")["Energy [kWh]"].sum().to_dict()

    # Convert pw_* (which are in model units) to kWh over the timestep, like you did:
    # your old code: /4 * 100 → that combined your timestep and sref; keep that mapping:
    # Here: kWh_node = pw_node * (timestep_hours) * sref  (adjust to match your scaling exactly)
    # To mimic the old values closely, we’ll keep /4*100 formulation:
    node_SC_kWh  = pw_sc  * (market_static["timestep"]/(60*60)) * (market_static["s_ref"])
    node_EXT_kWh = np.maximum(-pw_ext, 0.0) * (market_static["timestep"]/(60*60)) * (market_static["s_ref"])
    node_LEC_kWh = np.maximum(-pw_LEC, 0.0) * (market_static["timestep"]/(60*60)) * (market_static["s_ref"])

    # Bidders (consumers): split by demand shares at same node
    for _, row in demand_vector.iterrows():
        aid  = int(row["Agent ID"])
        node = int(row["Node"])
        e_kWh = float(row["Energy [kWh]"])

        if demand_node_sum.get(node, 0.0) > 0:
            share = e_kWh / demand_node_sum[node]
            a_sc  = node_SC_kWh[node]  * share
            a_ext = node_EXT_kWh[node] * share
            a_lec = node_LEC_kWh[node] * share
        else:
            a_sc = a_ext = a_lec = 0.0

        agent_pw_sc.loc[len(agent_pw_sc)]   = [aid,  a_sc]
        agent_pw_ext.loc[len(agent_pw_ext)] = [aid,  a_ext]
        agent_pw_lec.loc[len(agent_pw_lec)] = [aid,  a_lec]

    # Suppliers: negative sign (export perspective)
    for _, row in supply_vector.iterrows():
        aid  = int(row["Agent ID"])
        node = int(row["Node"])
        e_kWh = float(row["Energy [kWh]"])

        if supply_node_sum.get(node, 0.0) > 0:
            share = e_kWh / supply_node_sum[node]
            a_sc  = -node_SC_kWh[node]  * share
            a_ext = -np.maximum(pw_ext[node], 0.0) / 4.0 * 100.0 * share
            a_lec = -np.maximum(pw_LEC[node], 0.0) / 4.0 * 100.0 * share
        else:
            a_sc = a_ext = a_lec = 0.0

        agent_pw_sc.loc[len(agent_pw_sc)]   = [aid,  a_sc]
        agent_pw_ext.loc[len(agent_pw_ext)] = [aid,  a_ext]
        agent_pw_lec.loc[len(agent_pw_lec)] = [aid,  a_lec]


    
    # -------------------------------------------------------------------------
    # 4. Run power flow for grid results
    # -------------------------------------------------------------------------
    net2 = copy.deepcopy(grid_model)
    for n, (p, q) in enumerate(zip(netnod_pw_vector, netnod_qw_vector)):
        if p < 0:
            pp.create_load(net2, n, p_mw=-p * market_static["s_ref"] / 1000, q_mvar=-q * market_static["s_ref"] / 1000)
        elif p > 0:
            pp.create_sgen(net2, n, p_mw=p * market_static["s_ref"] / 1000, q_mvar=q * market_static["s_ref"] / 1000)
    pp.runpp(net2)
    v_d_max = abs(net2.res_bus.vm_pu - 1).max() * 100
    I_d_max = net2.res_line.loading_percent.max()
    trafo_load = float(np.hypot(net2.res_ext_grid.p_mw, net2.res_ext_grid.q_mvar).max() * 1000)

    # -------------------------------------------------------------------------
    # 5. Node results
    # -------------------------------------------------------------------------
    nodal_result_vector = pd.DataFrame({
        "Node": range(len(netnod_pw_vector)),
        "HN Power [kWh]": netnod_pw_vector * market_static["s_ref"] * (market_static["timestep"]/(60*60)),
        "Voltage Deviation [%]": (net2.res_bus.vm_pu - 1) * 100,
        "Line Loading [%]": I_d_max,
        "Transformer Load [kVA]": trafo_load,
    })

    # -------------------------------------------------------------------------
    # 6. External results summary
    # -------------------------------------------------------------------------
    external = pd.DataFrame([{
        "Social Welfare [€]": value(model.obj),
        "Max Voltage Deviation [%]": v_d_max,
        "Max Line Loading [%]": I_d_max,
        "Transformer Load [kVA]": trafo_load,
    }])

    # -------------------------------------------------------------------------
    # 7. Package results
    # -------------------------------------------------------------------------
    data = {
        "demand": demand_vector.round(4),
        "supply": supply_vector.round(4),
        "Node_results": nodal_result_vector.round(4),
        "External": external.round(4),
    }
    return data