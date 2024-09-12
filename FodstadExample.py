import pandas as pd
import networkx as nx
from objects import *

input_file = "Data/FodstadDataSmall.xlsx"

# Read the data from the Excel file
nodes_df = pd.read_excel(input_file, sheet_name="Nodes", skiprows=2, usecols="A:Q")
arcs_df = pd.read_excel(input_file, sheet_name="Arcs", skiprows=2, usecols="A:I")
probability2_df = pd.read_excel(input_file, sheet_name="Demand Stage 2", skiprows=0, nrows=1)
probability3_df = pd.read_excel(input_file, sheet_name="Demand Stage 3", skiprows=0, nrows=1)
parent3_df = pd.read_excel(input_file, sheet_name="Demand Stage 3", skiprows=1, nrows=1)
parameters_df = pd.read_excel(input_file, sheet_name="Parameters")

# Parameters
nr_stage2_nodes = int(parameters_df[parameters_df["Name"] == "Stage 2 nodes"]["Value"].values[0])
nr_stage3_nodes = int(parameters_df[parameters_df["Name"] == "Stage 3 nodes"]["Value"].values[0])
nr_markets = 3 #4
nr_hours = 1
allowed_percentage = float(parameters_df[parameters_df["Name"] == "Allowed percentage"]["Value"].values[0])
loss_rate = float(parameters_df[parameters_df["Name"] == "Loss rate"]["Value"].values[0])
gamma = float(parameters_df[parameters_df["Name"] == "Gamma"]["Value"].values[0])

# Transfer the node data to a Networkx-object.
nodes = []
for idx, row in nodes_df.iterrows():
    if pd.isna(row["Name"]):
        break
    node = (row["Name"], {col_name: row[col_name] for col_name in
                          ["ID", "Type", "Production Capacity", "Production Costs", "Storage Capacity",
                           "Entry Capacity", "Exit Capacity", "Entry Storage Costs", "Entry Costs Stage 1",
                           "Exit Costs Stage 1", "Entry Costs Stage 2", "Exit Costs Stage 2",
                           "Entry Costs Stage 3", "Exit Costs Stage 3"]})
    nodes.append(node)

# Transfer the arc data to a Networkx-object.
arcs = []
for idx, row in arcs_df.iterrows():
    arc = (row["Source"], row["Sink"], {col_name: row[col_name] for col_name in ["Name", "ID", "Capacity", "Flow costs", "Source", "Sink"]})
    arcs.append(arc)

# Make the Networkx-object.
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(arcs)
digraph = graph.to_directed()


# Arc costs and capacities
arc_costs = {(arc, "gas"): digraph.edges()[arc]["Flow costs"] for arc in digraph.edges()}
arc_capacities = {arc: digraph.edges()[arc]["Capacity"] for arc in digraph.edges()}

# Entry and exit costs
entry_costs1 = {(node, "gas"): digraph.nodes()[node]["Entry Costs Stage 1"] for node in digraph.nodes()}
entry_costs2 = {(node, "gas"): digraph.nodes()[node]["Entry Costs Stage 2"] for node in digraph.nodes()}
entry_costs3 = {(node, "gas"): digraph.nodes()[node]["Entry Costs Stage 3"] for node in digraph.nodes()}
exit_costs1 = {(node, "gas"): digraph.nodes()[node]["Exit Costs Stage 1"] for node in digraph.nodes()}
exit_costs2 = {(node, "gas"): digraph.nodes()[node]["Exit Costs Stage 2"] for node in digraph.nodes()}
exit_costs3 = {(node, "gas"): digraph.nodes()[node]["Exit Costs Stage 3"] for node in digraph.nodes()}

# Production costs and capacities
production_costs = {(1, node, "gas"): digraph.nodes()[node]["Production Costs"] for node in digraph.nodes()}
production_capacities = {(1, node, "gas"): digraph.nodes()[node]["Production Capacity"] for node in digraph.nodes()}

# Storage costs and capacities
storage_costs = {(1, node, "gas"): digraph.nodes()[node]["Entry Storage Costs"] for node in digraph.nodes()}
storage_capacities = {(1, node, "gas"): digraph.nodes()[node]["Storage Capacity"] for node in digraph.nodes()}

# Entry and exit capacities
entry_capacity = {(node, "gas"): digraph.nodes()[node]["Entry Capacity"] for node in digraph.nodes()}
exit_capacity = {(node, "gas"): digraph.nodes()[node]["Exit Capacity"] for node in digraph.nodes()}

# Node demands
def get_node_demands(sheet_name, skiprows, nr_stage_nodes):
    node_demands = {}
    for stage_node in range(nr_stage_nodes):
        cols = [1 + i + nr_markets * stage_node * 2 for i in range(nr_markets)]
        demand_df = pd.read_excel(input_file, sheet_name=sheet_name, skiprows=skiprows, usecols=cols)

        # Needed because markets occur multiple times in the same row
        demand_df.columns = [col.split(".")[0] for col in demand_df.columns]
        for hour in range(nr_hours):
            node_demands[(stage_node, hour)] = {}
            for node in digraph.nodes():
                if node in demand_df.columns:
                    demand = demand_df.at[hour, node]
                else:
                    demand = 0
                node_demands[(stage_node, hour)][(node, "gas or mix")] = demand
    return node_demands

node_demands2 = get_node_demands("Demand Stage 2", 3, nr_stage2_nodes)
node_demands3 = get_node_demands("Demand Stage 3", 4, nr_stage3_nodes)

# Probabilities
def is_float(value):
    # Source: ChatGPT
    if pd.isna(value):
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

probabilities2 = [float(p)for p in probability2_df.iloc[0][1:] if is_float(p)]
probabilities3 = [float(p) for p in probability3_df.iloc[0][1:] if is_float(p)]

# Parent nodes!
parents3 = [float(p)for p in parent3_df.iloc[0][1:] if is_float(p)]

# Objects
t1 = Trader(1, "Trader 1")
traders = [t1]
gas = Commodity(1, "gas")
hydrogen = Commodity(2, "hydrogen")
commodities = [gas, hydrogen]

stage_arcs = []
for arc in digraph.edges():
    id = digraph.edges()[arc]["ID"]
    source = arc[0]
    sink = arc[1]
    capacity = digraph.edges()[arc]["Capacity"]
    costs = {gas: arc_costs[(digraph.edges()[arc]["Source"], digraph.edges()[arc]["Sink"]), "gas"], hydrogen: 0}
    stage_arc = StageArc(id, source, sink, capacity, costs)
    stage_arcs.append(stage_arc)


stages = []
for stage_id in range(1, nr_stage2_nodes + nr_stage3_nodes + 2):
    for hour_id in range(1, nr_hours + 1):
        id = (stage_id - 1) * nr_hours + hour_id
        name = f"Node {id}"
        stage_nodes = []
        for node in digraph.nodes():
            node_id = digraph.nodes()[node]["ID"]
            if stage_id == 1:
                node_demands_temp = None
                entry_costs_temp = {(t1, gas): entry_costs1[node, "gas"], (t1, hydrogen): 0}
                exit_costs_temp = {(t1, gas): exit_costs1[node, "gas"], (t1, hydrogen): 0}
            elif stage_id <= nr_stage2_nodes + 1:
                node_demands_temp = {gas: node_demands2[stage_id-2, hour_id-1][(node, "gas or mix")], hydrogen: 0}
                entry_costs_temp = {(t1, gas): entry_costs2[node, "gas"], (t1, hydrogen): 0}
                exit_costs_temp = {(t1, gas): exit_costs2[node, "gas"], (t1, hydrogen): 0}
            else:
                node_demands_temp = {gas: node_demands3[stage_id-2-nr_stage2_nodes, hour_id-1][(node, "gas or mix")], hydrogen: 0}
                entry_costs_temp = {(t1, gas): entry_costs3[node, "gas"], (t1, hydrogen): 0}
                exit_costs_temp = {(t1, gas): exit_costs3[node, "gas"], (t1, hydrogen): 0}

            production_costs_temp = {(t1, gas): production_costs[1, node, "gas"], (t1, hydrogen): 0}
            production_capacity_temp = {(t1, gas): production_capacities[1, node, "gas"], (t1, hydrogen): 0}
            storage_costs_temp = {(t1, gas): storage_costs[1, node, "gas"], (t1, hydrogen): 0}
            storage_capacity_temp = {(t1, gas): storage_capacities[1, node, "gas"], (t1, hydrogen): 0}
            entry_capacity_temp = {gas: entry_capacity[node, "gas"], hydrogen: 0}
            exit_capacity_temp = {gas: exit_capacity[node, "gas"], hydrogen: 0}
            allowed_percentage_temp = allowed_percentage

            stage_node = StageNode(node_id, name, node_demands_temp, production_costs_temp, production_capacity_temp,
                                   storage_costs_temp, storage_capacity_temp, entry_capacity_temp, exit_capacity_temp,
                                   entry_costs_temp, exit_costs_temp, allowed_percentage_temp)
            stage_nodes.append(stage_node)

        if stage_id == 1:
            stage = Stage(id, "long term", 1, stage_nodes, stage_arcs, None)
        elif stage_id <= nr_stage2_nodes + 1:
            if hour_id > 1:
                parent = stages[-1]
            else:
                parent = stages[0]
            stage = Stage(id, "day ahead", probabilities2[stage_id-2], stage_nodes, stage_arcs, parent)
        else:
            if hour_id > 1:
                parent = stages[-1]
            else:
                parent_id = parents3[stage_id - nr_stage2_nodes - 2]
                parent = [stage for stage in stages if stage.stage_id == parent_id][0]
            stage = Stage(id, "intra day", probabilities3[stage_id - nr_stage2_nodes - 2], stage_nodes, stage_arcs, parent)

        stages.append(stage)

# Problem object
problem = Problem(digraph, stages, traders, loss_rate, commodities, gamma)

model = problem.build_model()
model.optimize()