import pandas as pd
import networkx as nx
from objects import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go

input_file = "Data/FodstadDataSmall.xlsx"
column_fuels = ["Gas", "Hydrogen"]

# Read the data from the Excel file
nodes_df = pd.read_excel(input_file, sheet_name="Nodes", skiprows=1)
arcs_df = pd.read_excel(input_file, sheet_name="Arcs")
probability2_df = pd.read_excel(input_file, sheet_name="Demand Stage 2", skiprows=0, nrows=1)
probability3_df = pd.read_excel(input_file, sheet_name="Demand Stage 3", skiprows=0, nrows=1)
parent3_df = pd.read_excel(input_file, sheet_name="Demand Stage 3", skiprows=1, nrows=1)
parameters_df = pd.read_excel(input_file, sheet_name="Parameters")

# Parameters
nr_stage2_nodes = int(parameters_df[parameters_df["Name"] == "Stage 2 nodes"]["Value"].values[0])
nr_stage3_nodes = int(parameters_df[parameters_df["Name"] == "Stage 3 nodes"]["Value"].values[0])
nr_markets = int(parameters_df[parameters_df["Name"] == "Number markets"]["Value"].values[0])
nr_hours = int(parameters_df[parameters_df["Name"] == "Number hours"]["Value"].values[0])
allowed_percentage = float(parameters_df[parameters_df["Name"] == "Allowed percentage"]["Value"].values[0])
loss_rate = float(parameters_df[parameters_df["Name"] == "Loss rate"]["Value"].values[0])
gamma = float(parameters_df[parameters_df["Name"] == "Gamma"]["Value"].values[0])
nr_shippers = int(parameters_df[parameters_df["Name"] == "Number shippers"]["Value"].values[0])

# Transfer the node data to a Networkx-object.
nodes = []
fuel_cols = []
for fuel in column_fuels:
    fuel_cols += [f"Production Capacity {fuel}", f"Production Costs {fuel}", f"Storage Capacity {fuel}",
           f"Entry Capacity {fuel}", f"Exit Capacity {fuel}", f"Entry Storage Costs {fuel}", f"Entry Costs {fuel} Stage 1",
           f"Exit Costs {fuel} Stage 1", f"Entry Costs {fuel} Stage 2", f"Exit Costs {fuel} Stage 2",
           f"Entry Costs {fuel} Stage 3", f"Exit Costs {fuel} Stage 3"]

for idx, row in nodes_df.iterrows():
    if pd.isna(row["Name"]):
        break
    node = (row["Name"], {col_name: row[col_name] for col_name in
                          ["ID", "Type", "X_coordinate", "Y_coordinate"] + fuel_cols})
    nodes.append(node)

# Transfer the arc data to a Networkx-object.
arcs = []
for idx, row in arcs_df.iterrows():
    arc = (row["Source"], row["Sink"], {col_name: row[col_name] for col_name in ["Name", "ID", "Capacity", "Flow costs Gas", "Flow costs Hydrogen", "Source", "Sink"]})
    arcs.append(arc)

# Make the Networkx-object.
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(arcs)
digraph = graph.to_directed()

# Arc costs and capacities
arc_costs = {(arc, fuel.lower()): digraph.edges()[arc][f"Flow costs {fuel}"] for arc in digraph.edges() for fuel in column_fuels}
arc_capacities = {arc: digraph.edges()[arc]["Capacity"] for arc in digraph.edges()}

# Entry and exit costs
entry_costs1 = {(node, fuel.lower()): digraph.nodes()[node][f"Entry Costs {fuel} Stage 1"] for node in digraph.nodes() for fuel in column_fuels}
entry_costs2 = {(node, fuel.lower()): digraph.nodes()[node][f"Entry Costs {fuel} Stage 2"] for node in digraph.nodes() for fuel in column_fuels}
entry_costs3 = {(node, fuel.lower()): digraph.nodes()[node][f"Entry Costs {fuel} Stage 3"] for node in digraph.nodes() for fuel in column_fuels}
exit_costs1 = {(node, fuel.lower()): digraph.nodes()[node][f"Exit Costs {fuel} Stage 1"] for node in digraph.nodes() for fuel in column_fuels}
exit_costs2 = {(node, fuel.lower()): digraph.nodes()[node][f"Exit Costs {fuel} Stage 2"] for node in digraph.nodes() for fuel in column_fuels}
exit_costs3 = {(node, fuel.lower()): digraph.nodes()[node][f"Exit Costs {fuel} Stage 3"] for node in digraph.nodes() for fuel in column_fuels}

# Production costs and capacities
production_costs = {(t, node, fuel.lower()): digraph.nodes()[node][f"Production Costs {fuel}"] for node in digraph.nodes() for t in range(1, nr_shippers+1) for fuel in column_fuels}
production_capacities = {(t, node, fuel.lower()): digraph.nodes()[node][f"Production Capacity {fuel}"] for node in digraph.nodes() for t in range(1, nr_shippers+1) for fuel in column_fuels}

# Storage costs and capacities
storage_costs = {(t, node, fuel.lower()): digraph.nodes()[node][f"Entry Storage Costs {fuel}"] for node in digraph.nodes() for t in range(1, nr_shippers+1) for fuel in column_fuels}
storage_capacities = {(t, node, fuel.lower()): digraph.nodes()[node][f"Storage Capacity {fuel}"] for node in digraph.nodes() for t in range(1, nr_shippers+1) for fuel in column_fuels}

# Entry and exit capacities
entry_capacity = {(node, fuel.lower()): digraph.nodes()[node][f"Entry Capacity {fuel}"] for node in digraph.nodes() for fuel in column_fuels}
exit_capacity = {(node, fuel.lower()): digraph.nodes()[node][f"Exit Capacity {fuel}"] for node in digraph.nodes() for fuel in column_fuels}

# Node demands
def get_node_demands(sheet_name, skiprows, nr_stage_nodes):
    node_demands = {}
    for stage_node in range(nr_stage_nodes):
        for d in ["gas_or_mix", "pure_hydrogen"]:
            if d == "gas_or_mix":
                cols = [1 + i + nr_markets * stage_node * 2 for i in range(nr_markets)]
            else:
                cols = [1 + nr_markets + i + nr_markets * stage_node * 2 for i in range(nr_markets)]
            demand_df = pd.read_excel(input_file, sheet_name=sheet_name, skiprows=skiprows, usecols=cols)

            # Needed because markets occur multiple times in the same row
            demand_df.columns = [col.split(".")[0] for col in demand_df.columns]
            for hour in range(nr_hours):
                for node in digraph.nodes():
                    if node in demand_df.columns:
                        demand = demand_df.at[hour, node]
                    else:
                        demand = 0
                    node_demands[(stage_node, hour, node, d)] = demand
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
d_dict = {gas: ["gas_or_mix"], hydrogen: ["gas_or_mix", "pure_hydrogen"]}
d_list = set([item for d in d_dict.values() for item in d])

stage_arcs = []
for arc in digraph.edges():
    id = digraph.edges()[arc]["ID"]
    name = digraph.edges()[arc]["Name"]
    source = arc[0]
    sink = arc[1]
    capacity = digraph.edges()[arc]["Capacity"]
    costs = {k: arc_costs[(digraph.edges()[arc]["Source"], digraph.edges()[arc]["Sink"]), k.name] for k in commodities}
    stage_arc = StageArc(id, name, source, sink, capacity, costs)
    stage_arcs.append(stage_arc)


stages = []
for stage_id in range(1, nr_stage2_nodes + nr_stage3_nodes + 2):
    for hour_id in range(1, nr_hours + 1):
        id = (stage_id - 1) * nr_hours + hour_id
        stage_nodes = []
        for node in digraph.nodes():
            node_id = digraph.nodes()[node]["ID"]
            name = node
            if stage_id == 1:
                node_demands_temp = None
                entry_costs_temp = {(t, k): entry_costs1[node, k.name] for k in commodities for t in traders}
                exit_costs_temp = {(t, k): exit_costs1[node, k.name] for k in commodities for t in traders}
            elif stage_id <= nr_stage2_nodes + 1:
                node_demands_temp = {d: node_demands2[stage_id-2, hour_id-1, node, d] for d in d_list}
                entry_costs_temp = {(t, k): entry_costs2[node, k.name] for k in commodities for t in traders}
                exit_costs_temp = {(t, k): exit_costs2[node, k.name] for k in commodities for t in traders}
            else:
                node_demands_temp = {d: node_demands3[stage_id-2-nr_stage2_nodes, hour_id-1, node, d] for d in d_list}
                entry_costs_temp = {(t, k): entry_costs3[node, k.name] for k in commodities for t in traders}
                exit_costs_temp = {(t, k): exit_costs3[node, k.name] for k in commodities for t in traders}

            production_costs_temp = {(t, k): production_costs[t.trader_id, node, k.name] for k in commodities for t in traders}
            production_capacity_temp = {(t, k): production_capacities[t.trader_id, node, k.name] for k in commodities for t in traders}
            storage_costs_temp = {(t, k): storage_costs[t.trader_id, node, k.name] for k in commodities for t in traders}
            storage_capacity_temp = {(t, k): storage_capacities[t.trader_id, node, k.name] for k in commodities for t in traders}
            entry_capacity_temp = {k: entry_capacity[node, k.name] for k in commodities }
            exit_capacity_temp = {k: exit_capacity[node, k.name] for k in commodities }
            allowed_percentage_temp = allowed_percentage

            stage_node = StageNode(node_id, name, node_demands_temp, production_costs_temp, production_capacity_temp,
                                   storage_costs_temp, storage_capacity_temp, entry_capacity_temp, exit_capacity_temp,
                                   entry_costs_temp, exit_costs_temp, allowed_percentage_temp)
            stage_nodes.append(stage_node)

        if stage_id == 1:
            if hour_id > 1:
                parent = stages[-1]
            else:
                parent = None
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
problem = Problem(digraph, stages, traders, loss_rate, commodities, gamma, d_dict)

model = problem.build_model()
model.optimize()

# Plot production values
for m in problem.third_stages:
    break
    production = {k.name: {} for k in problem.commodities}
    for n in m.nodes:
        if digraph.nodes()[n.name]['Type'] == "Field":
            for k in problem.commodities:
                production[k.name][n.name] = sum(model.getVarByName(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]").X for t in problem.traders)

    # Prepare data for plotting
    nodes = list(production['gas'].keys())  # ['Node 1', 'Node 3']
    gas_values = list(production['gas'].values())  # [5.0, 5.0]
    hydrogen_values = list(production['hydrogen'].values())  # [0.0, 0.0]

    # Plotting
    x = range(len(nodes))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, gas_values, width, label='Gas')
    ax.bar([i + width for i in x], hydrogen_values, width, label='Hydrogen')

    # Add labels and title
    ax.set_xlabel('Locations')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(nodes, rotation=45)
    ax.set_ylabel('Values')
    ax.set_title(f'Production in scenario node {m.stage_id}')
    ax.legend()

    plt.show()

# Plot booked capacity values
for t in problem.traders:
    # break
    for m in problem.third_stages:
        # Only print the last hours of each stage.
        if len(m.all_parents) == 2 + nr_hours - 1:
            booked_capacity = {}
            parents = m.all_parents + [m]
            parents = sorted(parents, key=lambda x: x.stage_id)
            for p in parents:
                x_plus = sum(model.getVarByName(f"x_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]").X for n in p.nodes for k in problem.commodities)
                x_minus = sum(model.getVarByName(f"x_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]").X for n in p.nodes for k in problem.commodities)
                y_plus = sum(model.getVarByName(f"y_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]").X for n in p.nodes for k in problem.commodities)
                y_minus = sum(model.getVarByName(f"y_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]").X for n in p.nodes for k in problem.commodities)

                if m.stage_id == 4:
                    print(f"Scenario node {p.stage_id}: x_plus = {x_plus}, x_minus = {x_minus}, y_plus = {y_plus}, y_minus = {y_minus}")

                if p.parent is not None:
                    previous = booked_capacity[f"Scenario node {p.parent.stage_id}"]
                else:
                    previous = 0

                booked_capacity[f"Scenario node {p.stage_id}"] = x_plus - y_plus + previous

            plt.bar(booked_capacity.keys(), booked_capacity.values())
            plt.title(f'Booked capacity for trader {t.trader_id}')
            plt.show()


# Plot flow values
for m in problem.third_stages:
    break
    flows = {}
    for a in m.arcs:
        flows[(a.source, a.sink)] = {"Source": a.source, "Sink": a.sink, "Name": a.name}
        for k in problem.commodities:
            flows[(a.source, a.sink)][k.name] = sum(model.getVarByName(f"f[{t.trader_id},{a.source},{a.sink},{m.stage_id},{k.commodity_id}]").X for t in problem.traders)

    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Plot nodes and annotate with names
    for node in digraph.nodes():
        x = digraph.nodes()[node]['X_coordinate']
        y = digraph.nodes()[node]['Y_coordinate']
        plt.scatter(x, y, s=100, color='blue', zorder=4)  # Plot the node
        plt.text(x, y, node, fontsize=12, ha='right', color='black', zorder=3)  # Annotate the node name

    # Plot edges, with line width corresponding to the sum of flows, and annotate the edge names
    for data in flows.values():
        edge_name = data['Name']
        # Calculate the total flow and filter based on threshold
        total_flow = data['gas'] + data['hydrogen']
        if total_flow > 0.0001:  # Only plot edges with total flow > 0.01
            source = data['Source']
            sink = data['Sink']
            source_x = digraph.nodes()[source]['X_coordinate']
            source_y = digraph.nodes()[source]['Y_coordinate']
            sink_x = digraph.nodes()[sink]['X_coordinate']
            sink_y = digraph.nodes()[sink]['Y_coordinate']

            # Plot the edge
            plt.plot([source_x, sink_x], [source_y, sink_y],
                     linewidth=total_flow/50, color='black', zorder=1)  # Line width corresponds to the total flow

            # Annotate the edge name at the midpoint
            mid_x = (source_x + sink_x) / 2
            mid_y = (source_y + sink_y) / 2
            plt.text(mid_x, mid_y, edge_name, fontsize=12, color='gray', ha='center', zorder=2)

    # Show the plot
    plt.title(f'Network flows at scenario node {m.stage_id}')
    plt.axis('off')
    plt.show()

# Storage values
for m in problem.third_stages:
    storage = {}
    for n in m.nodes:
        storage[n] = sum(model.getVarByName(f"v[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]").X for t in problem.traders for k in problem.commodities)

    print(storage)