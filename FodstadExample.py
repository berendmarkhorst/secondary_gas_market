from objects import *
import pickle


def run_optimizer(input_file, output_file):
    # Read the data from the Excel file
    nodes_df = pd.read_excel(input_file, sheet_name="Nodes", skiprows=1, skipfooter=3)
    arcs_df = pd.read_excel(input_file, sheet_name="Arcs")
    probability2_df = pd.read_excel(input_file, sheet_name="Demand Stage 2", skiprows=0, nrows=1)
    probability3_df = pd.read_excel(input_file, sheet_name="Demand Stage 3", skiprows=0, nrows=1)
    parent3_df = pd.read_excel(input_file, sheet_name="Demand Stage 3", skiprows=1, nrows=1)
    parameters_df = pd.read_excel(input_file, sheet_name="Parameters")
    gas_sales_df = pd.read_excel(input_file, sheet_name="SalesPriceGasMix")
    hydrogen_sales_df = pd.read_excel(input_file, sheet_name="SalesPricePureHydrogen")
    shipper_df = pd.read_excel(input_file, sheet_name="Traders")
    probabilities_df = pd.read_excel(input_file, sheet_name="Probabilities")
    trader_percentages_df = pd.read_excel(input_file, sheet_name="TraderPercentages")

    # Parameters
    nr_stage2_nodes = int(parameters_df[parameters_df["Name"] == "Stage 2 nodes"]["Value"].values[0])
    nr_stage3_nodes = int(parameters_df[parameters_df["Name"] == "Stage 3 nodes"]["Value"].values[0])
    nr_markets = int(parameters_df[parameters_df["Name"] == "Number markets"]["Value"].values[0])
    nr_hours = int(parameters_df[parameters_df["Name"] == "Number hours"]["Value"].values[0])
    allowed_percentage1 = float(parameters_df[parameters_df["Name"] == "Allowed percentage Stage 1"]["Value"].values[0])
    allowed_percentage2 = float(parameters_df[parameters_df["Name"] == "Allowed percentage Stage 2"]["Value"].values[0])
    allowed_percentage3 = float(parameters_df[parameters_df["Name"] == "Allowed percentage Stage 3"]["Value"].values[0])
    loss_rate = float(parameters_df[parameters_df["Name"] == "Loss rate"]["Value"].values[0])
    gamma = float(parameters_df[parameters_df["Name"] == "Gamma"]["Value"].values[0])
    nr_shippers = int(parameters_df[parameters_df["Name"] == "Number shippers"]["Value"].values[0])

    # Transfer the node data to a Networkx-object.
    nodes = []
    fuel_cols = []
    for fuel in column_fuels:
        fuel_cols += [f"TSO entry costs {fuel}", f"TSO exit costs {fuel}", f"Production Costs {fuel}", f"Storage Capacity {fuel}",
               f"Production Capacity {fuel}", f"Entry Storage Costs {fuel}", f"Entry Costs {fuel} Stage 1",
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
    digraph = nx.DiGraph()
    digraph.add_nodes_from(nodes)
    digraph.add_edges_from(arcs)
    # digraph = graph.to_directed()

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
    production_capacities = {(node, fuel.lower()): digraph.nodes()[node][f"Production Capacity {fuel}"] for node in digraph.nodes() for fuel in column_fuels}

    # TSO entry and exit costs
    tso_entry_costs = {(node, fuel.lower()): digraph.nodes()[node][f"TSO entry costs {fuel}"] for node in digraph.nodes() for fuel in column_fuels}
    tso_exit_costs = {(node, fuel.lower()): digraph.nodes()[node][f"TSO exit costs {fuel}"] for node in digraph.nodes() for fuel in column_fuels}

    # Storage costs and capacities
    storage_costs = {(t, node, fuel.lower()): digraph.nodes()[node][f"Entry Storage Costs {fuel}"] for node in digraph.nodes() for t in range(1, nr_shippers+1) for fuel in column_fuels}
    storage_capacities = {(t, node, fuel.lower()): digraph.nodes()[node][f"Storage Capacity {fuel}"] for node in digraph.nodes() for t in range(1, nr_shippers+1) for fuel in column_fuels}

    # Node demands
    def get_node_demands(sheet_name, skiprows, nr_stage_nodes):
        node_demands = {}
        for stage_node in range(nr_stage_nodes):
            for idx_d, d in enumerate(["gas_or_mix", "pure_hydrogen"]):
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
                        node_demands[(stage_node, hour, node, idx_d + 1)] = demand
        return node_demands

    node_demands1 = get_node_demands("Demand Stage 1", 3, 1)
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
    traders = []
    for trader_idx, trader_name in enumerate(shipper_df.columns):
        t = Trader(trader_idx + 1, trader_name)
        t.nodes = shipper_df[t.name].values
        traders.append(t)
    gas = Commodity(1, "gas")
    commodities = [gas]
    d_dict = {gas: [1]}
    d_list = set([item for d in d_dict.values() for item in d])

    stage_arcs = []
    for arc in digraph.edges():
        id = digraph.edges()[arc]["ID"]
        name = digraph.edges()[arc]["Name"]
        source = arc[0]
        sink = arc[1]
        if arcs_df[(arcs_df["Source"] == source) & (arcs_df["Sink"] == sink)].empty:
            capacity = 0
        else:
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

                trader_percentages = {t.name: trader_percentages_df[trader_percentages_df["Trader"] == t.name][name].values[0] if name in trader_percentages_df.columns else 0 for t in traders}

                if stage_id == 1:
                    node_demands_temp = {(d, t): node_demands1[0, hour_id-1, node, d] * trader_percentages[t.name] for d in d_list for t in traders}
                    entry_costs_temp = {(t, k): entry_costs1[node, k.name] for k in commodities for t in traders}
                    exit_costs_temp = {(t, k): exit_costs1[node, k.name] for k in commodities for t in traders}

                    allowed_percentage_temp = allowed_percentage1
                elif stage_id <= nr_stage2_nodes + 1:
                    node_demands_temp = {(d, t): node_demands2[stage_id-2, hour_id-1, node, d] * trader_percentages[t.name] for d in d_list for t in traders}
                    entry_costs_temp = {(t, k): entry_costs2[node, k.name] for k in commodities for t in traders}
                    exit_costs_temp = {(t, k): exit_costs2[node, k.name] for k in commodities for t in traders}

                    allowed_percentage_temp = allowed_percentage2
                else:
                    node_demands_temp = {(d, t): node_demands3[stage_id-2-nr_stage2_nodes, hour_id-1, node, d] * trader_percentages[t.name] for d in d_list for t in traders}
                    entry_costs_temp = {(t, k): entry_costs3[node, k.name] for k in commodities for t in traders}
                    exit_costs_temp = {(t, k): exit_costs3[node, k.name] for k in commodities for t in traders}

                    allowed_percentage_temp = allowed_percentage3

                production_costs_temp = {(t, k): production_costs[t.trader_id, node, k.name] for k in commodities for t in traders}
                production_capacities_temp = {k: production_capacities[node, k.name] for k in commodities} # * int(name in shipper_df[t.name].values)
                tso_entry_costs_temp = {k: tso_entry_costs[node, k.name] for k in commodities}
                tso_exit_costs_temp = {k: tso_exit_costs[node, k.name] for k in commodities}
                storage_costs_temp = {(t, k): storage_costs[t.trader_id, node, k.name] for k in commodities for t in traders}
                storage_capacity_temp = {(t, k): storage_capacities[t.trader_id, node, k.name] for k in commodities for t in traders}

                sales_prices = {}
                for t in traders:
                    if stage_id <= nr_stage2_nodes + 1:
                        if name in gas_sales_df.columns:
                            sales_prices[(t, 1)] = gas_sales_df[gas_sales_df["Hour"] == hour_id][node].values[0] # None
                        else:
                            sales_prices[(t, 1)] = 0
                        continue
                    elif (name.capitalize() == "Emden" or name.capitalize() == "Dornum") and (hour_id == 1 or hour_id == 3):
                        if (stage_id - nr_stage2_nodes - 2) % 4 == 0:
                            extra = 3
                        elif (stage_id - nr_stage2_nodes - 2) % 4 == 1:
                            extra = 3
                        elif (stage_id - nr_stage2_nodes - 2) % 4 == 2:
                            extra = -3
                        elif (stage_id - nr_stage2_nodes - 2) % 4 == 3:
                            extra = -3
                    elif name.capitalize() == "Zeebrugge" and (hour_id == 1 or hour_id == 3):
                        if (stage_id - nr_stage2_nodes - 2) % 4 == 0:
                            extra = 2
                        elif (stage_id - nr_stage2_nodes - 2) % 4 == 1:
                            extra = -2
                        elif (stage_id - nr_stage2_nodes - 2) % 4 == 2:
                            extra = 2
                        elif (stage_id - nr_stage2_nodes - 2) % 4 == 3:
                            extra = -2
                    else:
                        extra = 0

                    if name in gas_sales_df.columns:
                        sales_prices[(t, 1)] = gas_sales_df[gas_sales_df["Hour"] == hour_id][node].values[0] + extra
                    else:
                        sales_prices[(t, 1)] = 0

                    if name in hydrogen_sales_df.columns:
                        sales_prices[(t, "pure_hydrogen")] = hydrogen_sales_df[hydrogen_sales_df["Hour"] == hour_id][node].values[0] + extra
                    else:
                        sales_prices[(t, "pure_hydrogen")] = 0

                stage_node = StageNode(node_id, name, node_demands_temp, production_costs_temp, production_capacities_temp, tso_entry_costs_temp, tso_exit_costs_temp,
                                       storage_costs_temp, storage_capacity_temp,
                                       entry_costs_temp, exit_costs_temp, allowed_percentage_temp, sales_prices)
                stage_nodes.append(stage_node)

            if stage_id == 1:
                if hour_id > 1:
                    parent = stages[-1]
                    # continue
                else:
                    parent = None
                probability = 1 # probabilities_df[probabilities_df["Hour"] == hour_id]["Weight"].values[0]
                stage = Stage(id, "long term", probability, stage_nodes, stage_arcs, parent, hour_id)
            elif 1 < stage_id <= nr_stage2_nodes + 1:
                if hour_id > 1:
                    parent = stages[-1]
                else:
                    parent = stages[nr_hours-1]
                probability = probabilities2[stage_id-2] # * probabilities_df[probabilities_df["Hour"] == hour_id]["Weight"].values[0]
                stage = Stage(id, "day ahead", probability, stage_nodes, stage_arcs, parent, hour_id)
            else:
                if hour_id > 1:
                    parent = stages[-1]
                else:
                    parent_id = parents3[stage_id - nr_stage2_nodes - 2] + (nr_hours - 1)
                    parent = [stage for stage in stages if stage.stage_id == parent_id][0]
                probability = probabilities3[stage_id - nr_stage2_nodes - 2] # * probabilities_df[probabilities_df["Hour"] == hour_id]["Weight"].values[0]
                stage = Stage(id, "intra day", probability, stage_nodes, stage_arcs, parent, hour_id)

            stages.append(stage)

    markets = list(nodes_df[nodes_df["Type"] == "Market"]["Name"].values)

    # Problem object
    problem = Problem(digraph, stages, traders, loss_rate, commodities, gamma, d_dict, markets)

    # Store problem object
    with open(f"{output_file}.pkl", "wb") as file:
        pickle.dump(problem, file)

    # Print how many nodes and edges the graph has
    print("Number of nodes:", digraph.number_of_nodes())
    print("Number of edges:", digraph.number_of_edges())

    # Print how many scenario nodes the problem has
    print(f"Number of scenario nodes: {len(problem.stages)}")

    model, vars = problem.build_model(first_stage_constraint=False)

    # Write gurobi output to file
    model.setParam('OutputFlag', 1)
    model.setParam('LogFile', f"{output_file}.log")

    # Set a maximum runtime of 10 hours
    model.setParam('TimeLimit', 36000)

    model.optimize()
    problem.save_solution(vars, f"{output_file}.csv")

input_file = "Data/OurData2.xlsx"
output_file = "Results/result_v2_A"
column_fuels = ["Gas", "Hydrogen"]

run_optimizer(input_file, output_file)