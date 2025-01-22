import pickle
import json
import matplotlib.pyplot as plt
import vis
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np

data_file = "Data/OurData2.xlsx"
input_file = "Results/result_C9"
first_stage_constraint = False

nr_hours = 4
production_values = False
booked_capacity = False
flow_values = False
storage_values = False
benefit_large_traders = False
interactive_plot = False
first_stage_capacity = False
profit_per_trader = False
second_hand_market = False

def read_data(input_file):
    # Load the problem instance back from the file
    with open(f"{input_file}.pkl", "rb") as file:
        problem = pickle.load(file)

    # Read a dictionary from a json file
    with open(f"{input_file}.json", "r") as file:
        solution = json.load(file)

    return problem, solution

problem, solution = read_data(input_file)

# Check if being less strict leads to an efficiency increase
def experiment_b(beta_values):
    for beta in beta_values:
        for m in problem.stages:
            for n in m.nodes:
                n.allowed_percentage = beta
        # for n in problem.stages[0].nodes:
        #     n.allowed_percentage = beta

        output_file = f"Results/result_B4{beta}"

        # Save the problem instance to a file
        with open(f"{output_file}.pkl", "wb") as file:
            pickle.dump(problem, file)

        model = problem.build_model(first_stage_constraint=first_stage_constraint)

        # Write gurobi output to file
        model.setParam('OutputFlag', 1)
        model.setParam('LogFile', f"{output_file}.log")

        # Set a maximum runtime of 10 hours
        model.setParam('TimeLimit', 36000)

        model.optimize()
        problem.save_solution(model, f"{output_file}.json")


# experiment_b(np.linspace(0.5, 1, 11))
# experiment_b(np.linspace(0, 0.9, 19))

def get_value_from_solution(key, solution):
    return solution[key] if key in solution.keys() else 0

eps = 1e-6

def get_objective(problem, solution):
    storage_costs_dict = {}
    sales_dict = {}
    production_costs_dict = {}
    flow_costs_dict = {}
    entry_costs_dict = {}
    exit_costs_dict = {}
    objective = 0

    # First part of the objective
    for m in problem.stages:
        for k in problem.commodities:
            for n in m.nodes:
                for t in problem.traders:
                    supplier_entry_costs = (get_value_from_solution(f"x_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)) * n.entry_costs[(t, k)]
                    supplier_exit_costs = (get_value_from_solution(f"x_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)) * n.exit_costs[(t, k)]

                    entry_costs_dict[m, n, t, k] = m.probability * supplier_entry_costs
                    exit_costs_dict[m, n, t, k] = m.probability * supplier_exit_costs

                    objective -= problem.stages[m.stage_id - 1].probability * (supplier_entry_costs + supplier_exit_costs)

                    if m.name == "intra day" or first_stage_constraint:
                        production_costs = get_value_from_solution(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) * n.production_costs[(t, k)]
                        storage_costs = get_value_from_solution(f"w_plus[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) * n.storage_costs[(t, k)]
                        flow_costs = sum(get_value_from_solution(f"f[{t.trader_id},{a[0]},{a[1]},{m.stage_id},{k.commodity_id}]", solution) * m.get_arc(a).arc_costs[k] for a in problem.incoming_arcs[n.node_id])
                        sales = sum(get_value_from_solution(f"q_sales[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id},{d}]", solution) * n.sales_prices[t, d] for d in problem.d_dict[k])

                        storage_costs_dict[m, n, t, k] = problem.stages[m.stage_id - 1].probability * storage_costs
                        sales_dict[m, n, t, k] = problem.stages[m.stage_id - 1].probability * sales
                        production_costs_dict[m, n, t, k] = problem.stages[m.stage_id - 1].probability * production_costs
                        flow_costs_dict[m, n, t, k] = problem.stages[m.stage_id - 1].probability * flow_costs

                        objective += m.probability * (sales - production_costs - storage_costs - flow_costs)

    return objective, sales_dict, production_costs_dict, storage_costs_dict, flow_costs_dict, entry_costs_dict, exit_costs_dict

def overview_objective():
    profit, sales_dict, production_costs_dict, storage_costs_dict, flow_costs_dict, entry_costs_dict, exit_costs_dict = get_objective(problem, solution)
    print("Profit is", profit)

    # Make a pie chart for the 7 components of the objective
    labels = ['Profit', 'Production costs', 'Storage costs', 'Flow costs', 'Entry costs', 'Exit costs']
    sizes = [profit, sum(production_costs_dict.values()), sum(storage_costs_dict.values()), sum(flow_costs_dict.values()), sum(entry_costs_dict.values()), sum(exit_costs_dict.values())]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red', 'blue']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title("Objective function components")
    plt.show()

overview_objective()

def entry_and_exit_capacity_bought():
    total_entry_capacity_bought = {}
    total_exit_capacity_bought = {}

    print("Entry cap sold by TSO", sum(get_value_from_solution(f"s_plus[{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for k in problem.commodities))
    print("Exit cap sold by TSO", sum(get_value_from_solution(f"s_minus[{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for k in problem.commodities))

    total_entry_capacity_bought[1] = sum(get_value_from_solution(f"x_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_HOI_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for t in problem.traders for k in problem.commodities if m.name == "long term")
    total_entry_capacity_bought[2] = sum(m.probability * (get_value_from_solution(f"x_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_HOI_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)) for m in problem.second_stages for n in m.nodes for t in problem.traders for k in problem.commodities)
    total_entry_capacity_bought[3] = sum(m.probability * (get_value_from_solution(f"x_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_HOI_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)) for m in problem.third_stages for n in m.nodes for t in problem.traders for k in problem.commodities)

    total_exit_capacity_bought[1] = sum(get_value_from_solution(f"x_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_HOI_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) for m in problem.stages for n in problem.stages[0].nodes for t in problem.traders for k in problem.commodities if m.name == "long term")
    total_exit_capacity_bought[2] = sum(m.probability * (get_value_from_solution(f"x_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_HOI_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)) for m in problem.second_stages for n in m.nodes for t in problem.traders for k in problem.commodities)
    total_exit_capacity_bought[3] = sum(m.probability * (get_value_from_solution(f"x_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) - get_value_from_solution(f"y_HOI_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)) for m in problem.third_stages for n in m.nodes for t in problem.traders for k in problem.commodities)

    # Extract keys and values
    keys = list(total_entry_capacity_bought.keys())
    values1 = list(total_entry_capacity_bought.values())
    values2 = list(total_exit_capacity_bought.values())

    breakpoint()

    # Define the width of the bars
    bar_width = 0.25

    # Define the positions of the bars
    r1 = range(len(keys))
    r2 = [x + bar_width for x in r1]

    # Create the bar plots
    plt.bar(r1, values1, width=bar_width, label="Entry capacity")
    plt.bar(r2, values2, width=bar_width, label="Exit capacity")

    # Add labels and title
    plt.xlabel('Stages')
    plt.ylabel('Capacity Bought')
    plt.title('Entry and exit capacity bought in each stage')

    # Add the x-ticks
    plt.xticks([r + bar_width / 2 for r in range(len(keys))], keys)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

entry_and_exit_capacity_bought()

def impact_minimum_contracts(input_files, labels):
    result = {}
    for input_file, label in zip(input_files, labels):
        problem, solution = read_data(input_file)

        profit, sales_dict, production_costs_dict, storage_costs_dict, flow_costs_dict, entry_costs_dict, exit_costs_dict = get_objective(problem, solution)

        result[label] = profit

    # Sort the dictionary by the first key
    result = dict(sorted(result.items()))

    plt.plot([i for i in result.keys()], list(result.values()), "-o")
    plt.title("Profit different minimum contract levels for Equinor")
    plt.show()

impact_minimum_contracts([f"Results/result_C{level}" for level in range(10)], list(range(10)))

def deviations_in_storage():
    all_storage = {}

    for m in problem.third_stages:
        all_storage[((m.stage_id - 21) // 4 + 1, m.hour)] = sum(get_value_from_solution(f"w_plus[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for n in m.nodes for t in problem.traders for k in problem.commodities)

    # Plot heatmap of storage values. On x-axis, we have the first element of the tuple of the key and on the y-axis, we have the second element of the tuple of the key
    plt.imshow([[all_storage[(i, j)] for i in range(1, 17)] for j in range(1, 5)], cmap='hot', interpolation='nearest')

    # Add colorbar
    plt.colorbar()

    # Change y-values to [1,2,3,4]
    plt.yticks([0, 1, 2, 3], [1, 2, 3, 4])

    # Change x-values to [1,...,16]
    plt.xticks([i for i in range(16)], [i for i in range(1, 17)])

    plt.title("Storage values")

    plt.show()

deviations_in_storage()

def deviations_in_production():
    production = {}
    all_production = {}
    for m in problem.third_stages:
        all_production[((m.stage_id - 21) // 4 + 1, m.hour)] = sum(get_value_from_solution(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for n in m.nodes for t in problem.traders for k in problem.commodities)
        for n in m.nodes:
            if n.name not in production.keys():
                production[n.name] = []
            production[n.name] += [sum(get_value_from_solution(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for t in problem.traders for k in problem.commodities)]

    # Plot heatmap of production values. On x-axis, we have the first element of the tuple of the key and on the y-axis, we have the second element of the tuple of the key
    plt.imshow([[all_production[(i, j)] for i in range(1, 17)] for j in range(1, 5)], cmap='hot', interpolation='nearest')

    # Add colorbar
    plt.colorbar()

    # Change y-values to [1,2,3,4]
    plt.yticks([0, 1, 2, 3], [1, 2, 3, 4])

    # Change x-values to [1,...,16]
    plt.xticks([i for i in range(16)], [i for i in range(1, 17)])

    # Add labels
    plt.xlabel("Scenario number")
    plt.ylabel("Hour")

    plt.title("Production values")

    plt.show()

    # Remove all nodes with no production
    production = {k: v for k, v in production.items() if sum(v) > 0}

    # Sort the dictionary by the average production
    production = dict(sorted(production.items(), key=lambda item: sum(item[1]) / len(item[1]), reverse=True))

    plt.bar(production.keys(), [sum(p) / len(p) for p in production.values()], color="blue", zorder=2, label="Average production")
    plt.bar(production.keys(), [max(p) for p in production.values()], color="red", zorder=1, label="Maximum production")
    plt.bar(production.keys(), [min(p) for p in production.values()], color="green", zorder=3, label="Minimum production")
    plt.title("Production per node, aggregated over all scenario nodes")

    plt.xticks(rotation=90)

    plt.legend()

    plt.show()

deviations_in_production()

breakpoint()

def profit_per_unit():
    profit, sales_dict, production_costs_dict, storage_costs_dict, flow_costs_dict, entry_costs_dict, exit_costs_dict = get_objective(problem, solution)

    return profit / sum(get_value_from_solution(f"q_sales[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id},gas_or_mix]", solution) for m in problem.third_stages for n in m.nodes for t in problem.traders for k in problem.commodities)

print("Profit per unit is", profit_per_unit())

def get_total_entry_capacity_sold_first_stage(problem, solution):
    # Total entry capacity sold in the first stage
    total_entry_capacity_first_stage = sum(get_value_from_solution(f"s_plus[{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for k in problem.commodities)

    return total_entry_capacity_first_stage

total_entry_capacity_first_stage = get_total_entry_capacity_sold_first_stage(problem, solution)
print("Total entry capacity first stage", total_entry_capacity_first_stage)

def different_betas(input_files):
    result = {}
    result2 = {}
    for input_file in input_files:
        problem, solution = read_data(input_file)

        kpis = get_objective(problem, solution)
        result2[(problem.stages[0].nodes[0].allowed_percentage, problem.third_stages[0].nodes[0].allowed_percentage)] = kpis[0]

        total_entry_capacity_first_stage = get_total_entry_capacity_sold_first_stage(problem, solution)

        result[(problem.stages[0].nodes[0].allowed_percentage, problem.third_stages[0].nodes[0].allowed_percentage)] = total_entry_capacity_first_stage

    # Sort the dictionary by the first key
    result = dict(sorted(result.items()))
    result2 = dict(sorted(result2.items()))

    plt.plot([i for i, j in result.keys()], list(result.values()), "-o")
    plt.title("Total entry capacity sold in the first stage")
    plt.show()

    plt.plot([i for i, j in result2.keys()], list(result2.values()), "-o")
    plt.title("Profit")
    plt.show()

different_betas([f"Results/result_B4{beta}" for beta in list(np.linspace(0, 0.9, 19)) + [0.95, 0.96, 0.97, 0.98, 0.99, 1]])

# Total sales
total_sales = 0
total_production = 0
for m in problem.third_stages:
    total_sales += m.probability * sum(get_value_from_solution(f"q_sales[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id},gas_or_mix]", solution) for n in m.nodes for t in problem.traders for k in problem.commodities)
    total_production += m.probability * sum(get_value_from_solution(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for n in m.nodes for t in problem.traders for k in problem.commodities)
print("Total sales", total_sales)
print("Total production", total_production)

# # Find connected components
# temp = problem.digraph.to_undirected()
# components = nx.connected_components(temp)
#
# # Get subgraphs of each component
# subgraphs = [temp.subgraph(c).copy() for c in components]
#
# for subgraph in subgraphs:
#     print(subgraph.nodes())

sinks = ["EMDEN", "DORNUM", "ST.FERGUS", "EASINGTON", "TEESSIDE", "ZEEBRUGGE", "DUNKERQUE", "POLAND"]

# for node in problem.digraph.nodes():
#     break
#     if node not in sinks:
#         # Check if capacity incoming equals capacity outgoing
#         in_capacity = sum([a["Capacity"] for _, _, a in problem.digraph.in_edges(node, data=True)])
#         out_capacity = sum([a["Capacity"] for _, _, a in problem.digraph.out_edges(node, data=True)])
#
#         production_capacity = [n.production_capacity[problem.commodities[0]] for n in problem.stages[0].nodes if n.name == node][0]
#
#         # If the difference is bigger than 10 units, print the node
#         if production_capacity + in_capacity > out_capacity:
#             print(node, production_capacity + in_capacity, out_capacity)

# for node in problem.digraph.nodes():
#     break
#     production_capacity = [n.production_capacity[problem.commodities[0]] for n in problem.stages[0].nodes if n.name == node][0]
#
#     # Find all outgoing arcs from this node, and sum the capacities
#     capacity = sum([a["Capacity"] for _, _, a in problem.digraph.out_edges(node, data=True)])
#
#     if capacity < production_capacity:
#         print(node, capacity, production_capacity)

# # Find min cut
# temp_graph = problem.digraph.copy()
# 
# for node in problem.digraph.nodes():
#     if node not in sinks:
#         temp_graph.add_edge(f"dummy_{node}", node)
#         # Add capacity to the edge
#         temp_graph[f"dummy_{node}"][node]['Capacity'] = [n.production_capacity[problem.commodities[0]] for n in problem.stages[0].nodes if n.name == node][0]
# 
# # Add a super sink and connect all sinks
# super_sink = 'super_sink'
# for sink in sinks:
#     temp_graph.add_edge(sink, super_sink)
#     # Add capacity to the edge
#     temp_graph[sink][super_sink]['Capacity'] = 10000
# 
# super_source = 'super_source'
# temp_graph.add_node(super_source)
# for node in temp_graph.nodes():
#     if node.startswith("dummy"):
#         temp_graph.add_edge(super_source, node)
#         # Add capacity to the edge
#         temp_graph[super_source][node]['Capacity'] = 10000
# 
# value, cut = nx.minimum_cut(temp_graph, super_source, super_sink, capacity="Capacity")
# print("Min-cut is", value)
# 
# flow_value, flow_dict = nx.maximum_flow(temp_graph, super_source, super_sink, capacity = "Capacity")
# 
# print("Maximum flow value:", flow_value)
# 
# bottlenecks = []
# for u, v, data in temp_graph.edges(data=True):
#     if not u.startswith("dummy"):
#         flow = flow_dict[u][v]  # Flow along this edge
#         capacity = data['Capacity']  # Edge capacity
#         name = data["Name"] + f" {u} -> {v}" if "Name" in data.keys() else f"{u} -> {v}"
#         if flow == capacity:  # Fully saturated edge
#             bottlenecks.append(name)
# 
# print("Bottleneck edges:", bottlenecks)

# for m in problem.third_stages:
#     for n in m.nodes:
#         for t in problem.traders:
#             lhs = sum(get_value_from_solution(f"x_plus[{n.node_id},{m_tilde.stage_id},{t.trader_id},{k.commodity_id}]") - get_value_from_solution(f"y_plus[{n.node_id},{m_tilde.stage_id},{t.trader_id},{k.commodity_id}]") for k in problem.commodities for m_tilde in m.all_parents + [m] if m.hour == m_tilde.hour)
#             rhs = sum(get_value_from_solution(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]") for k in problem.commodities)
#             # if lhs != rhs:
#             #     breakpoint()

capacity_used = []
for m in problem.third_stages:
    for a in m.arcs:
        if a.arc_capacity > 0 :
            flow = sum(get_value_from_solution(f"f[{t.trader_id},{a.source},{a.sink},{m.stage_id},{k.commodity_id}]", solution) for t in problem.traders for k in problem.commodities)
            capacity_used += [flow/a.arc_capacity]

print("Average capacity used:", sum(capacity_used) / len(capacity_used))
print("Number of arcs used on full capacity:", sum(1 for c in capacity_used if c >= 1) / len(problem.third_stages))
print("Max capacity used:", max(capacity_used))

production_used = []
for m in problem.third_stages:
    for n in m.nodes:
        if n.production_capacity[problem.commodities[0]] > 0:
            q_production = sum(get_value_from_solution(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for t in problem.traders for k in problem.commodities)
            production_used += [q_production / n.production_capacity[problem.commodities[0]]]
            # if q_production < n.production_capacity[problem.commodities[0]] and q_production > 0:
                # print("Production capacity not used:", n.name, q_production, n.production_capacity[problem.commodities[0]])
                # pass

print("Average production used:", sum(production_used) / len(production_used))
print("Number of nodes used on full production capacity:", sum(1 for c in production_used if c >= 1) / len(problem.third_stages))
print("Max production used:", max(production_used))

all_storage = sum(m.probability * get_value_from_solution(f"v[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for t in problem.traders for m in problem.third_stages for n in m.nodes for k in problem.commodities)
print("Total storage", all_storage)

if second_hand_market:
    y_plus = sum(get_value_from_solution(f"y_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for t in problem.traders for k in problem.commodities)
    y_minus = sum(get_value_from_solution(f"y_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for t in problem.traders for k in problem.commodities)
    print("Second hand market entry:", y_plus)
    print("Second hand market exit:", y_minus)

    # Now, I want to compute the same but per trader
    y_plus_per_trader = {t.name: sum(get_value_from_solution(f"y_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for k in problem.commodities) for t in problem.traders}
    y_minus_per_trader = {t.name: sum(get_value_from_solution(f"y_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution) for m in problem.stages for n in m.nodes for k in problem.commodities) for t in problem.traders}
    print("Second hand market entry per trader:", y_plus_per_trader)
    print("Second hand market exit per trader:", y_minus_per_trader)

if profit_per_trader:
    profits = {t: 0 for t in problem.traders}

    for t in problem.traders:
        for m in problem.stages:
            for k in problem.commodities:
                for n in m.nodes:
                    x_plus = get_value_from_solution(f"x_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)
                    x_minus = get_value_from_solution(f"x_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)
                    y_plus = get_value_from_solution(f"y_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)
                    y_minus = get_value_from_solution(f"y_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]", solution)
                    supplier_entry_costs = (x_plus - y_plus) * n.entry_costs[(t, k)]
                    supplier_exit_costs = (x_minus - y_minus) * n.exit_costs[(t, k)]

                    profits[t] += m.probability * (supplier_entry_costs + supplier_exit_costs)

                    if m.name == "intra day" or first_stage_constraint:
                        q_production = get_value_from_solution(f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution)
                        storage = get_value_from_solution(f"v[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]", solution)

                        flow_costs = sum(get_value_from_solution(f"f[{t.trader_id},{a[0]},{a[1]},{m.stage_id},{k.commodity_id}]", solution) * m.get_arc(a).arc_costs[k] for a in problem.incoming_arcs[n.node_id])
                        production_costs = q_production * n.production_costs[(t, k)]
                        storage_costs = storage * n.storage_costs[(t, k)]

                        sales = sum(get_value_from_solution(f"q_sales[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id},{d}]", solution) * n.sales_prices[t, d] for d in problem.d_dict[k])

                        profits[t] += m.probability * (sales - production_costs - storage_costs - flow_costs)

    # Make a horizontal bar plot of the profits
    plt.barh([t.name for t in profits.keys()], profits.values())
    plt.show()
    print(profits)

if first_stage_capacity:
    s_plus = []
    s_minus = []

    for final_stage in problem.stages:
        if final_stage.name == "long term" and final_stage.hour == nr_hours:
            s_plus += [sum(get_value_from_solution(f"s_plus[{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for m in final_stage.all_parents + [final_stage] for n in m.nodes for k in problem.commodities)]
            s_minus += [sum(get_value_from_solution(f"s_minus[{n.node_id},{m.stage_id},{k.commodity_id}]", solution) for m in final_stage.all_parents + [final_stage] for n in m.nodes for k in problem.commodities)]

    print("First stage capacity entry:", sum(s_plus) / len(s_plus))
    print("First stage capacity exit:", sum(s_minus) / len(s_minus))

if interactive_plot:
    nodes_df = pd.read_excel(data_file, sheet_name="Nodes", skiprows=1)
    nodes_df["Lon"] = nodes_df["X_coordinate"]
    nodes_df["Lat"] = nodes_df["Y_coordinate"]

    arcs_df = pd.read_excel(data_file, sheet_name="Arcs")

    # Drop all the rows with missing geometry
    arcs_df = arcs_df.dropna(subset=['geometry'])

    gdf_pipes = gpd.GeoDataFrame(arcs_df, geometry=gpd.GeoSeries.from_wkt(arcs_df['geometry']))

    k = [commodity for commodity in problem.commodities if commodity.name == "gas"][0]

    # All names in both dataframes should be lowercase
    nodes_df["Name"] = nodes_df["Name"].str.lower()
    gdf_pipes["Name"] = gdf_pipes["Name"].str.lower()

    for m in problem.stages:
        for idx, row in gdf_pipes.iterrows():
            source = row["Source"]
            sink = row["Sink"]
            flow1 = sum(solution[f"f[{t.trader_id},{source},{sink},{m.stage_id},{k.commodity_id}]"] if f"f[{t.trader_id},{source},{sink},{m.stage_id},{k.commodity_id}]" in solution.keys() else 0 for t in problem.traders)
            flow2 = sum(solution[f"f[{t.trader_id},{sink},{source},{m.stage_id},{k.commodity_id}]"] if f"f[{t.trader_id},{sink},{source},{m.stage_id},{k.commodity_id}]" in solution.keys() else 0 for t in problem.traders)
            gdf_pipes.loc[(gdf_pipes["Source"] == source) & (gdf_pipes["Sink"] == sink), "Flow"] = flow1 + flow2

    vis.interactive_plot(gdf_pipes, nodes_df, extra_info=True)

if benefit_large_traders:
    delta_capacities_entry = {t: 0 for t in problem.traders}
    delta_capacities_exit = {t: 0 for t in problem.traders}

    first_stage_capacity_entry = {t: 0 for t in problem.traders}
    first_stage_capacity_exit = {t: 0 for t in problem.traders}

    for t in problem.traders:
        for m in problem.stages:
            for n in m.nodes:
                # for k in problem.commodities:
                k = [commodity for commodity in problem.commodities if commodity.name == "gas"][0]
                x_plus = solution[f"x_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]"] if f"x_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0
                x_minus = solution[f"x_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]"] if f"x_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0
                y_plus = solution[f"y_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]"] if f"y_plus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0
                y_minus = solution[f"y_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]"] if f"y_minus[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0

                if m.name == "long term":
                    first_stage_capacity_entry[t] += x_plus - y_plus
                    first_stage_capacity_exit[t] += x_minus - y_minus
                else:
                    delta_capacities_entry[t] += m.probability * (x_plus - y_plus)
                    delta_capacities_exit[t] += m.probability * (x_minus - y_minus)
    ratios_entry = {t.name: delta_capacities_entry[t] / (first_stage_capacity_entry[t]+eps) for t in problem.traders}
    ratios_exit = {t.name: delta_capacities_exit[t] / (first_stage_capacity_exit[t]+eps) for t in problem.traders}

    print("Entry ratios:", ratios_entry)
    # plt.bar(ratios_entry.keys(), ratios_entry.values())
    # plt.title("Entry ratios")
    # plt.show()

    print("Exit ratios:", ratios_exit)
    # plt.bar(ratios_exit.keys(), ratios_exit.values())
    # plt.title("Exit ratios")
    # plt.show()

# Plot production values
if production_values:
    for m in problem.third_stages:
        production = {k.name: {} for k in problem.commodities}
        for n in m.nodes:
            # if problem.digraph.nodes()[n.name]['Type'] == "Field":
            for k in problem.commodities:
                production[k.name][n.name] = sum(solution[f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]"] if f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]" in solution.keys() else 0 for t in problem.traders)

        # Make all keys lowercase strings
        production["gas"] = {k.lower(): v for k, v in production["gas"].items()}

        # Sort the keys from the dictionary alphabetically
        production["gas"] = dict(sorted(production["gas"].items(), reverse=True))

        # Plot the production values
        plt.barh([str(k) for k in production["gas"].keys()], production["gas"].values())

        plt.ylabel("Locations")
        plt.xlabel("Values");
        plt.title(f'Production in scenario node {m.stage_id}')
        plt.show()

if booked_capacity:
    # Plot booked capacity values
    for t in problem.traders:
        for m in problem.third_stages:
            # Only print the last hours of each stage.
            # if m.hour == nr_hours: #len(m.all_parents) == 2 + nr_hours - 1:
            booked_capacity = {}
            parents = [m_tilde for m_tilde in m.all_parents if m_tilde.hour == m.hour] + [m]
            parents = sorted(parents, key=lambda x: x.stage_id)

            for idx, p in enumerate(parents):
                x_plus = sum(solution[f"x_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"x_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in [problem.commodities[0]])
                x_minus = sum(solution[f"x_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"x_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in [problem.commodities[0]])
                y_plus = sum(solution[f"y_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"y_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in [problem.commodities[0]])
                y_minus = sum(solution[f"y_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"y_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in [problem.commodities[0]])

                if p.name != "long term":
                    previous = booked_capacity[f"Scenario node {parents[idx-1].stage_id}"]
                else:
                    previous = 0

                booked_capacity[f"Scenario node {p.stage_id}"] = x_plus - y_plus + previous

            plt.bar(booked_capacity.keys(), booked_capacity.values())
            plt.title(f'Booked entry capacity for trader {t.name}')
            plt.show()


if flow_values:
    # Plot flow values
    for m in problem.third_stages:
        flows = {}
        for a in m.arcs:
            flows[(a.source, a.sink)] = {"Source": a.source, "Sink": a.sink, "Name": a.name, "Capacity": a.arc_capacity}
            for k in problem.commodities:
                flows[(a.source, a.sink)][k.name] = sum(solution[f"f[{t.trader_id},{a.source},{a.sink},{m.stage_id},{k.commodity_id}]"] if f"f[{t.trader_id},{a.source},{a.sink},{m.stage_id},{k.commodity_id}]" in solution.keys() else 0 for t in problem.traders)

        # Initialize the plot
        plt.figure(figsize=(8, 6))

        # Plot nodes and annotate with names
        for node in problem.digraph.nodes():
            x = problem.digraph.nodes()[node]['X_coordinate']
            y = problem.digraph.nodes()[node]['Y_coordinate']
            plt.scatter(x, y, s=100, color='blue', zorder=4)  # Plot the node
            plt.text(x, y, node, fontsize=12, ha='right', color='black', zorder=3)  # Annotate the node name

        # Plot edges, with line width corresponding to the sum of flows, and annotate the edge names
        for data in flows.values():
            edge_name = data['Name']
            # Calculate the total flow and filter based on threshold
            total_flow = data['gas'] # + data['hydrogen']
            if total_flow > 0.0001:  # Only plot edges with total flow > 0.01
                source = data['Source']
                sink = data['Sink']
                source_x = problem.digraph.nodes()[source]['X_coordinate']
                source_y = problem.digraph.nodes()[source]['Y_coordinate']
                sink_x = problem.digraph.nodes()[sink]['X_coordinate']
                sink_y = problem.digraph.nodes()[sink]['Y_coordinate']

                # Plot the edge
                line_color = 'red' if total_flow/data["Capacity"] >= 1 else 'black'
                plt.plot([source_x, sink_x], [source_y, sink_y],
                         linewidth=total_flow/25, color=line_color, zorder=1)  # Line width corresponds to the total flow

                # Annotate the edge name at the midpoint
                mid_x = (source_x + sink_x) / 2
                mid_y = (source_y + sink_y) / 2
                plt.text(mid_x, mid_y, edge_name, fontsize=12, color='gray', ha='center', zorder=2)

        # Show the plot
        plt.title(f'Network flows at scenario node {m.stage_id}')
        plt.axis('off')
        plt.show()

if storage_values:
    # Storage values
    for m in problem.third_stages:
        storage = {}
        for n in m.nodes:
            storage[n.name] = sum(solution[f"w_plus[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]"] if f"w_plus[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]" in solution.keys() else 0 for t in problem.traders for k in problem.commodities)

        # Make all keys lowercase strings
        storage = {k.lower(): v for k, v in storage.items()}

        # Sort the keys from the dictionary alphabetically
        storage = dict(sorted(storage.items(), reverse=True))

        # Plot the storage values
        plt.barh([str(k) for k in storage.keys()], storage.values())

        # Make the value on the y-axis red if the string is a sink
        for i, key in enumerate(storage.keys()):
            if key in [s.lower() for s in sinks]:
                plt.gca().get_yticklabels()[i].set_color('red')

        plt.title(f"Insert into storage at scenario node {m.stage_id}")
        plt.xlabel("Storage values")
        plt.ylabel("Nodes")

        plt.show()