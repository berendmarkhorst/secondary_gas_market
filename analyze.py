import pickle
import json
import matplotlib.pyplot as plt

input_file = "Results/result2"
nr_hours = 4
production_values = False
booked_capacity = True
flow_values = False
storage_values = False

# Load the problem instance back from the file
with open(f"{input_file}.pkl", "rb") as file:
    problem = pickle.load(file)

# Read a dictionary from a json file
with open(f"{input_file}.json", "r") as file:
    solution = json.load(file)

# Plot production values
if production_values:
    for m in problem.third_stages:
        production = {k.name: {} for k in problem.commodities}
        for n in m.nodes:
            if problem.digraph.nodes()[n.name]['Type'] == "Field":
                for k in problem.commodities:
                    production[k.name][n.name] = sum(solution[f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]"] if f"q_production[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]" in solution.keys() else 0 for t in problem.traders)

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

if booked_capacity:
    # Plot booked capacity values
    for t in problem.traders:
        for m in problem.third_stages:
            # Only print the last hours of each stage.
            # if m.hour == nr_hours: #len(m.all_parents) == 2 + nr_hours - 1:
            booked_capacity = {}
            parents = [m_tilde for m_tilde in m.all_parents if m_tilde.hour == m.hour or m_tilde.stage_id == 1] + [m]
            parents = sorted(parents, key=lambda x: x.stage_id)

            for idx, p in enumerate(parents):
                x_plus = sum(solution[f"x_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"x_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in problem.commodities)
                x_minus = sum(solution[f"x_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"x_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in problem.commodities)
                y_plus = sum(solution[f"y_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"y_plus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in problem.commodities)
                y_minus = sum(solution[f"y_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]"] if f"y_minus[{n.node_id},{p.stage_id},{t.trader_id},{k.commodity_id}]" in solution.keys() else 0 for n in p.nodes for k in problem.commodities)

                if p.stage_id > 1:
                    previous = booked_capacity[f"Scenario node {parents[idx-1].stage_id}"]
                else:
                    previous = 0

                booked_capacity[f"Scenario node {p.stage_id}"] = x_plus - y_plus + previous

            plt.bar(booked_capacity.keys(), booked_capacity.values())
            plt.title(f'Booked capacity for trader {t.trader_id}')
            plt.show()


if flow_values:
    # Plot flow values
    for m in problem.third_stages:
        flows = {}
        for a in m.arcs:
            flows[(a.source, a.sink)] = {"Source": a.source, "Sink": a.sink, "Name": a.name}
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
            total_flow = data['gas'] + data['hydrogen']
            if total_flow > 0.0001:  # Only plot edges with total flow > 0.01
                source = data['Source']
                sink = data['Sink']
                source_x = problem.digraph.nodes()[source]['X_coordinate']
                source_y = problem.digraph.nodes()[source]['Y_coordinate']
                sink_x = problem.digraph.nodes()[sink]['X_coordinate']
                sink_y = problem.digraph.nodes()[sink]['Y_coordinate']

                # Plot the edge
                plt.plot([source_x, sink_x], [source_y, sink_y],
                         linewidth=total_flow/25, color='black', zorder=1)  # Line width corresponds to the total flow

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
            storage[n.name] = sum(solution[f"v[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]"] if f"v[{t.trader_id},{n.node_id},{m.stage_id},{k.commodity_id}]" in solution.keys() else 0 for t in problem.traders for k in problem.commodities)

        plt.bar([str(k) for k in storage.keys()], storage.values())

        # Rotate the x labels 45 degrees
        plt.xticks(rotation=45)

        plt.title(f'Storage values at scenario node {m.stage_id}')
        plt.ylabel("Storage values")
        plt.xlabel("Nodes")

        plt.show()