import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple
import json
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix

class Commodity:
    def __init__(self, commodity_id: int, name: str):
        self.commodity_id = commodity_id
        self.name = name


class Trader:
    def __init__(self, trader_id: int, name: str):
        self.trader_id = trader_id
        self.name = name


class StageNode:
    def __init__(self, node_id: int, name: str,
                 node_demands: Dict[Tuple[Commodity, Trader], float], production_costs: Dict[Tuple[Trader, Commodity], float],
                 production_capacities: Dict[Commodity, float],
                 tso_entry_costs: Dict[Commodity, float], tso_exit_costs: Dict[Commodity, float],
                 storage_costs: Dict[Tuple[Trader, Commodity], float],
                 storage_capacity: Dict[Tuple[Trader, Commodity], float],
                 entry_costs: Dict[Tuple[Trader, Commodity], float], exit_costs: Dict[Tuple[Trader, Commodity], float],
                 allowed_percentage: float, sales_prices: Dict[Tuple[Trader, str], float]):
        self.node_id = node_id
        self.name = name
        self.node_demands = node_demands
        self.production_costs = production_costs
        self.production_capacity = production_capacities
        self.tso_entry_costs = tso_entry_costs
        self.tso_exit_costs = tso_exit_costs
        self.storage_costs = storage_costs
        self.storage_capacity = storage_capacity
        self.entry_costs = entry_costs
        self.exit_costs = exit_costs
        self.allowed_percentage = allowed_percentage
        self.sales_prices = sales_prices

    def __repr__(self):
        return f"Node {self.node_id}"


class StageArc:
    def __init__(self, arc_id: int, name: str, source: int, sink: int, arc_capacity: float, arc_costs: Dict[Commodity, float]):
        self.arc_id = arc_id
        self.name = name
        self.source = source
        self.sink = sink
        self.arc_capacity = arc_capacity
        self.arc_costs = arc_costs

    def __repr__(self):
        return f"Arc {self.arc_id}"


class Stage:
    def __init__(self, stage_id: int, name: str, probability: float, nodes: List[StageNode], arcs: List[StageArc],
                 parent: 'Stage', hour: int):
        self.stage_id = stage_id
        self.name = name
        self.probability = probability
        self.nodes = nodes
        self.arcs = arcs
        self.parent = parent
        self.all_parents = self.get_all_parents()
        self.ids_all_parents = self.get_ids_all_parents()
        self.hour = hour

    def __repr__(self):
        return f"Stage {self.stage_id}"

    def get_all_parents(self):
        all_parents = []
        parent = self.parent
        while parent is not None:
            all_parents.append(parent)
            parent = parent.parent
        return all_parents

    def get_ids_all_parents(self):
        return [parent.stage_id for parent in self.all_parents]

    def get_arc(self, id: int):
        """Returns the Arc object with the given source and sink."""
        return [a for a in self.arcs if a.arc_id == id][0]


class Problem:
    def __init__(self, digraph: nx.Graph, stages: List[Stage], traders: List[Trader], loss_rate: float,
                 commodities: List[Commodity], gamma: float, d_dict: Dict[Commodity, List[str]],
                 markets: List[str]):
        self.markets = markets
        self.digraph = digraph
        self.nodes = list(digraph.nodes)
        self.arcs = list(digraph.edges)
        self.arc_ids = [digraph.edges[arc]["ID"] for arc in digraph.edges]
        self.incoming_arcs = {digraph.nodes()[node]["ID"]: [digraph.edges[arc]["ID"] for arc in digraph.edges if arc[1] == node] for node in self.nodes}
        self.outgoing_arcs = {digraph.nodes()[node]["ID"]: [digraph.edges[arc]["ID"] for arc in digraph.edges if arc[0] == node] for node in self.nodes}
        self.stages = stages
        self.stage_ids = [stage.stage_id for stage in self.stages]  # Starts at 1!
        self.stage_ids_star = self.stage_ids[1:]  # Without 1!
        self.first_stage_id = [stage.stage_id for stage in self.stages if stage.name == "long term"][0]
        self.second_stages = [stage for stage in self.stages if stage.name == "day ahead"]
        self.second_stage_ids = [stage.stage_id for stage in self.second_stages]
        self.third_stages = [stage for stage in self.stages if stage.name == "intra day"]
        self.third_stage_ids = [stage.stage_id for stage in self.third_stages]
        self.traders = traders
        self.trader_ids = [trader.trader_id for trader in self.traders]
        self.loss_rate = loss_rate
        self.commodities = commodities
        self.commodity_ids = [commodity.commodity_id for commodity in self.commodities]
        self.gamma = gamma
        self.node_ids = [digraph.nodes()[node]["ID"] for node in digraph.nodes()]
        self.d_dict = d_dict
        self.d_list = set([item for d in d_dict.values() for item in d])
        self.k_dict = {d: [] for d in self.d_list}
        for k, d_all in self.d_dict.items():
            for d in d_all:
                self.k_dict[d].append(k)

    def build_model(self, first_stage_constraint = False):
        model = gp.Model("Stochastic Secondary Energy Market")

        # model.setParam(GRB.Param.MIPFocus, 1)  # Prioritize feasible solutions quickly
        # model.setParam(GRB.Param.Heuristics, 0.9)  # Set to a higher value for more heuristic effort
        # model.setParam("ImproveStartGap", 0.05)  # Set to 5% gap for example
        # model.Params.Presolve = 2  # Aggressive presolve


        # self.commodities = [k for k in self.commodities if k.name != "hydrogen"]
        # self.commodity_ids = [k.commodity_id for k in self.commodities]
        # self.d_list = ["gas_or_mix"]

        # Decision variables
        x_plus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="x_plus", lb=0.0)
        x_minus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="x_minus", lb=0.0)
        y_plus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="y_plus", lb=0.0)
        y_minus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="y_minus", lb=0.0)
        s_plus = model.addVars(self.node_ids, self.stage_ids, self.commodity_ids, name="s_plus", lb=0.0)
        s_minus = model.addVars(self.node_ids, self.stage_ids, self.commodity_ids, name="s_minus", lb=0.0)
        if first_stage_constraint:
            f = model.addVars(self.trader_ids, self.arc_ids, self.stage_ids, self.commodity_ids, name="f", lb=0.0)
            q_sales = model.addVars(self.trader_ids, self.node_ids, self.stage_ids, self.commodity_ids, self.d_list, name="q_sales", lb=0.0)
            q_production = model.addVars(self.trader_ids, self.node_ids, self.stage_ids, self.commodity_ids, name="q_production", lb=0.0)
            v = model.addVars(self.trader_ids, self.node_ids, self.stage_ids, self.commodity_ids, name="v", lb=0.0)
            w_plus = model.addVars(self.trader_ids, self.node_ids, self.stage_ids, self.commodity_ids, name="w_plus", lb=0.0)
            w_minus = model.addVars(self.trader_ids, self.node_ids, self.stage_ids, self.commodity_ids, name="w_minus", lb=0.0)
        else:
            f = model.addVars(self.trader_ids, self.arc_ids, self.third_stage_ids, self.commodity_ids, name="f", lb=0.0)
            q_sales = model.addVars(self.trader_ids, self.node_ids, self.second_stage_ids + self.third_stage_ids, self.commodity_ids, self.d_list, name="q_sales", lb=0.0)
            q_production = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="q_production", lb=0.0)
            v = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="v", lb=0.0)
            w_plus = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="w_plus", lb=0.0)
            w_minus = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="w_minus", lb=0.0)

        # Objective function
        objective = 0

        # Debug helpers
        model._storage_costs = {}
        model._sales = {}
        model._production_costs = {}
        model._flow_costs = {}
        model._entry_costs = {}
        model._exit_costs = {}

        # First part of the objective
        for m in self.stages:
            for k in self.commodities:
                for n in m.nodes:
                    for t in self.traders:
                        supplier_entry_costs = (x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] - y_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id]) * n.entry_costs[(t, k)]
                        supplier_exit_costs = (x_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] - y_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id]) * n.exit_costs[(t, k)]

                        model._entry_costs[m, n, t, k] = supplier_entry_costs
                        model._exit_costs[m, n, t, k] = supplier_exit_costs

                        objective -= self.stages[m.stage_id-1].probability * (supplier_entry_costs + supplier_exit_costs)

                        if m.name == "intra day" or first_stage_constraint:
                            production_costs = q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] * n.production_costs[(t,k)]
                            storage_costs = w_plus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] * n.storage_costs[(t, k)]
                            flow_costs = gp.quicksum(f[t.trader_id, a, m.stage_id, k.commodity_id] * m.get_arc(a).arc_costs[k] for a in self.incoming_arcs[n.node_id])
                            sales = gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] * n.sales_prices[t, d] for d in self.d_dict[k])

                            model._storage_costs[m,n,t,k] = storage_costs
                            model._sales[m,n,t,k] = sales
                            model._production_costs[m,n,t,k] = production_costs
                            model._flow_costs[m,n,t,k] = flow_costs

                            objective += self.stages[m.stage_id - 1].probability * (sales - production_costs - storage_costs - flow_costs)

        model.setObjective(objective, gp.GRB.MAXIMIZE)

        # Bounds of production and sales decision variables
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.third_stages
        for m in self.stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        if n.name not in t.nodes or k.name == "hydrogen":
                            if m in stages:
                                model.addConstr(q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] <= 0, name=f"production_bounds[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

                            # Traders cannot book entry capacity at markets they're not active at!
                            model.addConstr(x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] <= 0, name=f"x_plus_bounds[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

                            if m in stages:
                                for d in self.d_list:
                                    if n.name not in self.markets:
                                        model.addConstr(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] <= 0, name=f"sales_bounds[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id},{d}]")

        # TSO constraints
        # Equation 1b
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.third_stages
        for m in stages:
            for a in self.arc_ids:
                model.addConstr(gp.quicksum(f[t.trader_id, a, m.stage_id, k.commodity_id] for t in self.traders for k in self.commodities) <= m.nodes[0].allowed_percentage * m.get_arc(a).arc_capacity, # self.stages[m.stage_id-1]
                                name=f"eq1b[{m.stage_id},{a}]")

        # Equation 1c
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.second_stages + self.third_stages
        for m in stages:
            for n in m.nodes:
                for d in self.d_list:
                    for t in self.traders:
                        model.addConstr(gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for k in self.k_dict[d]) >= n.node_demands[(d, t)],
                                        name=f"eq1c[{n.node_id},{m.stage_id},{t.trader_id}]")

        # Supplier constraints
        # Equation 1d
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.third_stages
        for m in stages:
            for n in m.nodes:
                for k in self.commodities:
                    model.addConstr(gp.quicksum(q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] for t in self.traders) <= n.production_capacity[k],
                                    name=f"eq1d[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1e
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.third_stages
        for m in stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        lhs = q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + w_minus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + gp.quicksum((1 - self.loss_rate) * f[t.trader_id, a, m.stage_id, k.commodity_id] for a in self.incoming_arcs[n.node_id])
                        rhs = gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for d in self.d_dict[k]) + w_plus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(f[t.trader_id, a, m.stage_id, k.commodity_id] for a in self.outgoing_arcs[n.node_id])
                        model.addConstr(lhs == rhs, name=f"eq1e[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1f
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.second_stages + self.third_stages
        for m in stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m] if m_tilde.hour == m.hour) >= gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for d in self.d_dict[k]),
                                        name=f"eq1f[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]") #  or m_tilde.stage_id == 1

        # Equation 1g
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.third_stages
        for m in stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m] if m_tilde.hour == m.hour) >= q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id],
                                        name=f"eq1g[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]") #  or m_tilde.stage_id == 1

        # Equation 1h1
        if first_stage_constraint:
            for m in self.stages:
                if m.name == "long term":
                    for n in m.nodes:
                        for t in self.traders:
                            for k in self.commodities:
                                model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] == gp.quicksum(w_plus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] - w_minus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] for m_tilde in m.all_parents + [m] if m_tilde.name == "long term"), name=f"eq1h3[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1h2
        if first_stage_constraint:
            for m in self.second_stages:
                for n in m.nodes:
                    for t in self.traders:
                        for k in self.commodities:
                            model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] == gp.quicksum(w_plus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] - w_minus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] for m_tilde in m.all_parents + [m] if m_tilde.name == "day ahead"), name=f"eq1h2[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1h3
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] == gp.quicksum(w_plus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] - w_minus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] for m_tilde in m.all_parents + [m] if m_tilde.name == "intra day"),
                                        name=f"eq1h3[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1i
        if first_stage_constraint:
            stages = self.stages
        else:
            stages = self.third_stages
        for m in stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] <= n.storage_capacity[(t, k)],
                                        name=f"eq1i[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Market constraints
        # Equation 1j
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    lhs = gp.quicksum(x_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    rhs = s_minus[n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(y_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1s[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1k
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    lhs = gp.quicksum(x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    rhs = s_plus[n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(y_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1t[{n.node_id},{m.stage_id},{k.commodity_id}]")

        vars = {"x_plus": x_plus, "x_minus": x_minus, "y_plus": y_plus, "y_minus": y_minus, "s_plus": s_plus, "s_minus": s_minus,
                "f": f, "q_sales": q_sales, "q_production": q_production, "v": v, "w_plus": w_plus, "w_minus": w_minus}

        return model, vars

    def save_solution(self, vars, output_file: str):
        for name, var in vars.items():
            solution = {}
            for keys, value in var.items():
                solution[keys] = value.x

            df = pd.Series(solution).to_frame()

            df.to_csv(f"{output_file}_{name}.csv", sep=";")