import networkx as nx
import gurobipy as gp
from typing import List, Dict, Tuple


class Stage:
    def __init__(self, stage_id: int, arc_costs: Dict[Tuple[int, int], float], entry_costs: Dict[int, float],
                 exit_costs: Dict[int, float],
                 arc_capacities: Dict[Tuple[int, int], float], node_capacities: Dict[int, float],
                 probability: float, node_demands: Dict[int, float], production_costs: Dict[int, float],
                 production_capacities: Dict[Tuple[int, int], float]):
        self.stage_id = stage_id
        self.arc_costs = arc_costs
        self.entry_costs = entry_costs
        self.exit_costs = exit_costs
        self.arc_capacities = arc_capacities
        self.node_capacities = node_capacities
        self.probability = probability
        self.node_demands = node_demands
        self.production_costs = production_costs
        self.production_capacities = production_capacities


class Problem:
    def __init__(self, digraph: nx.Graph, stages: List[Stage], traders: List[int], loss_rate: float,
                 allowed_percentage: float):
        self.digraph = digraph
        self.nodes = list(digraph.nodes)
        self.arcs = list(digraph.edges)
        self.incoming_arcs = {node: [arc for arc in self.arcs if arc[1] == node] for node in self.nodes}
        self.outgoing_arcs = {node: [arc for arc in self.arcs if arc[0] == node] for node in self.nodes}
        self.stages = stages
        self.stage_ids = [stage.stage_id for stage in self.stages]  # Starts at 1!
        self.stage_ids_star = self.stage_ids[1:]  # Without 1!
        self.traders = traders
        self.loss_rate = loss_rate
        self.allowed_percentage = allowed_percentage


    def build_model(self):
        model = gp.Model("Stochastic Secondary Energy Market")

        # Decision variables
        x_plus = model.addVars(self.nodes, self.stage_ids, self.traders, name="x_plus", lb=0.0)
        x_minus = model.addVars(self.nodes, self.stage_ids, self.traders, name="x_minus", lb=0.0)
        y_plus = model.addVars(self.nodes, self.stage_ids, self.traders, name="y_plus", lb=0.0)
        y_minus = model.addVars(self.nodes, self.stage_ids, self.traders, name="y_minus", lb=0.0)
        s_plus = model.addVars(self.nodes, self.stage_ids, name="s_plus", lb=0.0)
        s_minus = model.addVars(self.nodes, self.stage_ids, name="s_minus", lb=0.0)
        f = model.addVars(self.traders, self.arcs, self.stage_ids, name="f", lb=0.0)
        q_sales = model.addVars(self.traders, self.nodes, self.stage_ids, name="q_sales", lb=0.0)
        q_production = model.addVars(self.traders, self.nodes, self.stage_ids, name="q_production", lb=0.0)

        # Objective function
        obj = gp.quicksum(self.stages[m-1].probability * gp.quicksum(self.stages[m-1].entry_costs[n] * x_plus[n, m, t] + self.stages[m-1].exit_costs[n] * x_minus[n, m, t] +
                                                         self.stages[m-1].production_costs[n] * q_production[t, n, m] +
                                                         gp.quicksum(self.stages[m-1].arc_costs[a] * f[t, a[0], a[1], m] for a in self.incoming_arcs[n]) for t in self.traders)
                          for m in self.stage_ids for n in self.nodes)

        model.setObjective(obj, gp.GRB.MINIMIZE)

        # TSO constraints
        # Equation 1b
        for m in self.stage_ids:
            for a in self.arcs:
                model.addConstr(gp.quicksum(f[t, a[0], a[1], m] for t in self.traders) <= self.stages[m-1].arc_capacities[a],
                                name="eq1b")

        # Equation 1c
        for n in self.nodes:
            for m in self.stage_ids_star:
                lhs = gp.quicksum((1 - self.loss_rate) * f[t, a[0], a[1], m] + q_production[t, n, m] for a in self.incoming_arcs[n] for t in self.traders) - gp.quicksum(f[t, a[0], a[1], m] for a in self.outgoing_arcs[n] for t in self.traders)
                rhs = gp.quicksum(x_minus[n, 1, t] + x_minus[n, m, t] - y_minus[n, m, t] for t in self.traders)
                model.addConstr(lhs >= rhs, name="eq1c")

        # Equation 1d
        for n in self.nodes:
            for m in self.stage_ids_star:
                lhs = gp.quicksum(f[t, a[0], a[1], m] + q_sales[t, n, m] for a in self.outgoing_arcs[n] for t in self.traders) - gp.quicksum((1 - self.loss_rate) * f[t, a[0], a[1], m] for a in self.incoming_arcs[n] for t in self.traders)
                rhs = gp.quicksum(x_plus[n, 1, t] + x_plus[n, m, t] - y_plus[n, m, t] for t in self.traders)
                model.addConstr(lhs >= rhs, name="eq1d")

        # Equation 1e
        for n in self.nodes:
            for m in self.stage_ids_star:
                model.addConstr(s_minus[n, m] <= self.stages[m-1].node_capacities[n] - gp.quicksum(x_minus[n, 1, t] for t in self.traders),
                                name="eq1e")

        # Equation 1f
        for n in self.nodes:
            for m in self.stage_ids_star:
                model.addConstr(s_plus[n, m] <= self.stages[m-1].node_capacities[n] - gp.quicksum(x_plus[n, 1, t] for t in self.traders),
                                name="eq1f")

        # Equation 1g
        for n in self.nodes:
            lhs = gp.quicksum(x_minus[n, 1, t] for t in self.traders)
            rhs = self.allowed_percentage * self.stages[0].node_capacities[n]
            model.addConstr(lhs <= rhs, name="eq1g")

        # Equation 1h
        for n in self.nodes:
            lhs = gp.quicksum(x_plus[n, 1, t] for t in self.traders)
            rhs = self.allowed_percentage * self.stages[0].node_capacities[n]
            model.addConstr(lhs <= rhs, name="eq1h")

        # Equation 1i
        for n in self.nodes:
            for m in self.stage_ids_star:
                model.addConstr(gp.quicksum(q_sales[t, n, m] for t in self.traders) == self.stages[m-1].node_demands[n], name="eq1i")

        # Supplier constraints
        # Equation 1j
        for n in self.nodes:
            for m in self.stage_ids:
                for t in self.traders:
                    model.addConstr(q_production[t, n, m] <= self.stages[m-1].production_capacities[(t, n)], name="eq1j")

        # Equation 1k
        for n in self.nodes:
            for m in self.stage_ids:
                for t in self.traders:
                    lhs = q_production[t, n, m] + gp.quicksum((1 - self.loss_rate) * f[t, a[0], a[1], m] for a in self.incoming_arcs[n])
                    rhs = q_sales[t, n, m] + gp.quicksum(f[t, a[0], a[1], m] for a in self.outgoing_arcs[n])
                    model.addConstr(lhs == rhs, name="eq1k")

        # Equation 1l
        for n in self.nodes:
            for m in self.stage_ids_star:
                for t in self.traders:
                    model.addConstr(x_minus[n, 1, t] + x_minus[n, m, t] - y_minus[n, m, t] >= q_sales[t, n, m], name="eq1l")

        # Equation 1m
        for n in self.nodes:
            for m in self.stage_ids_star:
                for t in self.traders:
                    model.addConstr(x_plus[n, 1, t] + x_plus[n, m, t] - y_plus[n, m, t] >= q_production[t, n, m], name="eq1m")

        # Market constraints
        # Equation 1n
        for n in self.nodes:
            for m in self.stage_ids_star:
                lhs = gp.quicksum(x_minus[n, m, t] for t in self.traders)
                rhs = s_minus[n, m] + gp.quicksum(y_minus[n, m, t] for t in self.traders)
                model.addConstr(lhs == rhs, name="eq1n")

        # Equation 1o
        for n in self.nodes:
            for m in self.stage_ids_star:
                lhs = gp.quicksum(x_plus[n, m, t] for t in self.traders)
                rhs = s_plus[n, m] + gp.quicksum(y_plus[n, m, t] for t in self.traders)
                model.addConstr(lhs == rhs, name="eq1o")

        return model



