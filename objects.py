import networkx as nx
import gurobipy as gp
from typing import List, Dict, Tuple


class Commodity:
    def __init__(self, commodity_id: int, name: str):
        self.commodity_id = commodity_id
        self.name = name


class Trader:
    def __init__(self, supplier_id: int, name: str):
        self.supplier_id = supplier_id
        self.name = name


class Node:
    def __init__(self, node_id: int, name: str, node_capacities: Dict[Commodity, float],
                 node_demands: Dict[Commodity, float]):
        self.node_id = node_id
        self.name = name
        self.node_capacities = node_capacities
        self.node_demands = node_demands


class Arc:
    def __init__(self, arc_id: int, source: int, sink: int, arc_capacities: Dict[Commodity, float],
                 arc_costs: Dict[Commodity, float]):
        self.arc_id = arc_id
        self.source = source
        self.sink = sink
        self.arc_capacities = arc_capacities
        self.arc_costs = arc_costs


class Stage:
    def __init__(self, stage_id: int, name: str, arc_costs: Dict[str, Dict[Tuple[int, int], float]], entry_costs: Dict[Tuple[int, str], float],
                 exit_costs: Dict[Tuple[int, str], float],
                 arc_capacities: Dict[Tuple[int, int], float], node_capacities: Dict[Tuple[int, str], float],
                 probability: float, node_demands: Dict[Tuple[int, str], float], production_costs: Dict[Tuple[int, str], float],
                 production_capacities: Dict[Tuple[int, int, str], float],
                 storage_costs: Dict[Tuple[int, int, str], float], storage_capacities: Dict[Tuple[int, int, str], float],
                 parent: 'Stage'):
        self.stage_id = stage_id
        self.name = name
        self.arc_costs = arc_costs
        # Node
        self.entry_costs = entry_costs
        self.exit_costs = exit_costs
        # Arcs
        self.arc_capacities = arc_capacities
        self.node_capacities = node_capacities
        self.probability = probability
        # Node
        self.node_demands = node_demands
        self.production_costs = production_costs
        self.production_capacities = production_capacities
        self.storage_costs = storage_costs
        self.storage_capacities = storage_capacities
        self.parent = parent
        self.all_parents = self.get_all_parents()
        self.ids_all_parents = self.get_ids_all_parents()

    def get_all_parents(self):
        all_parents = []
        parent = self.parent
        while parent is not None:
            all_parents.append(parent)
            parent = parent.parent
        return all_parents

    def get_ids_all_parents(self):
        return [parent.stage_id for parent in self.all_parents]


class Problem:
    def __init__(self, digraph: nx.Graph, stages: List[Stage], traders: List[int], loss_rate: float,
                 allowed_percentage: Dict[int, float], commodities:List[str], gamma: float):
        self.digraph = digraph
        self.nodes = list(digraph.nodes)
        self.arcs = list(digraph.edges)
        self.incoming_arcs = {node: [arc for arc in self.arcs if arc[1] == node] for node in self.nodes}
        self.outgoing_arcs = {node: [arc for arc in self.arcs if arc[0] == node] for node in self.nodes}
        self.stages = stages
        self.stage_ids = [stage.stage_id for stage in self.stages]  # Starts at 1!
        self.stage_ids_star = self.stage_ids[1:]  # Without 1!
        self.first_stage_ids = [stage.stage_id for stage in self.stages if stage.name == "long term"]
        self.second_stage_ids = [stage.stage_id for stage in self.stages if stage.name == "day ahead"]
        self.third_stage_ids = [stage.stage_id for stage in self.stages if stage.name == "intra day"]
        self.traders = traders
        self.loss_rate = loss_rate
        self.allowed_percentage = allowed_percentage
        self.commodities = commodities
        self.gamma = gamma

    def build_model(self):
        model = gp.Model("Stochastic Secondary Energy Market")

        # Decision variables
        x_plus = model.addVars(self.nodes, self.stage_ids, self.traders, self.commodities, name="x_plus", lb=0.0)
        x_minus = model.addVars(self.nodes, self.stage_ids, self.traders, self.commodities, name="x_minus", lb=0.0)
        y_plus = model.addVars(self.nodes, self.stage_ids, self.traders, self.commodities, name="y_plus", lb=0.0)
        y_minus = model.addVars(self.nodes, self.stage_ids, self.traders, self.commodities, name="y_minus", lb=0.0)
        s_plus = model.addVars(self.nodes, self.stage_ids, self.commodities, name="s_plus", lb=0.0)
        s_minus = model.addVars(self.nodes, self.stage_ids, self.commodities, name="s_minus", lb=0.0)
        f = model.addVars(self.traders, self.arcs, self.third_stage_ids, self.commodities, name="f", lb=0.0)
        q_sales = model.addVars(self.traders, self.nodes, self.third_stage_ids, self.commodities, name="q_sales", lb=0.0)
        q_production = model.addVars(self.traders, self.nodes, self.third_stage_ids, self.commodities, name="q_production", lb=0.0)
        v = model.addVars(self.traders, self.nodes, self.third_stage_ids, self.commodities, name="v", lb=0.0)
        w_plus = model.addVars(self.traders, self.nodes, self.third_stage_ids, self.commodities, name="w_plus", lb=0.0)
        w_minus = model.addVars(self.traders, self.nodes, self.third_stage_ids, self.commodities, name="w_minus", lb=0.0)

        # Objective function
        first_and_second_stage = gp.quicksum(self.stages[0].probability * gp.quicksum(self.stages[0].entry_costs[(n,k)] * x_plus[n, 1, t, k] + self.stages[0].exit_costs[(n,k)] * x_minus[n, 1, t, k] for n in self.nodes for t in self.traders for k in self.commodities) for m in self.first_stage_ids + self.second_stage_ids)
        third_stage = gp.quicksum(self.stages[m-1].probability * gp.quicksum(self.stages[m-1].entry_costs[(n,k)] * x_plus[n, m, t, k] + self.stages[m-1].exit_costs[(n,k)] * x_minus[n, m, t, k] +
                                                         self.stages[m-1].production_costs[(n,t,k)] * q_production[t, n, m, k] + self.stages[m-1].storage_costs[(t,n,k)] * v[t, n, m, k] +
                                                         gp.quicksum(self.stages[m-1].arc_costs[(a,k)] * f[t, a[0], a[1], m, k] for a in self.incoming_arcs[n]) for t in self.traders)
                          for m in self.third_stage_ids for n in self.nodes for k in self.commodities)

        model.setObjective(first_and_second_stage + third_stage, gp.GRB.MINIMIZE)

        # TSO constraints
        # Equation 1b
        for m in self.third_stage_ids:
            for a in self.arcs:
                model.addConstr(gp.quicksum(f[t, a[0], a[1], m, k] for t in self.traders for k in self.commodities) <= self.stages[m-1].arc_capacities[a],
                                name=f"eq1b[{m},{a}]")

        # # Equation 1c
        # for n in self.nodes:
        #     for m in self.stage_ids_star:
        #         for k in self.commodities:
        #             lhs = gp.quicksum((1 - self.loss_rate) * f[t, a[0], a[1], m, k] + q_production[t, n, m, k] + w_minus[t, n, m, k] for a in self.incoming_arcs[n] for t in self.traders) - gp.quicksum(f[t, a[0], a[1], m, k] for a in self.outgoing_arcs[n] for t in self.traders)
        #             rhs = gp.quicksum(x_minus[n, m_tilde, t, k] - y_minus[n, m_tilde, t, k] for t in self.traders for m_tilde in self.stages[m-1].ids_all_parents + [m])
        #             model.addConstr(lhs == rhs, name=f"eq1c[{n},{m},{k}]")
        #
        # # Equation 1d
        # for n in self.nodes:
        #     for m in self.stage_ids_star:
        #         for k in self.commodities:
        #             lhs = gp.quicksum(f[t, a[0], a[1], m, k] + q_sales[t, n, m, k] + w_plus[t, n, m, k] for a in self.outgoing_arcs[n] for t in self.traders) - gp.quicksum((1 - self.loss_rate) * f[t, a[0], a[1], m, k] for a in self.incoming_arcs[n] for t in self.traders)
        #             rhs = gp.quicksum(x_plus[n, m_tilde, t, k] - y_plus[n, m_tilde, t, k] for t in self.traders for m_tilde in self.stages[m-1].ids_all_parents + [m])
        #             model.addConstr(lhs == rhs, name=f"eq1d[{n},{m},{k}]")

        # Equation 1c
        for n in self.nodes:
            for m in self.stage_ids:
                for k in self.commodities:
                    model.addConstr(s_minus[n, m, k] <= self.stages[m-1].node_capacities[(n, k)] - gp.quicksum(x_minus[n, m_tilde, t, k] for t in self.traders for m_tilde in self.stages[m-1].ids_all_parents),
                                    name=f"eq1c[{n},{m},{k}]")

        # Equation 1d
        for n in self.nodes:
            for m in self.stage_ids:
                for k in self.commodities:
                    model.addConstr(s_plus[n, m, k] <= self.stages[m-1].node_capacities[(n, k)] - gp.quicksum(x_plus[n, m_tilde, t, k] for t in self.traders for m_tilde in self.stages[m-1].ids_all_parents),
                                    name=f"eq1d[{n},{m},{k}]")

        # Equation 1e
        for n in self.nodes:
            for k in self.commodities:
                lhs = s_minus[n, 1, k]
                rhs = self.allowed_percentage[n] * self.stages[0].node_capacities[(n,k)]
                model.addConstr(lhs <= rhs, name=f"eq1e[{n},{k}]")

        # Equation 1f
        for n in self.nodes:
            for k in self.commodities:
                lhs = s_plus[n, m, k]
                rhs = self.allowed_percentage[n] * self.stages[0].node_capacities[(n,k)]
                model.addConstr(lhs <= rhs, name=f"eq1f[{n},{k}]")

        # Equation 1g
        for n in self.nodes:
            for m in self.third_stage_ids:
                for k in self.commodities:
                    model.addConstr(gp.quicksum(q_sales[t, n, m, k] for t in self.traders) == self.stages[m-1].node_demands[(n,k)],
                                    name=f"eq1g[{n},{m},{k}]")

        # Equation 1h
        for m in self.third_stage_ids:
            for a in self.arcs:
                lhs = gp.quicksum(f[t, a[0], a[1], m, "hydrogen"] for t in self.traders)
                rhs = gp.quicksum(f[t, a[0], a[1], m, "gas"] for t in self.traders) * self.gamma
                model.addConstr(lhs <= rhs, name=f"eq1j[{m},{a}]")

        # Supplier constraints
        # Equation 1i
        for n in self.nodes:
            for m in self.third_stage_ids:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(q_production[t, n, m, k] <= self.stages[m-1].production_capacities[(t, n, k)], name=f"eq1i[{n},{m},{t},{k}]")

        # Equation 1j
        for n in self.nodes:
            for m in self.third_stage_ids:
                for t in self.traders:
                    for k in self.commodities:
                        lhs = q_production[t, n, m, k] + w_minus[t, n, m, k] + gp.quicksum((1 - self.loss_rate) * f[t, a[0], a[1], m, k] for a in self.incoming_arcs[n])
                        rhs = q_sales[t, n, m, k] + w_plus[t, n, m, k] + gp.quicksum(f[t, a[0], a[1], m, k] for a in self.outgoing_arcs[n])
                        model.addConstr(lhs == rhs, name=f"eq1j[{n},{m},{t},{k}]")

        # Equation 1k
        k1 = "hydrogen"
        k2 = "gas"
        for n in self.nodes:
            for m in self.third_stage_ids:
                # Wat komt er in?
                # Hydrogen
                rhs_numerator = gp.quicksum(
                    (1 - self.loss_rate) * f[t, a2[0], a2[1], m, k1] for a2 in self.incoming_arcs[n] for t in
                    self.traders)
                rhs_numerator += gp.quicksum(q_production[t, n, m, k1] for t in self.traders)
                rhs_numerator += gp.quicksum(w_minus[t, n, m, k1] for t in self.traders)

                # Gas
                rhs_denominator = gp.quicksum(
                    (1 - self.loss_rate) * f[t, a2[0], a2[1], m, k2] for a2 in self.incoming_arcs[n] for t in
                    self.traders)
                rhs_denominator += gp.quicksum(q_production[t, n, m, k2] for t in self.traders)
                rhs_denominator += gp.quicksum(w_minus[t, n, m, k2] for t in self.traders)

                for a1 in self.outgoing_arcs[n]:
                    # Wat gaat er uit?
                    # Hydrogen
                    lhs_numerator = gp.quicksum(f[t, a1[0], a1[1], m, k1] for t in self.traders)
                    lhs_numerator += gp.quicksum(q_sales[t, n, m, k1] for t in self.traders)
                    lhs_numerator += gp.quicksum(w_plus[t, n, m, k1] for t in self.traders)

                    # Gas
                    lhs_denominator = gp.quicksum(f[t, a1[0], a1[1], m, k2] for t in self.traders)
                    lhs_denominator += gp.quicksum(q_sales[t, n, m, k2] for t in self.traders)
                    lhs_denominator += gp.quicksum(w_plus[t, n, m, k2] for t in self.traders)

                    model.addConstr(lhs_numerator * rhs_denominator == lhs_denominator * rhs_numerator, name=f"eq1k[{n},{m},{a1}]")

                # lhs_numerator = gp.quicksum(
                #     f[t, a1[0], a1[1], m, k1] + q_sales[t, n, m, k1] + w_plus[t, n, m, k1] for t in self.traders)
                # lhs_denominator = gp.quicksum(
                #     f[t, a1[0], a1[1], m, k2] + q_sales[t, n, m, k2] + w_plus[t, n, m, k2] for t in self.traders)
                #
                # # Hydrogen
                # rhs_numerator = gp.quicksum(
                #     (1 - self.loss_rate) * f[t, a2[0], a2[1], m, k1] for a2 in self.incoming_arcs[n] for t in
                #     self.traders)
                # rhs_numerator += gp.quicksum(q_production[t, n, m, k1] for t in self.traders)
                # rhs_numerator += gp.quicksum(w_minus[t, n, m, k1] for t in self.traders)
                #
                # # Gas
                # rhs_denominator = gp.quicksum(
                #     (1 - self.loss_rate) * f[t, a2[0], a2[1], m, k2] for a2 in self.incoming_arcs[n] for t in
                #     self.traders)
                # rhs_denominator += gp.quicksum(q_production[t, n, m, k2] for t in self.traders)
                # rhs_denominator += gp.quicksum(w_minus[t, n, m, k2] for t in self.traders)

        # Equation 1l
        for n in self.nodes:
            for m in self.third_stage_ids:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_minus[n, m_tilde, t, k] - y_minus[n, m_tilde, t, k] for m_tilde in self.stages[m-1].ids_all_parents + [m]) >= q_sales[t, n, m, k], name=f"eq1l[{n},{m},{t},{k}]")

        # Equation 1m
        for n in self.nodes:
            for m in self.third_stage_ids:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_plus[n, m_tilde, t, k] - y_plus[n, m_tilde, t, k] for m_tilde in self.stages[m-1].ids_all_parents + [m]) >= q_production[t, n, m, k], name=f"eq1m[{n},{m},{t},{k}]")

        # Equation 1n
        for n in self.nodes:
            for m in self.third_stage_ids:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t, n, m, k] == gp.quicksum(w_plus[t, n, m_tilde, k] - w_minus[t, n, m_tilde, k] for m_tilde in self.stages[m-1].ids_all_parents + [m] if self.stages[m_tilde-1].name == "intra day"), name=f"eq1n[{n},{m},{t},{k}]")

        # Equation 1o
        for n in self.nodes:
            for m in self.third_stage_ids:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t, n, m, k] <= self.stages[m-1].storage_capacities[(t, n, k)], name=f"eq1o[{n},{m},{t},{k}]")

        # Market constraints
        # Equation 1p
        for n in self.nodes:
            for m in self.stage_ids:
                for k in self.commodities:
                    lhs = gp.quicksum(x_minus[n, m, t, k] for t in self.traders)
                    rhs = s_minus[n, m, k] + gp.quicksum(y_minus[n, m, t, k] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1p[{n},{m},{k}]")

        # Equation 1q
        for n in self.nodes:
            for m in self.stage_ids:
                for k in self.commodities:
                    lhs = gp.quicksum(x_plus[n, m, t, k] for t in self.traders)
                    rhs = s_plus[n, m, k] + gp.quicksum(y_plus[n, m, t, k] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1q[{n},{m},{k}]")

        return model



