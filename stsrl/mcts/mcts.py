import math
import random
import numpy as np

import stsrl.slaythespire as sts


class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, is_chance=False, is_terminal=False):
        self.edges = []
        self.simulation_count = 0
        self.evaluation_sum = 0.0
        self.is_chance = is_chance
        self.is_terminal = is_terminal

    def __str__(self):
        edge_str = '\n-- '.join([str(e).replace("-- ", "---- ") for e in self.edges])
        node_str = "{" + f"visits: {self.simulation_count}, avg_evaluation: {self.evaluation_sum / self.simulation_count if self.simulation_count > 0 else 0}" + "}"
        if len(self.edges) > 0:
            node_str += "children: \n-- " + edge_str
        return node_str

    def is_expanded(self):
        return len(self.children) > 0

    def avg_value(self, minimax: MinMaxStats) -> float:
        return 0 if self.visits == 0 else minimax.normalize(self.value) / self.visits

    def add_edge(self, edge):
        self.edges.append(edge)


class Edge:
    def __init__(self, action, node: Node, description=None):
        self.action = action
        self.node: Node = node
        self.description = description

    def __str__(self):
        return f"{self.description}->{str(self.node)}" if self.description else "->" + str(self.node)


class Environment:
    pass


class NodeEvaluator:
    def evaluate_node_state(self, node_state):
        pass


class RolloutPolicy:
    def rollout(self, state):
        pass


class ExplorationPolicy:
    def exploration_bonus(self, node: Node, edge: Edge):
        pass


class UpperConfidenceBoundPolicy(ExplorationPolicy):
    def __init__(self, exploration_parameter=math.sqrt(2.0)):
        self.exploration_parameter = exploration_parameter

    def exploration_bonus(self, node: Node, edge: Edge):
        return self.exploration_parameter * np.sqrt(
            np.log(node.simulation_count + 1) / (1 + edge.node.simulation_count))


class MCTS:
    def __init__(self,
                 nodeEvaluator: NodeEvaluator,
                 rolloutPolicy: RolloutPolicy,
                 explorationPolicy: ExplorationPolicy,
                 replayBuffer):
        self.root = Node(is_chance=False, is_terminal=False)
        self.state = None
        self.nodeEvaluator = nodeEvaluator
        self.rolloutPolicy = rolloutPolicy
        self.explorationPolicy = explorationPolicy
        self.reset()

    def reset(self):
        self.min_max_stats = MinMaxStats()
        self.search_path = []
        self.action_stack = []
        self.best_action_sequence = []

    def set_root(self, root_state: sts.BattleContext):
        self.root = Node(root_state)
        self.state = root_state
        self._expand(self.root)

    def search(self):
        self._simulate_multiple(self.num_simulations)

    def _simulate(self):
        node = self.root
        state = self.state.copy()
        while node.is_expanded() and not node.is_terminal():
            action = self._select(node)
            node = node.children[action]

            self.search_path.append(node)
        self._expand(node)
        for _ in range(self.num_rollouts):
            reward = self._rollout(state)
            self._backpropagate(node, reward)

    def _simulate_multiple(self, num_simulations):
        for _ in range(num_simulations):
            self._simulate()

    def _select(self, node: Node):
        if node.is_terminal:
            return
        if not node.is_expanded():
            self._expand(node)

        if len(node.edges) == 1:
            return 0
        if node.simulation_count == 0:
            return random.randrange(0, len(node.edges))

        best_edge = 0
        best_edge_value = self._evaluate_edge(node, best_edge)

        for i in range(0, len(node.edges)):
            value = self._evaluate_edge(node, i)
            if value > best_edge_value:
                best_edge = i
                best_edge_value = value

        return best_edge

    def _evaluate_edge(self, parent: Node, edge_idx):
        edge = parent.edges[edge_idx]
        quality_value = parent.avg_value(self.min_max_stats)
        if parent.is_chance:
            quality_value = 0
        exploration_value = self.explorationPolicy.exploration_bonus(parent, edge)
        return quality_value + exploration_value

    def _expand(self, node, cur_state):
        if not self.is_terminal_state(cur_state):
            if node.is_chance:
                for _ in range(self.chance_sampling_breath):
                    node.edges.append(Edge(
                        self.rand_gen.randint(1, 100), Node(is_chance=False, is_terminal=False),
                        description="SampleRngCounters"))
                # logger.debug("Expanded chance node with %d edges", self.chance_sampling_breath)
            else:
                for action in cur_state.get_available_actions():
                    next_state = sts.BattleContext(cur_state)
                    action.execute(next_state)
                    is_chance = not cur_state.is_same_rng_counters(next_state)
                    node.edges.append(Edge(action, Node(is_chance=is_chance, is_terminal=next_state.is_terminal()),
                                           description=action.print_desc(cur_state)))
                # logger.debug("Expanded node with %d edges", len(node.edges))

    def _rollout(self, state):
        final_state = self.rolloutPolicy.rollout(state)
        return self.nodeEvaluator.evaluate_node_state(final_state)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_action(self):
        # Get action with the highest value
        best_action = None
        best_value = -float('inf')
        for action, child in self.root.children.items():
            if child.avg_value() > best_value:
                best_value = child.value_model
                best_action = action
        return best_action
