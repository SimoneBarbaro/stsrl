import random
import math
import stsrl.slaythespire as sts
import logging

logger = logging.getLogger(__name__)


class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = 0 # float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, is_chance=False):
        self.edges = []
        self.simulation_count = 0
        self.evaluation_sum = 0.0
        self.is_chance = is_chance
    def __str__(self):
        edge_str = '\n-- '.join([str(e).replace("-- ", "---- ") for e in self.edges])
        node_str = "{"+ f"visits: {self.simulation_count}, avg_evaluation: {self.evaluation_sum / self.simulation_count if self.simulation_count > 0 else 0}" + "}"
        if len(self.edges) > 0:
            node_str += "children: \n-- " + edge_str
        return node_str


class Edge:
    def __init__(self, action, node, description=None):
        self.action = action
        self.node = node
        self.description = description
    def __str__(self):
        return f"{self.description}->{str(self.node)}" if self.description else "->"+str(self.node)


class BattleScrumEvaluator:
    def evaluate_end_state(bc):
        potion_score = len(bc.potions) * 4

        if bc.outcome == sts.BattleOutcome.PLAYER_VICTORY:
            return 10 * (35 + bc.player.hp / bc.player.max_hp + potion_score - (bc.turn * 0.01))
        else:
            # could_have_spikers = bc.encounter in [sts.MonsterEncounter.THREE_SHAPES, sts.MonsterEncounter.FOUR_SHAPES]
            energy_penalty = bc.player.energy * -0.2
            draw_bonus = bc.cards_drawn * 0.03
            alive_score = len(bc.monsters) * -1
            return (1 - BattleScrumEvaluator.get_non_minion_monster_cur_hp_ratio(
                bc)) * 10 + alive_score + energy_penalty + draw_bonus + potion_score / 2 + (bc.turn * 0.2)

    def get_non_minion_monster_cur_hp_ratio(bc):
        cur_hp_total = 0
        max_hp_total = 0

        for monster in bc.monsters:
            if not monster.is_minion:
                cur_hp_total += monster.hp
                max_hp_total += monster.max_hp

        if cur_hp_total == 0 or max_hp_total == 0:
            return 0

        return cur_hp_total / max_hp_total


class BattleScumSearcher2:
    def __init__(self, bc, eval_fnc=BattleScrumEvaluator.evaluate_end_state, seed=0, chance_sampling_breath=4):
        self.root_state = sts.BattleContext(bc)
        self.eval_fnc = eval_fnc
        self.rand_gen = random.Random(seed)
        self.root = Node()
        self.search_stack = []
        self.action_stack = []
        self.best_action_sequence = []
        self.min_max_stats = MinMaxStats()
        self.outcome_player_hp = 0
        self.exploration_parameter = math.sqrt(2.0)
        self.chance_sampling_breath = chance_sampling_breath
        # logger.info("Initialized BattleScumSearcher2")

    def reset(self):
        self.search_stack = []
        self.action_stack = []
        self.best_action_sequence = []
        self.min_max_stats = MinMaxStats()
        self.outcome_player_hp = 0

    def update_root(self, action):
        for edge in self.root.edges:
            if edge.action == action:
                self.root = edge.node
        logger.info(f"Attempting root update: {action.print_desc(self.root_state)}")
        action.execute(self.root_state)
        self.reset()

    def search(self, simulations):
        if self.is_terminal_state(self.root_state):
            evaluation = self.eval_fnc(self.root_state)
            self.outcome_player_hp = self.root_state.player.hp
            self.best_action_sequence = []
            self.root.evaluation_sum = evaluation
            self.root.simulation_count = 1
            # logger.info("Terminal state reached at root with evaluation: %s", evaluation)
            return

        # logger.info("Starting search with %d simulations", simulations)
        for _ in range(simulations):
            self.step()
        # logger.info("Search completed")

    def get_best_action(self):
        if len(self.best_action_sequence) > 0:
            return self.best_action_sequence[0]
        logger.warning("No best action sequence found, check if this is a bug")
        best_edge = None
        best_edge_value = -float('inf')
        for i in range(len(self.root.edges)):
            edge_eval = self.evaluate_edge(self.root, i)
            if edge_eval > best_edge_value:
                best_edge = self.root.edges[i]
        return best_edge.action if best_edge is not None else None

    def step(self):
        self.search_stack = [self.root]
        self.action_stack.clear()
        cur_state = sts.BattleContext(self.root_state)
        # logger.debug("Starting new search step")

        while True:
            cur_node = self.search_stack[-1]

            if self.is_terminal_state(cur_state):
                # logger.debug("Terminal state reached during step, updating search tree")
                self.update_from_playout(self.search_stack, self.action_stack, cur_state)
                return

            if len(cur_node.edges) == 0:
                self._expand(cur_node, cur_state)
                select_idx = self.select_first_action_for_leaf_node(cur_node)
                self.select_edge(cur_state, cur_node, select_idx)
                self.playout_random(cur_state, self.action_stack)
                # logger.debug("Expanded and played out leaf node, updating search tree")
                self.update_from_playout(self.search_stack, self.action_stack, cur_state)

                return
            else:
                if cur_node.is_chance:
                    select_idx = random.choice([i for i in range(len(cur_node.edges))])
                else:
                    select_idx = self.select_best_edge_to_search(cur_node)
                self.select_edge(cur_state, cur_node, select_idx)
                # logger.debug("Selected edge %d for further search", select_idx)

    def select_edge(self, cur_state, cur_node: Node, select_idx: int):
        edge_taken = cur_node.edges[select_idx]
        # For chance nodes action = randomize battle + deck,
        # then execute action that was left in stack from previous node
        if cur_node.is_chance:
            cur_state.randomize_rng_counters(edge_taken.action)
            # logger.debug("Randomized RNG counters for chance node")
            cur_state.execute_actions()
        else:
            edge_taken.action.submit(cur_state)
            # If node is chance, leave action in stack instead of executing so we randomize it later
            # logger.debug("Adding action: %s to search stack", edge_taken.action)
            if not edge_taken.node.is_chance:
                cur_state.execute_actions()
        self.action_stack.append(edge_taken.action)
        self.search_stack.append(edge_taken.node)

    def _expand(self, node, cur_state):
        if not self.is_terminal_state(cur_state):
            if node.is_chance:
                for _ in range(self.chance_sampling_breath):
                    node.edges.append(Edge(
                        self.rand_gen.randint(1, 100), Node(is_chance=False), description="SampleRngCounters"))
                # logger.debug("Expanded chance node with %d edges", self.chance_sampling_breath)
            else:
                for action in cur_state.get_available_actions():
                    next_state = sts.BattleContext(cur_state)
                    action.execute(next_state)
                    is_chance = not cur_state.is_same_rng_counters(next_state)
                    node.edges.append(Edge(action, Node(is_chance=is_chance), description=action.print_desc(cur_state)))
                # logger.debug("Expanded node with %d edges", len(node.edges))

    def update_from_playout(self, stack, action_stack, end_state):
        # logger.debug("Evaluating action sequence: %s", action_stack)
        evaluation = self.eval_fnc(end_state)
        if evaluation > self.min_max_stats.maximum:
            self.best_action_sequence = action_stack[:]
            self.outcome_player_hp = end_state.player.hp
        self.min_max_stats.update(evaluation)
        for node in reversed(stack):
            node.simulation_count += 1
            node.evaluation_sum += evaluation
        # logger.debug("Updated nodes from playout with evaluation: %s", evaluation)

    def is_terminal_state(self, bc):
        return bc.outcome != sts.BattleOutcome.UNDECIDED

    def evaluate_edge(self, parent, edge_idx):
        edge = parent.edges[edge_idx]
        quality_value = 0
        if self.best_action_sequence:
            avg_evaluation = edge.node.evaluation_sum / (edge.node.simulation_count + 1)
            quality_value = self.min_max_stats.normalize(avg_evaluation)
        exploration_value = self.exploration_parameter * math.sqrt(
            math.log(parent.simulation_count + 1) / (edge.node.simulation_count + 1))
        return quality_value + exploration_value

    def select_best_edge_to_search(self, cur):
        if len(cur.edges) == 1:
            return 0

        best_edge = 0
        best_edge_value = self.evaluate_edge(cur, best_edge)

        for i in range(1, len(cur.edges)):
            value = self.evaluate_edge(cur, i)
            if value > best_edge_value:
                best_edge = i
                best_edge_value = value

        return best_edge

    def select_first_action_for_leaf_node(self, leaf_node):
        return self.rand_gen.randint(0, len(leaf_node.edges) - 1)

    def playout_random(self, state, action_stack):
        #logger.debug(f"Init playout at state:{state}")
        while not self.is_terminal_state(state):
            actions = state.get_available_actions()
            if len(actions) == 0:
                raise RuntimeError("No available actions")
            selected_idx = self.rand_gen.randint(0, len(actions) - 1)
            action = actions[selected_idx]
            action_stack.append(action)
            #logger.debug(f"submit action to playout:{action.print_desc(state)}")
            action.execute(state)
