import numpy as np
import math
import random

class Node:
    def __init__(self, game, args, state, dice_rolls=None, parent=None, action_taken=None, is_chance_node=False):
        self.game = game
        self.args = args
        self.state = state
        self.dice_rolls = dice_rolls
        self.parent = parent
        self.action_taken = action_taken
        self.is_chance_node = is_chance_node
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        
        if is_chance_node:
            self.expandable_moves = game.get_all_possible_dice_rolls()
        else:
            self.expandable_moves = game.get_valid_moves(state, 1, dice_rolls)
            
    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        if self.is_chance_node:
            # For chance nodes, select a random child
            return random.choice(self.children) 

        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child
    
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        if self.is_chance_node: # should never happen
            dice_roll = random.choice(self.expandable_moves)
            self.expandable_moves.remove(dice_roll)
            
            child = Node(self.game, self.args, self.state, dice_rolls=dice_roll, parent=self, action_taken=self.action_taken, is_chance_node=False)
        else:
            action = random.choice(self.expandable_moves)
            self.expandable_moves.remove(action)
            
            child_state = self.state.copy()
            child_state = self.game.get_next_state(child_state, action, 1)
            child_state = self.game.change_perspective(child_state, -1)
            child = Node(self.game, self.args, child_state, dice_rolls=None, parent=self, action_taken=action, is_chance_node=True)

        self.children.append(child)
        return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1
        dice_rolls = self.dice_rolls if not self.is_chance_node else self.game.roll_dice()
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state, rollout_player, dice_rolls)
            action = random.choice(valid_moves)
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state)

            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value

            rollout_player = self.game.get_opponent(rollout_player)
            dice_rolls = self.game.roll_dice()
            
    def backpropagate(self, value):
        if self.is_chance_node:
            self.value_sum += value
            self.visit_count += 1
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value) 
        else:
            self.value_sum += value
            self.visit_count += 1
            if self.parent is not None:
                self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state, dice_rolls):
        root = Node(self.game, self.args, state, dice_rolls=dice_rolls, is_chance_node=False)
            
        for _ in range(self.args['num_searches']):
            node = root

            # Selection
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state)
            value = self.game.get_opponent_value(value)
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
            
            node.backpropagate(value)

        # Choose the action with the highest visit count
        action_probs = {}
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        total_visits = sum(action_probs.values())
        if total_visits > 0:
            action_probs = {action: count / total_visits for action, count in action_probs.items()}
        best_action = max(action_probs, key=action_probs.get)
        return best_action, action_probs
