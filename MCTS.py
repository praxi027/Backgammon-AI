import numpy as np
import math
from Backgammon import Backgammon
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
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
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
        child = Node(self.game, self.args, self.state, is_chance_node=True)
        if self.is_chance_node:
            dice_roll = random.choice(self.expandable_moves)
            self.expandable_moves.remove(dice_roll)
            
            child = Node(self.game, self.args, self.state, dice_rolls=dice_roll, parent=self, action_taken=self.action_taken, is_chance_node=False)
        else:
            action = random.choice(self.expandable_moves)
            self.expandable_moves.remove(action)
            
            child_state = self.state.copy()
            child_state = self.game.get_next_state(self.state, action, 1)
            child_state = self.game.change_perspective(child_state, -1)
            child = Node(self.game, self.args, child_state, dice_rolls=None, parent=self, action_taken=action, is_chance_node=True)

        self.children.append(child)
        return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            dice_roll = self.game.roll_dice()
            valid_moves = self.game.get_valid_moves(rollout_state, rollout_player, dice_roll)
            action = random.choice(valid_moves)
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, rollout_player)

            value, is_terminal = self.game.get_value_and_terminated(rollout_state, rollout_player)
            if is_terminal:
                return value

            rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
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
            while node.is_fully_expanded() and not node.is_chance_node:
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            # Expansion
            if not is_terminal:
                node = node.expand()
                # Simulation
                value = node.simulate()

            # Backpropagation
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
    
    
# ------------------------------------------------------------------------------------------------------    
args = {
    'C': 1.41,
    'num_searches': 1000
}

backgammon = Backgammon()
mcts = MCTS(backgammon, args)

state = backgammon.get_initial_state()
player = 1  # 1 for human, -1 for AI

while True:
    print(state)
    
    # Roll the dice
    dice_rolls = backgammon.roll_dice()
    print(f"Player {player} rolled: {dice_rolls}")

    # Get valid moves based on the dice roll
    valid_moves = backgammon.get_valid_moves(state, player, dice_rolls)
    if not valid_moves:
        print(f"Player {player} has no valid moves.")
        player = -player
        continue

    if player == 1:
        print("Valid moves:")
        for idx, move in enumerate(valid_moves):
            print(f"{idx}: {move}")
        move_index = int(input(f"Player {player}, select a move: "))
        if 0 <= move_index < len(valid_moves):
            action = valid_moves[move_index]
        else:
            print("Invalid move selection. Try again.")
            continue
    else:
        ai_state = state
        ai_state = backgammon.change_perspective(state, player)
        action ,mcts_probs = mcts.search(ai_state, dice_rolls)
        print(f"AI selects move: {action}")

    # Apply the move to the state
    state = backgammon.get_next_state(state, action, player)
    
    # Check if the game has ended
    value, is_terminal = backgammon.get_value_and_terminated(state, player)
    if is_terminal:
        print(state)
        print(f"Player {player} wins.")
        break

    # Switch players
    player = -player