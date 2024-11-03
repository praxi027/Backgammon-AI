import numpy as np
import math
import random
import torch 
from Backgammon import Backgammon

class Node:
    def __init__(self, game, args, state, dice_rolls=None, parent=None, action_taken=None, is_chance_node=False, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.dice_rolls = dice_rolls
        self.parent = parent
        self.action_taken = action_taken
        self.is_chance_node = is_chance_node
        self.prior = prior
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        
        if self.is_chance_node:
            for dice_roll in self.game.get_all_possible_dice_rolls():
            # Create a child node for each possible dice roll
                child = Node(
                    self.game,
                    self.args,
                    self.state,
                    dice_rolls=dice_roll,
                    parent=self,
                    action_taken=self.action_taken,  # No specific action; this is a chance node
                    is_chance_node=False,  # Next nodes will be decision/action nodes
                    prior = self.prior
                )
                self.children.append(child)
            
    def is_fully_expanded(self):
        return len(self.children) > 0
    
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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:        
                child_state = self.state.copy()
                decoded_action = self.game.decode_move(action, self.dice_rolls)
                child_state = self.game.get_next_state(self.state, decoded_action, 1)
                child_state = self.game.change_perspective(child_state, -1)
                child = Node(self.game, self.args, child_state, dice_rolls=None, parent=self, action_taken=decoded_action, is_chance_node=True, prior = prob)
                self.children.append(child)
        return child
            
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


class MCTS_AlphaZero:
    def __init__(self, game, model, args):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state, dice_rolls):
        root = Node(self.game, self.args, state, dice_rolls=dice_rolls, is_chance_node=False)
            
        for _ in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state)
            value = self.game.get_opponent_value(value)
            if not is_terminal:
                policy, value = self.model(
                torch.tensor(self.game.get_encoded_state(node.state, node.dice_rolls)).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                policy *= self.game.get_valid_moves_encoded(node.state, node.dice_rolls)
                policy /= np.sum(policy)
                value = value.item()
                
                node = node.expand(policy)
            
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
