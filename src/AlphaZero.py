import numpy as np
import random
import torch
from MCTS_AlphaZero import MCTS_AlphaZero
from tqdm import tqdm
from Backgammon import Backgammon
from Model import ResNet
import torch.nn.functional as F

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS_AlphaZero(game, model, args)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            dice_roll = self.game.roll_dice()
            neutral_state = self.game.change_perspective(state, player)
            _, action_probs = self.mcts.search(neutral_state, dice_roll)

            probs = np.zeros(self.game.action_size, dtype=np.float32)
            for move, prob in action_probs.items():
                index = self.game.encode_move(move)  
                probs[index] = prob
            memory.append((neutral_state, dice_roll, probs, player))
            
            temperature_action_probs = probs ** (1/self.args['temperature'])
            # Normalize to make sure it sums to 1
            temperature_action_probs /= temperature_action_probs.sum()
            action = np.random.choice(self.game.action_size, p = temperature_action_probs)

            action = self.game.decode_move(action, dice_roll)
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_dice_roll, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state, hist_dice_roll),
                        np.array(hist_action_probs, dtype=np.float32),
                        hist_outcome
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player)
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device = self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device = self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device = self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step() 
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for _ in tqdm(range(self.args['num_selfPlay_iterations'])):
                memory += self.selfPlay()
                
            self.model.train()
            for _ in tqdm(range(self.args['num_epochs'])):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"../trained_models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"../trained_models/optimizer_{iteration}.pt")
            
            
game = Backgammon()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ResNet(game, 4, 64, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)

args = {
    'C': 1.5,
    'num_searches': 60,
    'num_iterations': 1,
    'num_selfPlay_iterations': 3,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25
}

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()
            
