import numpy as np
import random
import torch
from MCTS_AlphaZero import MCTS_AlphaZero
from tqdm import tqdm
from Backgammon import Backgammon
from Model import ResNet
import torch.nn.functional as F
import ray  
import os

@ray.remote
class AlphaZeroWorker:
    def __init__(self, model_state_dict, args):
        self.game = Backgammon()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet(self.game, 4, 64, self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()  # Set model to evaluation mode
        self.args = args
        self.mcts = MCTS_AlphaZero(self.game, self.model, self.args)

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
            temperature_action_probs /= temperature_action_probs.sum()
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

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

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.kl_div(F.log_softmax(out_policy, dim=1), policy_targets, reduction='batchmean')
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step() 

    def learn(self):
        ray.init(ignore_reinit_error=True) 
        for iteration in range(self.args['num_iterations']):
            memory = []

            # Get the model state dict to pass to workers
            model_state_dict = self.model.state_dict()
            num_workers = self.args.get('num_workers', os.cpu_count())
            num_selfPlay_per_worker = self.args['num_selfPlay_iterations'] // num_workers

            workers = [AlphaZeroWorker.remote(model_state_dict, self.args) for _ in range(num_workers)]

            futures = []
            for worker in workers:
                for _ in range(num_selfPlay_per_worker):
                    futures.append(worker.selfPlay.remote())

            with tqdm(total=len(futures), desc="Self-Play Progress", leave=True) as pbar:
                while futures:
                    done, futures = ray.wait(futures, timeout=1)
                    for result in done:
                        memory += ray.get(result)
                        pbar.update(1)

            self.model.train()
            for _ in tqdm(range(self.args['num_epochs']), desc="Training Progress", leave=True):
                self.train(memory)

            save_dir = "../trained_models/"
            os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), f"{save_dir}model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"{save_dir}optimizer_{iteration}.pt")


        ray.shutdown()  

game = Backgammon()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = ResNet(game, 4, 64, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
args = {
    'C': 1.5,
    'num_searches': 60,
    'num_iterations': 1,
    'num_selfPlay_iterations': 8,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'num_workers': 4  
}

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()
