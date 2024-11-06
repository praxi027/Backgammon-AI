import cProfile
import pstats
from io import StringIO
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
            best_action, action_probs = self.mcts.search(neutral_state, dice_roll)
            
            probs = np.zeros(self.game.action_size, dtype=np.float32)
            for move, prob in action_probs.items():
                index = self.game.encode_move(move)
                probs[index] = prob
            memory.append((neutral_state, dice_roll, probs, player))
            
            action = np.random.choice(self.game.action_size, p=probs)
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
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state = torch.tensor(np.array(state), dtype=torch.float32)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32)
            
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
            
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

def profile_learn():
    game = Backgammon()
    model = ResNet(game, 4, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_selfPlay_iterations': 1,
        'num_epochs': 4,
        'batch_size': 64
    }
    alphaZero = AlphaZero(model, optimizer, game, args)

    # Profile the learn function with detailed stats
    with cProfile.Profile() as profiler:
        alphaZero.learn()

    # Save and display profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)  # Top 20 functions by cumulative time
    stats.print_callers(10)  # Show callers for top 10 time-consuming calls
    stats.dump_stats("learn_profile.prof")  # Save data to file for later analysis

def profile_alpha_zero():
    game = Backgammon()
    model = ResNet(game, 4, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_selfPlay_iterations': 1,
        'num_epochs': 4,
        'batch_size': 64
    }
    alphaZero = AlphaZero(model, optimizer, game, args)

    profiler = cProfile.Profile()
    profiler.enable()
    alphaZero.learn()
    profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.TIME)

    total_time = sum([stat[2] for stat in stats.stats.values()])  # stat[2] is `tt`

    func_percentages = [
        (func, stat[2], (stat[2] / total_time) * 100 if total_time > 0 else 0)
        for func, stat in stats.stats.items()
    ]

    top_functions_by_percentage = sorted(func_percentages, key=lambda x: x[2], reverse=True)[:20]

    print(f"{'Function':<60} {'Total Time':<15} {'Percentage of Total Time'}")
    print("=" * 90)

    for func, tottime, percentage in top_functions_by_percentage:
        func_name = f"{func[0]}:{func[1]}({func[2]})"
        print(f"{func_name:<60} {tottime:<15.6f} {percentage:.2f}%")

    stats.print_stats(20)


if __name__ == "__main__":
    print("Profiling AlphaZero's learn method with percentage time breakdown (Top 20 by Percentage)...")
    profile_alpha_zero()
