from Backgammon import Backgammon
from MCTS import MCTS
import random
from tqdm import tqdm
import time
from MCTS_AlphaZero import MCTS_AlphaZero
from Model import ResNet

def play_multiplayer():
    game = Backgammon()              
    state = game.get_initial_state()
    dice_rolls = game.roll_dice()
    player = 1

    while True:
        print("Current State:", state)
        dice_rolls = game.roll_dice()
        print("Dice Rolls:", dice_rolls)

        valid_moves = game.get_valid_moves(state, player, dice_rolls)
        print("Valid Moves:", valid_moves)

        action = int(input(f"Player {player}, select your action (0-{len(valid_moves) - 1}): "))
        
        if action < 0 or action >= len(valid_moves):
            print("Action not valid, please choose a valid action.")
            continue
            
        state = game.get_next_state(state, valid_moves[action], player)
        
        value, is_terminal = game.get_value_and_terminated(state)
        
        if is_terminal:
            print("Player ", player, " wins!")
            break
            
        player = -player  # Switch player
        
def play_vs_ai(args):
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
        value, is_terminal = backgammon.get_value_and_terminated(state)
        if is_terminal:
            print(state)
            print(f"Player {player} wins.")
            break

        # Switch players
        player = -player

def simulate_ai_vs_random(num_games, args):
    backgammon = Backgammon()
    mcts = MCTS(backgammon, args)
    
    results = {'AI Wins': 0, 'Random Wins': 0}

    for _ in tqdm(range(num_games), desc="Playing Games"):
        state = backgammon.get_initial_state()
        
        player = 1  if _ % 2 == 0 else -1

        while True:
            # Roll the dice
            dice_rolls = backgammon.roll_dice()

            # Get valid moves based on the dice roll
            valid_moves = backgammon.get_valid_moves(state, player, dice_rolls)

            if player == -1:
                # AI player
                state = backgammon.change_perspective(state, player)
                action ,mcts_probs = mcts.search(state, dice_rolls)
                state = backgammon.change_perspective(state, player)
            else:
                action = random.choice(valid_moves)

            # Apply the move to the state
            state = backgammon.get_next_state(state, action, player)

            # Check if the game has ended
            value, is_terminal = backgammon.get_value_and_terminated(state)
            if is_terminal:
                if player == -1:
                    results['AI Wins'] += 1
                    print()
                    print("AI Wins")
                else:
                    results['Random Wins'] += 1
                    print()
                    print("Random Wins")
                break

            # Switch players
            player = -player
    return results

def simulate_ai_vs_random_AlphaZero(num_games, args):
    backgammon = Backgammon()
    model = ResNet(backgammon, 4, 64)
    model.eval()
    mcts = MCTS_AlphaZero(backgammon, model, args)
    
    results = {'AI Wins': 0, 'Random Wins': 0}

    for _ in tqdm(range(num_games), desc="Playing Games"):
        state = backgammon.get_initial_state()
        
        player = 1  if _ % 2 == 0 else -1

        while True:
            # Roll the dice
            dice_rolls = backgammon.roll_dice()

            # Get valid moves based on the dice roll
            valid_moves = backgammon.get_valid_moves(state, player, dice_rolls)

            if player == -1:
                # AI player
                ai_state = state
                ai_state = backgammon.change_perspective(state, player)
                action ,mcts_probs = mcts.search(ai_state, dice_rolls)
            else:
                action = random.choice(valid_moves)

            # Apply the move to the state
            state = backgammon.get_next_state(state, action, player)

            # Check if the game has ended
            value, is_terminal = backgammon.get_value_and_terminated(state)
            if is_terminal:
                if player == -1:
                    results['AI Wins'] += 1
                    print()
                    print("AI Wins")
                else:
                    results['Random Wins'] += 1
                    print()
                    print("Random Wins")
                break

            # Switch players
            player = -player
    return results

def run_simulation(args):
    # Get the number of games from the user
    num_games = int(input("Enter the number of games to simulate: "))
    searches = int(input("Enter the number of searches for MCTS: "))
    args['num_searches'] = searches
    # Start timing the simulation
    start_time = time.time()
    results = simulate_ai_vs_random(num_games, args)  # Assuming simulate_ai_vs_random is defined elsewhere
    end_time = time.time()
    
    # Print results
    print(f"Training time for {num_games} games: {(end_time - start_time)/60:.2f} minutes")
    print(f"Results after {num_games} games:")
    print(f"AI Wins: {results['AI Wins']}")
    print(f"Random Wins: {results['Random Wins']}")
    
def run_simulation_AlphaZero(args):
    num_games = int(input("Enter the number of games to simulate: "))
    searches = int(input("Enter the number of searches for MCTS: "))
    args['num_searches'] = searches
    # Start timing the simulation
    start_time = time.time()
    results = simulate_ai_vs_random_AlphaZero(num_games, args)  # Assuming simulate_ai_vs_random is defined elsewhere
    end_time = time.time()
    
    # Print results
    print(f"Training time for {num_games} games: {(end_time - start_time)/60:.2f} minutes")
    print(f"Results after {num_games} games:")
    print(f"AI Wins: {results['AI Wins']}")
    print(f"Random Wins: {results['Random Wins']}")
    
args = {
    'C': 1.41,
    'num_searches': 100
}

if __name__ == "__main__":
    print("Choose an option to run:")
    print("1: Play Backgammon Multiplayer")
    print("2: Play Backgammon vs AI")
    print("3: Run AI vs Random Simulation")
    print("4: Run AI vs Random Simulation (AlphaZero)")
    choice = int(input("Enter your choice (1-4): "))

    if choice == 1:
        play_multiplayer()
    elif choice == 2:
        play_vs_ai(args)
    elif choice == 3:
        run_simulation(args)
    elif choice == 4:
        run_simulation_AlphaZero(args)
    else:
        print("Invalid choice. Please run the program again and choose a valid option.")

