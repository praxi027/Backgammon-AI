import numpy as np 
import random

class Backgammon:
    def __init__(self):
        self.board = np.array([])
        self.board_size = 24  # 24 positions on the board
        
    def get_initial_state(self):
        state = np.zeros(self.board_size, dtype=int)
        state[0] = 2   
        state[12] = -2
        # initial state (15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return state
    
    def roll_dice(self):
        dice_rolls = (random.randint(1, 6), random.randint(1, 6))
        return dice_rolls
    
    def change_perspective(self, state, player):
        shifted_state = np.zeros(self.board_size, dtype=int)
        if player == -1:
            # Shift the state for player 2
            shifted_state = state.copy()
            for i in range(self.board_size):
                shifted_state[i-12] = state[i]
            return shifted_state
        return state  # For player 1, no shift is needed
    
    def can_throw_away(self, state, player):
        # Check if all pieces are in the final quarter (positions 18 to 23) and not occupied by the opponent
        for pos in range(0, 18):
            if state[pos] * player > 0:  # If any position is occupied by the opponent or empty
                return False
        return True
    
    def get_next_state(self, state, action, player): 
        # Unpack the action tuple
        first_move, second_move = action
        if (player == -1):
            state = self.change_perspective(state, player)

        # First move
        from_pos1, to_pos1 = first_move
        state[from_pos1] -= player  
        if(to_pos1 != 'throw_away'): state[to_pos1] += player  

        # Second move
        from_pos2, to_pos2 = second_move
        state[from_pos2] -= player  
        if(to_pos2 != 'throw_away'): state[to_pos2] += player  
        
        return self.change_perspective(state, player) if player == -1 else state

    
    def find_single_moves(self, state, player, roll):
        single_moves = []
        for from_pos in range(self.board_size):
            if state[from_pos] * player > 0:  # Player has pieces at from_pos
                to_pos = from_pos + roll 
                if 0 <= to_pos < self.board_size and state[to_pos] * player >= 0:  # Valid move (empty or own pieces)
                    single_moves.append((from_pos, to_pos))
                elif 0 <= to_pos and self.can_throw_away(state, player):  
                    single_moves.append((from_pos, 'throw_away'))
        return single_moves
    
    def get_valid_moves(self, state, player, dice_rolls):
        # Change perspective if player 2
        state = self.change_perspective(state, player)
        valid_moves = set()
        # should try to play the biggest dice first if both moves not possible
        dice_rolls = sorted(dice_rolls, reverse=True)
        
        first_moves = self.find_single_moves(state, player, dice_rolls[0])  # Find valid moves for the biggest dice
        if not first_moves:
            dice_rolls = sorted(dice_rolls)
            first_moves = self.find_single_moves(state, player, dice_rolls[0])  # if biggest dice not possible, try smallest first
            if not first_moves:
                return [((0, 0), (0, 0))]  # No moves possible, return a dummy move
            
        for move in first_moves:
            new_state = state.copy()
            from_pos1, to_pos1 = move
            # Create a new state after the first move
            new_state[from_pos1] -= player  
            if(to_pos1 != 'throw_away'): new_state[to_pos1] += player  

            # Find valid second moves for the new state using the second die roll
            second_moves = self.find_single_moves(new_state, player, dice_rolls[1])
            if(not second_moves):
                valid_moves.add((move, (0, 0)))
            for second_move in second_moves:
                from_pos2, to_pos2 = second_move  
                if(from_pos1 == from_pos2 == 0):    # Check if two moves were made from index 0
                    continue
                move_pair = tuple(sorted([(from_pos1, to_pos1), (from_pos2, to_pos2)]))     
                valid_moves.add(move_pair)

        return list(valid_moves)

    def get_value_and_terminated(self, state):
        if not np.any(state > 0):  
            return 1, True  

        if not np.any(state < 0): 
            return -1, True  

        return 0, False  
                                    

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
        print("Final State:", state)
        if value == -1:
            print("Player 2 wins!")
        elif value == 1:
            print("Player 1 wins!")
        else:
            print("Game ended in a draw!")
        break
        
    player = -player  # Switch player