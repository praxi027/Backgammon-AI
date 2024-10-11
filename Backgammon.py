import numpy as np 
import random

class Backgammon:
    def __init__(self):
        self.board = np.array([])
        self.board_size = 24  # 24 positions on the board
        self.action_size = 225  # 15 possible moves
        
    def get_initial_state(self):
        state = np.zeros(self.board_size, dtype=int)
        state[0] = 15
        state[12] = -15
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
                shifted_state[i-12] = -state[i]
            return shifted_state
        return state  # For player 1, no shift is needed
    
    def can_throw_away(self, state):
        # Check if all pieces are in the final quarter (positions 18 to 23) and not occupied by the opponent
        for pos in range(0, 18):
            if state[pos] > 0:  # If any position is occupied by the opponent or empty
                return False
        return True
    
    def get_next_state(self, state, action, player): 
        # Unpack the action tuple
        first_move, second_move = action
        if (player == -1):
            state = self.change_perspective(state, player)

        # First move
        from_pos1, to_pos1 = first_move
        state[from_pos1] -= 1  
        if(to_pos1 != 24): state[to_pos1] += 1 

        # Second move
        from_pos2, to_pos2 = second_move
        state[from_pos2] -= 1 
        if(to_pos2 != 24): state[to_pos2] += 1
        
        return self.change_perspective(state, player) if player == -1 else state
    
    def find_single_moves(self, state, roll, can_play_from_bar=True):
        single_moves = []
        start_position = 0 if can_play_from_bar else 1   
        for from_pos in range(start_position, self.board_size):
            if state[from_pos] > 0:  # Player has pieces at from_pos
                to_pos = from_pos + roll 
                if 0 <= to_pos < self.board_size and state[to_pos] >= 0:  # Valid move (empty or own pieces)
                    single_moves.append((from_pos, to_pos))
                elif 0 <= to_pos and self.can_throw_away(state):  
                    single_moves.append((from_pos, 24))  # Throw away piece
        return single_moves
    
    def get_valid_moves(self, currentState, player, dice_rolls):
        # Change perspective if player 2
        state = self.change_perspective(currentState, player)
        valid_moves = set()
        # should try to play the biggest dice first if both moves not possible
        dice_rolls = sorted(dice_rolls, reverse=True)
        
        first_moves = self.find_single_moves(state, dice_rolls[0])  # Find valid moves for the biggest dice
        if not first_moves:
            dice_rolls = sorted(dice_rolls)
            first_moves = self.find_single_moves(state, dice_rolls[0])  # if biggest dice not possible, try smallest first
            if not first_moves:
                return [((0, 0), (0, 0))]  # No moves possible, return a dummy move
            
        for move in first_moves:
            new_state = state.copy()
            from_pos1, to_pos1 = move
            # Create a new state after the first move
            new_state[from_pos1] -= 1  
            if(to_pos1 != 24): new_state[to_pos1] += 1

            # Find valid second moves for the new state using the second die roll
            second_moves = self.find_single_moves(new_state, dice_rolls[1], can_play_from_bar=from_pos1 != 0)
            if(not second_moves):
                valid_moves.add((move, (0, 0)))
            for second_move in second_moves:
                from_pos2, to_pos2 = second_move  
                move_pair = tuple(sorted([(from_pos1, to_pos1), (from_pos2, to_pos2)]))     
                valid_moves.add(move_pair)

        return list(valid_moves)

    def get_value_and_terminated(self, state, player):
        if not np.any(state > 0) and player == 1:  
            return 1, True  

        if not np.any(state < 0) and player == -1: 
            return 1, True  

        return 0, False  
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def get_all_possible_dice_rolls(self):
        return [(i, j) for i in range(1, 7) for j in range(i, 7)]
    
    def get_encoded_state(self, state):
        player1_layer = np.clip(state, 0, None).astype(np.float32)
        player2_layer = np.clip(-state, 0, None).astype(np.float32)
        empty_layer = (state == 0).astype(np.float32)
        encoded_state = np.stack((player1_layer, player2_layer, empty_layer))
        
        return encoded_state