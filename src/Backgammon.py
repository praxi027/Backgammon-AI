import numpy as np 
import random

class Backgammon:
    def __init__(self):
        self.board = np.array([])
        self.board_size = 24  # 24 positions on the board
        self.action_size = 1352 # move encoding is explained below
        
    def get_initial_state(self):
        state = np.zeros(self.board_size, dtype=int)
        state[0] = 15
        state[12] = -15
        # initial state (15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return state
    
    def roll_dice(self):
        return random.choices(range(1, 7), k=2)
    
    def change_perspective(self, state, player):
        if player == -1:
            return -np.roll(state, 12)
        return state
    
    def can_throw_away(self, state):
        return np.all(state[:18] <= 0)
    
    def get_next_state(self, state, action, player): 
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
                elif self.board_size == to_pos and self.can_throw_away(state):  
                    single_moves.append((from_pos, 24))  # Throw away piece
                elif self.board_size < to_pos and self.can_throw_away(state) and np.all(state[:from_pos - 1] <= 0): # checks if there are preceding checkers 
                    single_moves.append((from_pos, 24))  
        return single_moves
    
    def get_valid_moves(self, currentState, player, dice_rolls):
        # Change perspective if player 2
        state = self.change_perspective(currentState.copy(), player)
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

    def get_value_and_terminated(self, state):
        if not np.any(state > 0) or not np.any(state < 0):  
            return 1, True  

        return 0, False  
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def get_all_possible_dice_rolls(self):
        return [
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 3), (3, 4), (3, 5), (3, 6),
            (4, 4), (4, 5), (4, 6),
            (5, 5), (5, 6),
            (6, 6)
        ]

    @staticmethod
    def encode_board_state(board):
        """
        Encodes the board state into separate channels for each player.

        Args:
            board (np.ndarray): A NumPy array of size 24 representing the number of checkers.
                                Positive values for Player 1, negative for Player 2, zero for empty spots.

        Returns:
            player1_channel (np.ndarray): Array of shape (4, 6) for Player 1's checkers.
            player2_channel (np.ndarray): Array of shape (4, 6) for Player 2's checkers.
        """
        # Initialize channels for both players
        player1_channel = np.zeros((4, 6), dtype=np.float32)
        player2_channel = np.zeros((4, 6), dtype=np.float32)

        for point in range(24):
            row = point // 6   # Row index (0 to 3)
            col = point % 6    # Column index (0 to 5)
            checkers = board[point]

            if checkers > 0:
                player1_channel[row, col] = checkers / 15.0  # Normalize by max checkers
            elif checkers < 0:
                player2_channel[row, col] = (-checkers) / 15.0

        return player1_channel, player2_channel

    @staticmethod
    def encode_dice(dice):
        """
        Encodes the dice values into two channels.

        Args:
            dice (tuple): A tuple of two integers representing the dice rolls.

        Returns:
            die1_channel (np.ndarray): Array of shape (4, 6) with the normalized value of Die 1.
            die2_channel (np.ndarray): Array of shape (4, 6) with the normalized value of Die 2.
        """
        die1_value = dice[0] / 6.0  # Normalize to range [0, 1]
        die2_value = dice[1] / 6.0

        # Create channels filled with the normalized dice values
        die1_channel = np.full((4, 6), die1_value, dtype=np.float32)
        die2_channel = np.full((4, 6), die2_value, dtype=np.float32)

        return die1_channel, die2_channel

    def get_encoded_state(self, board, dice):
        p1_channel, p2_channel = self.encode_board_state(board)
        die1_channel, die2_channel = self.encode_dice(dice)

        input_array = np.stack([p1_channel, p2_channel, die1_channel, die2_channel], axis=0)

        return input_array

    
    def get_valid_moves_encoded(self, state, dice_roll):
        valid_moves = self.get_valid_moves(state, 1, dice_roll)
        encoded = np.zeros(26*26*2)
        for move in valid_moves:
            encoded_move = self.encode_move(move)
            encoded[encoded_move] = 1
        return encoded
            
    def encode_move(self, move):
        """
        Encodes a move pair into a unique integer based on die order.
        Handles dummy moves (0, 0) by assigning index 25.
        """
        result = 0
        move1, move2 = move
        dice1 = abs(move1[1] - move1[0]) if move1 != (0, 0) else 0  
        dice2 = abs(move2[1] - move2[0]) if move2 != (0, 0) else 0  

        # Determine if smaller die moves first by checking dice values
        if dice1 < dice2:
            move1, move2 = move2, move1
            result += 26 * 26 

        start1_index = move1[0] if move1 != (0, 0) else 25  
        start2_index = move2[0] if move2 != (0, 0) else 25  

        result += start1_index * 26 + start2_index
        return result

    def decode_move(self, encoded, dice_roll):
        """
        Decodes an encoded integer back into a move pair based on die roll order.
        If a position is 25, it's treated as a dummy move (0, 0).
        If an end position goes beyond 23, it’s capped at 24.
        """
        # Determine if the smaller die moved first based on encoding range
        if encoded >= 26 * 26:
            encoded -= 26 * 26
            smaller_first = True
        else:
            smaller_first = False

        start1 = encoded // 26
        start2 = encoded % 26

        if dice_roll[0] < dice_roll[1]:
            small_die, large_die = dice_roll[0], dice_roll[1]
        else:
            small_die, large_die = dice_roll[1], dice_roll[0]

        if smaller_first:
            move1 = (0, 0) if start2 == 25 else (start2, min(start2 + small_die, 24))
            move2 = (0, 0) if start1 == 25 else (start1, min(start1 + large_die, 24))
        else:
            move1 = (0, 0) if start1 == 25 else (start1, min(start1 + large_die, 24))
            move2 = (0, 0) if start2 == 25 else (start2, min(start2 + small_die, 24))

        return move1, move2