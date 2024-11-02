import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, game, num_blocks, num_hidden):
        super(ResNet, self).__init__()
        num_input_channels = 4
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_input_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_blocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 6,  1352)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 6, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        
        return policy, value
    
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

    
def encode_board_state(board):
    """
    Encodes the board state into separate channels for each player.

    Args:
        board (np.ndarray): A NumPy array of size 24 with integers representing the number of checkers.
                            Positive for Player 1, negative for Player 2, zero for empty spots.

    Returns:
        player1_channel (torch.Tensor): Tensor of shape (4, 6) for Player 1's checkers.
        player2_channel (torch.Tensor): Tensor of shape (4, 6) for Player 2's checkers.
    """
    # Initialize channels for both players
    player1_channel = torch.zeros(4, 6)
    player2_channel = torch.zeros(4, 6)

    for point in range(24):
        side = point // 6   # Row index (0 to 3)
        spot = point % 6    # Column index (0 to 5)
        checkers = board[point]

        if checkers > 0:
            # Normalize the number of checkers (e.g., divide by 15, the maximum number of checkers per player)
            player1_channel[side, spot] = checkers / 15.0
        elif checkers < 0:
            player2_channel[side, spot] = (-checkers) / 15.0
        # Empty spots remain zero

    return player1_channel, player2_channel

def encode_dice(dice):
    """
    Encodes the dice values into two channels.

    Args:
        dice (tuple): A tuple of two integers representing the dice rolls.

    Returns:
        die1_channel (torch.Tensor): Tensor of shape (4, 6) with the normalized value of Die 1.
        die2_channel (torch.Tensor): Tensor of shape (4, 6) with the normalized value of Die 2.
    """
    die1_value = dice[0] / 6.0  # Normalize to range [0, 1]
    die2_value = dice[1] / 6.0

    # Create channels filled with the normalized dice values
    die1_channel = torch.full((4, 6), die1_value)
    die2_channel = torch.full((4, 6), die2_value)

    return die1_channel, die2_channel

def get_encoded_state(board, dice):
    """
    Combines the encoded board and dice into the final input tensor.

    Args:
        board (np.ndarray): The board state as a NumPy array of size 24.
        dice (tuple): A tuple of two integers representing the dice rolls.

    Returns:
        input_tensor (torch.Tensor): The input tensor of shape (1, 4, 4, 6).
    """
    # Encode the board state
    p1_channel, p2_channel = encode_board_state(board)

    # Encode the dice values
    die1_channel, die2_channel = encode_dice(dice)

    # Stack the channels to form the input tensor
    input_tensor = torch.stack([p1_channel, p2_channel, die1_channel, die2_channel], dim=0)

    # Add batch dimension (set to 1 for a single sample)
    input_tensor = input_tensor.unsqueeze(0)  # Shape: (1, 4, 4, 6)

    return input_tensor


if __name__ == "__main__":
    # Define the board state
    board = np.zeros(24, dtype=int)
    board[0] = 5       # Player 1
    board[5] = 3       # Player 1
    board[18] = -4     # Player 2
    board[23] = -2     # Player 2

    # Define the dice values
    dice = (3, 5)

    # Get the encoded state
    input_tensor = get_encoded_state(board, dice)
    print("Input Tensor Shape:", input_tensor.shape)  # Output: (1, 4, 4, 6)

    # Initialize the game object with board dimensions
    class BackgammonGame:
        def __init__(self):
            self.board_height = 4
            self.board_width = 6
            self.action_size = 1000  # Adjust based on your action space

    game = BackgammonGame()

    # Initialize the neural network
    model = ResNet(game, 4, 64)

    # Forward pass
    policy, value = model(input_tensor)
    
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print(value, policy)



