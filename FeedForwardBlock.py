import torch

class FeedForwardBlock(torch.nn.Module): # class
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1): # constructor
        super(FeedForwardBlock, self).__init__()
        # Above use: calling the parent class constructor
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        # Above is a fully connected layer that maps the input tensor to a hidden tensor
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        # Above is a fully connected layer that maps the hidden tensor to the output tensor
        self.dropout = torch.nn.Dropout(dropout)
        # Above is a dropout layer to prevent overfitting
        self.relu = torch.nn.ReLU()
        # Above is a ReLU activation function to introduce non-linearity

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":  # Example usage
    feed_forward_block = FeedForwardBlock(input_dim=512, hidden_dim=2048, output_dim=512)
    output = feed_forward_block(torch.randn(64, 512))
    print(output)
    # Above is an example usage of the FeedForwardBlock module