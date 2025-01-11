import torch
import math 
from LayerNormalization import LayerNormalization

class ResidualConnection(torch.nn.Module):
    def __init__(self, embedding_vector_size, dropout=0.1):  # constructor
        super(ResidualConnection, self).__init__()  
        # Above use: calling the parent class constructor
        self.embedding_vector_size = embedding_vector_size
        # above is the size of the embedding vector
        self.dropout = torch.nn.Dropout(dropout)
        # above is the dropout layer
        self.ln = LayerNormalization(self.embedding_vector_size) #torch.nn.LayerNorm(embedding_vector_size)
        # above is the layer normalization layer
        # The layer normalization layer is initialized with the embedding vector size

    def forward(self, x, sublayer): # forward method
        return x + self.dropout(sublayer(self.ln(x)))
        # Above is the forward method that applies the residual connection to the input tensor x
        # The input tensor x is passed through the layer normalization layer
        # The output of the layer normalization layer is passed through the sublayer
        # The output of the sublayer is passed through the dropout layer
        # The output of the dropout layer is added to the input tensor x
        # The sum is returned
        # Parameters:
            #x: The input tensor of shape (batch_size, sequence_length, embedding_vector_size).
            #sublayer: The sublayer to be applied to the input tensor x.
        # Returns:
            #The output tensor of shape (batch_size, sequence_length, embedding_vector_size) after applying the residual connection.


if __name__ == "__main__": # Example usage
    rc = ResidualConnection(embedding_vector_size=16)
    sample_input = torch.randn(2, 4, 16)  # Batch of 2 sequences
    output = rc(sample_input, lambda x: x * 2)
    print(output)
    # Above is an example usage of the ResidualConnection module