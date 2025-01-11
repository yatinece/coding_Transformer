import torch
import math 
from MultiHeadAttention import MultiHeadAttention
from FeedForwardBlock import FeedForwardBlock
from LayerNormalization import LayerNormalization
from ResidualConnection import ResidualConnection



class EncoderBlock(torch.nn.Module):
    def __init__(self, embedding_vector_size, num_heads, feedforward_dim, dropout=0.1):  # constructor
        super(EncoderBlock, self).__init__()  
        # Above use: calling the parent class constructor
        self.embedding_vector_size = embedding_vector_size
        # above is the size of the embedding vector
        self.num_heads = num_heads
        # above is the number of attention heads
        self.feedforward_dim = feedforward_dim
        # above is the dimension of the feedforward network
        self.dropout = torch.nn.Dropout(dropout)
        # above is the dropout layer
        self.ln1 = LayerNormalization(embedding_vector_size)
        # above is the first layer normalization layer
        self.mha = MultiHeadAttention(embedding_vector_size, num_heads , dropout)
        # above is the multi-head attention layer
        self.ln2 = LayerNormalization(embedding_vector_size)
        # above is the second layer normalization layer
        self.ffn = FeedForwardBlock(embedding_vector_size, feedforward_dim, embedding_vector_size, dropout)
        # above is the feedforward network layer
        # The feedforward network layer is initialized with the embedding vector size and feedforward dimension
        self.residual_connection1 = ResidualConnection(embedding_vector_size, dropout)
        # above is the first residual connection layer
        self.residual_connection2 = ResidualConnection(embedding_vector_size, dropout)

    def forward(self, x, mask): # forward method

        x =self.residual_connection1( x , lambda x: self.mha(x, x, x, mask))
        # Above is the first residual connection layer applied to the input tensor x
        # The input tensor x is passed through the multi-head attention layer with the mask

        x = self.residual_connection2(x, self.ffn)
        # Above is the second residual connection layer applied to the input tensor x
        # The input tensor x is passed through the feedforward network layer

        return x  # The output tensor is returned


if __name__ == "__main__": # Example usage
    encoder_block = EncoderBlock(embedding_vector_size=512, num_heads=8, feedforward_dim=2048)
    output = encoder_block(torch.randn(64, 10, 512), mask=None)
    print(output)
    print(output.shape)
    # Above is an example usage of the EncoderBlock module
