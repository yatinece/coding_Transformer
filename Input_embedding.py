import torch 
import math


# Used to create input emnedding for transformer input
# this will convert the input tnto a vector 

class InputEmbedding(torch.nn.Module):
    def __init__(self, embedding_vector_size, vocab_size):  # constructor
        super(InputEmbedding, self).__init__()  
        # Above use: calling the parent class constructor, not using might imact the code and forward method
        self.embedding = torch.nn.Embedding(vocab_size, embedding_vector_size) 
        # Above is a lookup table that maps from integer indices (representing words or tokens) to dense vectors of fixed size.
        # Parameters:
            #vocab_size: The size of the vocabulary, i.e., the number of unique tokens.
            #embedding_vector_size: The size of each embedding vector, i.e., the dimensionality of the dense vector for each token.
        self.embedding_vector_size = embedding_vector_size 
        # above is the size of the embedding vector
        self.vocab_size = vocab_size

    def forward(self, input): # forward method  # differentiation is done here
        return self.embedding(input) * math.sqrt(self.embedding_vector_size) 
        # Above is the forward method that takes an input tensor of shape (batch_size, sequence_length) 
        # and returns the corresponding embedding tensor of shape (batch_size, sequence_length, embedding_vector_size). 
        # The embedding tensor is then multiplied by the square root of the embedding vector size and returned.
        # Parameters:
            #input: The input tensor of shape (batch_size, sequence_length) containing integer indices representing words or tokens.
        # Returns:
            #The embedding tensor of shape (batch_size, sequence_length, embedding_vector_size) for the input tensor.


# Example usage
if __name__ == "__main__":
    emb = InputEmbedding(embedding_vector_size=16, vocab_size=100)
    sample_input = torch.tensor([[1, 2, 2, 3], [4, 5, 6, 6]])  # Batch of 2 sequences
    output = emb(sample_input)
    print(output)