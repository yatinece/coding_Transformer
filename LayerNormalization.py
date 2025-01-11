import torch 
import math 

class LayerNormalization(torch.nn.Module):
    def __init__(self, embedding_vector_size, eps=1e-6):  # constructor
        super(LayerNormalization, self).__init__()  
        # Above use: calling the parent class constructor
        self.embedding_vector_size = embedding_vector_size
        # above is the size of the embedding vector
        self.eps = eps
        # above is the epsilon value used for numerical stability
        self.alpha = torch.nn.Parameter(torch.ones(embedding_vector_size))
        # above is a tensor of ones of shape (embedding_vector_size)
        # The tensor is wrapped in a torch.nn.Parameter to make it learnable
        self.bias = torch.nn.Parameter(torch.zeros(embedding_vector_size))
        # above is a tensor of zeros of shape (embedding_vector_size)
        # The tensor is wrapped in a torch.nn.Parameter to make it learnable

    def forward(self, input): # forward method  # differentiation is done here
        mean = input.mean(-1, keepdim=True)
        # Above is the mean of the input tensor along the last dimension
        # keepdim=True ensures that the output tensor has the same number of dimensions as the input tensor
        # input.mean(-1) will return a tensor of shape (batch_size, sequence_length) if keepdim=False
        # input.mean(-1, keepdim=True) will return a tensor of shape (batch_size, sequence_length, 1)
        std = input.std(-1, keepdim=True)
        # Above is the standard deviation of the input tensor along the last dimension
        return self.alpha * (input - mean) / (std + self.eps) + self.bias
        # Above is the layer normalization operation applied to the input tensor
        # The input tensor is normalized using the mean and standard deviation along the last dimension
        # The normalized tensor is then scaled by alpha and shifted by bias
        # The normalized tensor is returned
        # Parameters:
            #input: The input tensor of shape (batch_size, sequence_length, embedding_vector_size) to be normalized.
        # Returns:
            #The normalized tensor of shape (batch_size, sequence_length, embedding_vector_size).


if __name__ == "__main__": # Example usage 
    ln = LayerNormalization(embedding_vector_size=16)
    sample_input = torch.randn(2, 4, 16)  # Batch of 2 sequences
    output = ln(sample_input)
    print(output)
    # Above is an example usage of the LayerNormalization module
