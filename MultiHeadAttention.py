import torch
import math

class MultiHeadAttention(torch.nn.Module):  # class
    def __init__(self, embedding_vector_size, num_heads, dropout=0.1):  # constructor
        super(MultiHeadAttention, self).__init__()
        # Above use: calling the parent class constructor
        self.embedding_vector_size = embedding_vector_size
        # above is the size of the embedding vector 
        self.num_heads = num_heads
        # above is the number of attention heads
        assert embedding_vector_size % num_heads == 0, "Embedding size must be divisible by the number of heads."

        self.head_dim = embedding_vector_size // num_heads
        # above is the dimension of each attention head example 512/8=64
        self.scale = self.head_dim ** -0.5
        # above is the scaling factor for the dot product attention
        self.dropout = torch.nn.Dropout(dropout)            
        # above is a dropout layer to prevent overfitting
        self.query = torch.nn.Linear(embedding_vector_size, embedding_vector_size)
        # above is a linear layer for the query projection, think like query is a question or like in a hash table query is a value looking for
        self.key = torch.nn.Linear(embedding_vector_size, embedding_vector_size)
        # above is a linear layer for the key projection, think like key is a key to the answer or like in a hash table key is a hask
        self.value = torch.nn.Linear(embedding_vector_size, embedding_vector_size)
        # above is a linear layer for the value projection , think like value is the answer or like in a hash table value is a data
        self.fc_out = torch.nn.Linear(embedding_vector_size, embedding_vector_size)
        # above is a linear layer for the output projection, think like output is the final answer
        # The output tensor is then returned
        # Parameters:
            #query: The query tensor of shape (batch_size, sequence_length, embedding_vector_size).
            #key: The key tensor of shape (batch_size, sequence_length, embedding_vector_size).
            #value: The value tensor of shape (batch_size, sequence_length, embedding_vector_size).
        # Returns:  The output tensor of shape (batch_size, sequence_length, embedding_vector_size).
        # The output tensor is the result of applying the scaled dot-product attention mechanism to the query, key, and value tensors.
        # The output tensor is then returned


    @staticmethod
    def attention(query, key, value,  mask=None , dropout = None):  # attention method
        scale = query.size(-1) ** -0.5
        # Compute the dot product of the query and key tensors
        attention_scores = torch.matmul(query, key.permute(0 ,1, 3, 2)) / scale
        # Above is the dot product of the query and key tensors
        # The query tensor is multiplied    by the key tensor transposed along the last two dimensions
        # The result is then scaled by the square root of the head dimension
        # The attention_scores tensor has shape (batch_size, num_heads, query_len, key_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        # Above is the mask tensor
        # The scores tensor is masked by setting the masked elements to negative infinity
        attention_scores = attention_scores.softmax(dim=-1)
        # Above is the softmax function applied to the attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Above is the dropout layer applied to the scores

        values = torch.matmul(attention_scores, value)
        # Above is the dot product of the attention and value tensors
        # The attention tensor is multiplied by the value tensor
        # The result is the weighted sum of the value tensor
        # The values tensor has shape (batch_size, num_heads, query_len, head_dim)
        return values , attention_scores


    def forward(self, query, key, value, mask=None):  # forward method  # differentiation is done here
        head_dim = self.head_dim    
        # above is the dimension of each attention head example 512/8=64
        batch_size = query.shape[0]
        # above is the batch size
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        # above is the length of the query, key, and value sequences
        # Split the embedding_vector_size into num_heads heads
        query = self.query(query) # batch size, query len, embedding vector size
        key = self.key(key) # batch size, key len, embedding vector size
        value = self.value(value) # batch size, value len, embedding vector size
        # above is the query, key, and value projections
        query = query.view(batch_size, query_len, self.num_heads, head_dim)   # batch size, query len, num heads, head dim
        key = key.view(batch_size, key_len, self.num_heads, head_dim) # batch size, key len, num heads, head dim
        value = value.view(batch_size, value_len, self.num_heads, head_dim) # batch size, value len, num heads, head dim
        # above is the reshaping of the query, key, and value tensors to split the embedding_vector_size into embedding which heads will attend to 
        query = query.permute(0, 2, 1, 3) # batch size, num heads, query len, head dim
        key = key.permute(0, 2, 1, 3)   # batch size, num heads, key len, head dim
        value = value.permute(0, 2, 1, 3) # batch size, num heads, value len, head dim
        
        x , attention = MultiHeadAttention.attention(query, key, value, mask , self.dropout)

        # batch size, num heads, query len, head dim

        x =x.permute(0,2,1,3).contiguous() # batch size, query len, num heads, head dim
        # contiguous() is used to make the memory contiguous , if not used it will raise an error because of the permutation

        x = x.view(batch_size, query_len, self.embedding_vector_size) # batch size, query len, embedding vector size

        x = self.fc_out(x)

        # above is the output projection
        return x

# Example usage
if __name__ == "__main__":
    embed_size = 32
    num_heads = 4
    batch_size = 2
    query_len = 5
    key_len = 5
    model = MultiHeadAttention(embed_size, num_heads)
    query = torch.randn(batch_size, query_len, embed_size)
    values = torch.randn(batch_size, key_len, embed_size)
    keys = torch.randn(batch_size, key_len, embed_size)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
        ]).unsqueeze(1).unsqueeze(2) 
    print(mask.shape)
    output = model(values, keys, query, mask)
    print(output.shape)