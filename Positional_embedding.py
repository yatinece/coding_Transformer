import torch 
import math 

# Used to create positional emnedding for transformer input  for rotary, sinusoidal and learnable positional encoding, 
# please read the below formulas 
# sinonoidal positional encoding: forumla: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#learnable positional encoding: forumla: PE(pos, 2i) = learnable(pos, 2i) and PE(pos, 2i+1) = learnable(pos, 2i+1) 
#rotary positional angle encoding: forumla: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) 
# this is applied as a rotation x = x * cos(pos) - y * sin(pos) and y = x * sin(pos) + y * cos(pos)

class PositionalEmbedding(torch.nn.Module): # class
    def __init__(self, embedding_vector_size, sequence_length , dropout , type_method="sinusoidal"):  # constructor
        super(PositionalEmbedding, self).__init__()  
        # Above use: calling the parent class constructor
        self.embedding_vector_size = embedding_vector_size 
        # above is the size of the embedding vector
        self.sequence_length = sequence_length 
        #   above is the sequence length
        self.dropout= torch.nn.Dropout(dropout) 
        self.type =type_method 
        # dropout
        if type_method=="sinusoidal":
            self.sinusoidal_positional_encoding(sequence_length, embedding_vector_size)
        # Above is the positional encoding function that generates the positional embedding matrix
        elif type_method=="learnable":
            self.learnable_positional_embedding(sequence_length, embedding_vector_size)  
        # Above is the learnable positional encoding function that generates the learnable positional embedding matrix
        elif type_method=="rotary":
            assert embedding_vector_size % 2 == 0, "Embedding dimension must be even for rotary encoding."
            self.rotary_positional_encoding(sequence_length, embedding_vector_size)
        # Above is the rotary positional encoding function that generates the rotary positional embedding matrix
        else:
            raise ValueError(f"Unknown positional encoding type: {type_method}")

    def learnable_positional_embedding(self, sequence_length, embedding_vector_size):
        # Create a learnable positional embedding matrix 
        # This matrix will be added to the input embeddings to introduce positional information.
        # The matrix is of shape (sequence_length, embedding_vector_size)

        positional_embedding = torch.nn.Parameter(torch.randn(sequence_length, embedding_vector_size))
        # Above is a tensor of zeros of shape (sequence_length, embedding_vector_size)
        # The tensor is wrapped in a torch.nn.Parameter to make it learnable

        self.register_parameter('positional_embedding', positional_embedding)
        self.positional_embedding=positional_embedding



    def rotary_positional_encoding(self, sequence_length, embedding_vector_size):
        # Create a rotary positional encoding matrix 
        # This matrix will be added to the input embeddings to introduce positional information.
        # The matrix is of shape (sequence_length, embedding_vector_size)
        # The rotary positional encoding matrix is created by concatenating multiple sinusoidal positional encoding matrices along different axes
        # Compute the positional embeddings
        positional_embedding = torch.randn(sequence_length, embedding_vector_size)
        position = torch.arange(0, sequence_length , dtype =  torch.float).unsqueeze(1)
        # Above is a torch.arange creating a 10 dimension tensor and unsqueexe making it tensor of shape (sequence_length, 1) containing values from 0 to sequence_length - 1
        scaling_factor = torch.exp (torch.arange(0,embedding_vector_size,2).float() * math.log(10000.0) / embedding_vector_size)
        # Above is a tensor of shape (embedding_vector_size // 2) containing values for the scale
        sin_encoding = torch.sin(position * scaling_factor)
        cos_encoding = torch.cos(position * scaling_factor)
        positional_embedding[..., 0::2] = sin_encoding
        positional_embedding[..., 1::2] = cos_encoding
        positional_embedding = positional_embedding.unsqueeze(0)
        # Above is the positional embedding matrix unsqueezed to shape (1, sequence_length, embedding_vector_size)
        # This is done to allow adding the positional embeddings to the input embeddings directly

        self.register_buffer('positional_embedding', positional_embedding)
        self.positional_embedding=positional_embedding # register buffer with the positional embedding matrix
        # Buffer is saved in the state_dict and moved to the device along with the model
        # if you want to save the model and load it in another device, this buffer will be moved to the new device along with the model

    def sinusoidal_positional_encoding(self, sequence_length, embedding_vector_size):
        # Create a positional embedding matrix 
        # This matrix will be added to the input embeddings to introduce positional information.
        # The matrix is of shape (sequence_length, embedding_vector_size)

        positional_embedding = torch.zeros(sequence_length, embedding_vector_size)
        # Above is a tensor of zeros of shape (sequence_length, embedding_vector_size)

        # Compute the positional embeddings
        position = torch.arange(0, sequence_length , dtype =  torch.float).unsqueeze(1)
        # Above is a torch.arange creating a 10 dimension tensor and unsqueexe making it tensor of shape (sequence_length, 1) containing values from 0 to sequence_length - 1
        div_term = torch.exp(torch.arange(0, embedding_vector_size, 2).float() * (-math.log(10000.0) / embedding_vector_size))
        # Above is a tensor of shape (embedding_vector_size // 2) containing values for the denominator of the positional encoding function
        positional_embedding[..., 0::2] = torch.sin(position * div_term)
        positional_embedding[..., 1::2] = torch.cos(position * div_term)
        # Above is the positional embedding matrix  
        positional_embedding = positional_embedding.unsqueeze(0)
        # Above is the positional embedding matrix unsqueezed to shape (1, sequence_length, embedding_vector_size)
        # This is done to allow adding the positional embeddings to the input embeddings directly

        self.register_buffer('  ', positional_embedding)
        self.positional_embedding=positional_embedding # register buffer with the positional embedding matrix
        # Buffer is saved in the state_dict and moved to the device along with the model
        # if you want to save the model and load it in another device, this buffer will be moved to the new device along with the model


    def forward(self, input): # forward method  # differentiation is done here
        if self.type=="rotary":
            # Split the input embeddings into even and odd indices along the last dimension
            input_even = input[... , 0::2]
            input_odd = input[... , 1::2]
            # Split the positional embeddings into even and odd indices along the last dimension
            sin_encoding = self.positional_embedding[... , 0::2]
            cos_encoding = self.positional_embedding[... , 1::2]
            
            # Apply the rotary transformation
            rotated_embeddings = torch.empty_like(input , dtype = torch.float)
            rotated_embeddings[..., ::2] = input_even * cos_encoding - input_odd * sin_encoding
            rotated_embeddings[..., 1::2] = input_even * sin_encoding + input_odd * cos_encoding

            return self.dropout(rotated_embeddings) # return the input embeddings with positional information added and dropout applied

        # Above we add the learnable positional embedding matrix to the input embeddings
        else:
            return self.dropout(input + self.positional_embedding)
        input = input + self.positional_embedding 
        # Above we add the positional embedding matrix to the input embeddings




if __name__ == "__main__": # Example usage
    embedding_vector_size=16 
    sequence_length=10 
    dropout=0.1
    emb = PositionalEmbedding(embedding_vector_size, sequence_length, dropout , type_method="learnable")
    sample_input = torch.randn(2, 10, 16)  # Batch of 2 sequences
    output = emb(sample_input)
    print(output)