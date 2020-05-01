
import torch.nn as nn
import torch

class ChunkEmbedding():

    def __init__(self, emb_reader):
        self.vocab_size = emb_reader.vocab_len()
        self.vocab_dim = emb_reader.vocab_dim()
        self.padding_idx = self.vocab_size - 1 
        self.emb_reader = emb_reader
        self.embedding = None
        
    def _load_pretrained(f):
        def wrapper(self, *args, **kwargs):
            weight_matrix = self.emb_reader.embedding_matrix()
            weight_matrix = torch.FloatTensor(weight_matrix)
            self.embedding = nn.Embedding.from_pretrained(weight_matrix)
            self.embedding.weight.requires_grad = False
            return f(self, *args, **kwargs)
        return wrapper
    
    @_load_pretrained
    def torch_chunk_embedding(self):
        return self.embedding
