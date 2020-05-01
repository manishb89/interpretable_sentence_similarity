from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import numpy as np
import datetime
try:
    from pytorch_pretrained_bert import BertTokenizer
except:
    print 'pytorch_pretrained_bert not available'

# Change tempdir so gensim get_tmpfile doesn't clog /tmp on gpu box
import tempfile, platform
if platform.system() == 'Linux':
    tempfile.tempdir = '/var/nvidia/tmp'

# What Else
np.random.seed(42)


class EmbeddingReader(object):

    def __init__(self, emb_file_path, bert_model_name=None, emb_type="glove", special_symbols=["-not aligned-"]):
        self.emb_file_path = emb_file_path
        assert isinstance(emb_type, str), "emb_type must be a string"
        emb_type = emb_type.lower()
        assert emb_type in ("glove", "chunk", "bert"), "emb_type must be either glove/chunk or bert"
        self.emb_type = emb_type
        self.embedding = None
        self.special_symbols = special_symbols
        self.special_symbols_emb = []
        self.bert_model_name = bert_model_name
        self.unk_symbol_emb = None
        self.unk_token_idx = None
        self._read_status = False
        self.bert_tokenizer = None
        
    def _read(f):
        def wrapper(self, *args, **kwargs):
            if not self._read_status:

                if self.emb_type == "bert":
                    self.embedding = KeyedVectors.load_word2vec_format(self.emb_file_path)
                    self.stoi = {w:i for i,w in enumerate(self.embedding.index2word)}
                    self.itos = {i:w for i,w in enumerate(self.embedding.index2word)}
                    num_words = len(self.embedding.index2word)
                    self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
                    
                if self.emb_type == "glove":
                    glove_fp = datapath(self.emb_file_path)
                    tmp_file = get_tmpfile("_temp_w2v_{}.txt".format(datetime.datetime.now().strftime('%H_%M_%S')))
                    _ = glove2word2vec(glove_fp, tmp_file)
                    self.embedding = KeyedVectors.load_word2vec_format(tmp_file)

                elif self.emb_type == "chunk":
                    self.embedding = KeyedVectors.load(self.emb_file_path)
                    self.stoi = {w: i for i, w in enumerate(self.embedding.vocab.keys())}
                    self.itos = {i: w for i, w in enumerate(self.embedding.vocab.keys())}
                    num_words = len(self.embedding.vocab.keys())

                    
                if self.emb_type == "glove":
                    self.stoi = {w:i for i,w in enumerate(self.embedding.index2word)}
                    self.itos = {i:w for i,w in enumerate(self.embedding.index2word)}
                    num_words = len(self.embedding.index2word)

                dim,idx = self.embedding.vector_size, num_words
                for s in self.special_symbols:
                    self.stoi[s] = idx
                    self.itos[idx] = s
                    self.special_symbols_emb.append(2 * np.random.random(dim) - 1)
                    idx += 1

                self.special_symbols_emb = np.array(self.special_symbols_emb)
                
                self.unk_symbol_emb = np.zeros((1,dim))
                self.unk_token_idx = num_words + len(self.special_symbols)

                self.pad_symbol_emb = np.zeros((1,dim))
                self.pad_token_idx = self.unk_token_idx + 1

                self.stoi["-UNK-"] = self.unk_token_idx
                self.itos[self.unk_token_idx] = "-UNK-"

                self.stoi["-PAD-"] = self.pad_token_idx
                self.itos[self.pad_token_idx] = "-PAD-"
                self._read_status = True
            return f(self, *args, **kwargs)
        return wrapper

    @_read
    def vocab_len(self):
        return len(self.stoi) + len(self.special_symbols) + 2

    @_read
    def vocab_dim(self):
        return self.embedding.vector_size

    @_read
    def vocab_to_index(self, seq):
        indices = []
        for s in seq:
            idx = self.stoi.get(s, self.unk_token_idx)
            indices.append(idx)
        return indices

    @_read
    def embedding_matrix(self):
        """
        Creates a copy of the underlying embedding matrix, use judiciously 
        @return: Embedding matrix with expanded vocabulary 
        """
        if self.emb_type == "glove":
            emb_matrix = self.embedding.vectors.copy()
        elif self.emb_type == "chunk":
            emb_matrix = np.array(self.embedding.vocab.values())

        emb_matrix = np.concatenate((emb_matrix, self.special_symbols_emb), axis = 0)
        emb_matrix = np.concatenate((emb_matrix, self.unk_symbol_emb), axis = 0)
        emb_matrix = np.concatenate((emb_matrix, self.pad_symbol_emb), axis = 0)
        return emb_matrix
            
            
