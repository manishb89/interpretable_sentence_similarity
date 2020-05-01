import xml.etree.ElementTree as ET
import pprint
import codecs
import torch
import numpy as np
import json
from embed import vocab as VB
from xml.sax.saxutils import escape
from nltk.tree import Tree as NLTKTree
from tqdm import tqdm
from itertools import product
import syntactic as SS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial
from itertools import chain
from torch.utils.data import Dataset as TorchDataset
from embed import embedding as EM
from nltk.corpus import stopwords


STOP_WORDS = {w : 1.0 for w in {'a', 'in', 'the', 'of', 'on', 'to'}}
STOP_WORDS["-PAD-"] = 0.0


def chunk_tiler(text_chunks, chunks, accumulator):
    if text_chunks is None or text_chunks == "":
        return []

    if text_chunks in chunks:
        accumulator.append(text_chunks)
        return

    else:
        for idx in range(len(text_chunks)):
            if text_chunks[:idx + 1] in chunks:
                chunk_tiler(text_chunks[idx + 1:], chunks, accumulator)
                accumulator.append(text_chunks[:idx + 1])
                return
            

class SentenceAlignment(object):
    """
    Representation of chunk alignments in a sentence
    """
    def __init__(self, alignment_text, lc, rc, s1, s2, idx):
        self.alignment_text = alignment_text
        self.lc = lc
        self.rc = rc
        self.chunks = []
        self.sidx = idx
        self.s1 = s1
        self.s2 = s2
        self.form_alignments()
        
    def form_alignments(self):
        chunks_text = self.alignment_text.split("\n")
        for chunk_text in chunks_text:
            chunk_text = chunk_text.strip()
            if not chunk_text: continue
            chunk_token_id1, chunk_token_id2 = chunk_text.split("//")[0].split("==")

            chunk_token_id1 = [int(idx) for idx in chunk_token_id1.split()]
            chunk_token_id2 = [int(idx) for idx in chunk_token_id2.split()]

            chunk_text1, chunk_text2 = chunk_text.split("//")[-1].split("==")
            chunk_text1, chunk_text2 = chunk_text1.strip(), chunk_text2.strip()

            chunk_seg1 = zip(chunk_text1.split(), chunk_token_id1)
            chunk_seg2 = zip(chunk_text2.split(), chunk_token_id2)

            expanded_left_chunk = []
            expanded_right_chunk = []

            chunk_tiler(chunk_seg1, self.lc, expanded_left_chunk)
            chunk_tiler(chunk_seg2, self.rc, expanded_right_chunk)

            if expanded_left_chunk is None or expanded_right_chunk is None:
                continue

            for c1, c2 in product(expanded_left_chunk, expanded_right_chunk):
                chunk = ChunkAlignment(chunk_text, c1, c2)
                self.chunks.append(chunk)

class ChunkAlignment(object):
    """
    Representation of a chunk alignment 
    """
    def __init__(self, chunk_text, c1, c2):
        self.chunk_text = chunk_text
        self.c1 = c1
        self.c2 = c2
        self.idx = None
        self.form_chunks()
        
    def form_chunks(self):
        segments = self.chunk_text.split("//")
        self.type = segments[1].strip()
        self.chunk1 = self.c1
        self.chunk2 = self.c2
        self.score = segments[2].strip()


class DatasetReader(object):
    """
    Read the Interpretable sentence similarity dataset, 
    @param file_path_1: path of file with left sentences
    @param file_path_2: path of file with right sentences
    @param file_path_1_chunk: path of file with chunks from left sentence
    @param file_path_2_chunk: path of file with chunks from right sentence
    @param file_alignment: path of file with chunk level alignments
    """

    def __init__(self, file_path_1, file_path_2, file_path_1_chunk,
                 file_path_2_chunk, file_alignment):
        self.fp1 = file_path_1
        self.fp2 = file_path_2
        self.fp1_chunk = file_path_1_chunk
        self.fp2_chunk = file_path_2_chunk
        self.fp_align = file_alignment

        self.dataset = None
        self.left_chunks = None
        self.right_chunks = None
        self.alignments = None
        self.max_chunk_len = 0
        self.max_chunks = 0

    def __read_text(self, fpath):
        lines = None
        with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
            lines = fp.readlines()
            lines = [line.split("\n")[0].strip().lower() for line in lines]
        return lines

    def __read_chunks(self, fpath):
        chunks = None
        with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
            chunks = fp.readlines()
            chunks = [line.split("\n")[0].strip().lower() for line in chunks]
            chunks = [l[1:-1].split("] [") for l in chunks]
            chunks = [[e.strip() for e in l] for l in chunks]
            
            chunks = [[e.split() for ix,e in enumerate(l)] for l in chunks]
            chunks_with_id = []
            for chunk in chunks:
                chunk_with_id = []
                idx = 0
                for w in chunk:
                    wl = len(w)
                    w = [(w0, idx + i + 1) for i, w0 in enumerate(w)]
                    chunk_with_id.append(w)
                    idx += wl
                chunks_with_id.append(chunk_with_id)
            
            
            for chunk in chunks_with_id:
                if len(chunk) > self.max_chunks:
                    self.max_chunks = len(chunk)
                for e in chunk:
                    if len(e) > self.max_chunk_len:
                        self.max_chunk_len = len(e) * 5
        return chunks_with_id
        
    def read(self):
        left_lines = self.__read_text(self.fp1)
        right_lines = self.__read_text(self.fp2)
        self.left_chunks = self.__read_chunks(self.fp1_chunk)
        self.right_chunks = self.__read_chunks(self.fp2_chunk)
        
        self.dataset = zip(left_lines, right_lines)
        self.sentence_alignments = []
        
        # Read the alignments file as XML, this is not straightforward
        # some preprocessing is needed
        with codecs.open(self.fp_align, mode="rt", encoding="utf-8") as fp:
            alignments = fp.read()
            #This to escape failure of XML parsing
            alignments = alignments.replace("<==>", "==").replace("&", "&amp;")
            alignments = alignments.encode("ascii", errors="ignore")
            tree = ET.fromstring(alignments)
            sentences = tree.findall("sentence")
            for idx, sentence in enumerate(sentences):
                sentence_text = sentence.text.split("\n")[1:-1]
                s1, s2 = [s[2:].strip() for s in sentence_text]
                sentence_alignment = sentence.findall("alignment")[0].text.lower()
                lc = self.left_chunks[idx]
                rc = self.right_chunks[idx]
                sentence_alignment = SentenceAlignment(sentence_alignment, lc,
                                                       rc, s1, s2, idx+1)
                self.sentence_alignments.append(sentence_alignment)
        

def chunk_identifier(chunk, chunk_list):
    for idx, chunk_c in enumerate(chunk_list):
        if chunk == chunk_c: return idx
    return -1

class TrainerDatasetReader(TorchDataset):

    def __init__(self, dataset_reader, emb_reader, cfg):
        #assert isinstance(dataset_reader, DatasetReader), "dataset_reader must be an object of class DatasetReader"
        #assert isinstance(emb_reader, VB.EmbeddingReader), "emb_reader must be an instance of class EmbeddingReader"
        self.reader = dataset_reader
        self.emb_reader = emb_reader
        self.bert_tokenizer = self.emb_reader.bert_tokenizer
        chunk_emb_reader = EM.ChunkEmbedding(emb_reader)
        self.chunk_embedding = chunk_emb_reader.torch_chunk_embedding()
        self.emb_type = self.emb_reader.emb_type
        self.cfg = cfg

    def __len__(self):
        return len(self.reader.sentence_alignments)

    def _init_training(f):
        def wrapper(self, *args, **kwargs):
            if not wrapper.initialized:
                types = set()
                for s_idx, sentence_alignment in enumerate(self.reader.sentence_alignments):
                    chunk1_set,chunk2_set = set(), set()
                    aligned = set()

                    left_chunks = list(self.reader.left_chunks[s_idx])
                    if self.bert_tokenizer:
                        left_chunks = [[[(t, token_idx) for t in self.bert_tokenizer.tokenize(token)] for token,token_idx in chunk] for chunk in left_chunks]
                        left_chunks = [list(chain(*lc)) for lc in left_chunks]


                    left_chunks, left_chunk_ids = [zip(*lc)[0] for lc in left_chunks], [zip(*lc)[1] for lc in left_chunks]
                    left_chunks = [list(lc) for lc in left_chunks]
                    left_chunk_ids = [list(ids) for ids in left_chunk_ids]
                    
                    sentence_alignment.num_left_chunks = len(left_chunks)
                    sentence_alignment.left_chunk_ids = left_chunk_ids

                    for chunk in left_chunks:
                        chunk.extend(['-PAD-'] * (self.reader.max_chunk_len - len(chunk)))

                    right_chunks = list(self.reader.right_chunks[s_idx])
                    if self.bert_tokenizer:
                        right_chunks = [[[(t, token_idx) for t in self.bert_tokenizer.tokenize(token)] for token,token_idx in chunk] for chunk in right_chunks]
                        right_chunks = [list(chain(*lc)) for lc in right_chunks]


                    right_chunks, right_chunk_ids = [zip(*rc)[0] for rc in right_chunks], [zip(*rc)[1] for rc in right_chunks]

                    right_chunks = [list(rc) for rc in right_chunks]
                    right_chunk_ids = [list(ids) for ids in right_chunk_ids]
                    sentence_alignment.right_chunk_ids = right_chunk_ids
                    sentence_alignment.num_right_chunks = len(right_chunks)

                    for chunk in right_chunks:
                        chunk.extend(['-PAD-'] * (self.reader.max_chunk_len - len(chunk)))
                        
                    num_chunks_l, num_chunks_r = len(left_chunks), len(right_chunks)

                    pad_l = self.reader.max_chunks - num_chunks_l
                    pad_r = self.reader.max_chunks - num_chunks_r
                    
                    pad_l = [['-PAD-'] * self.reader.max_chunk_len] * pad_l
                    pad_r = [['-PAD-'] * self.reader.max_chunk_len] * pad_r
                    
                    left_chunks.extend(pad_l)
                    right_chunks.extend(pad_r)

                    s_id = str(s_idx + 1)
                    r_chunk_ids = [np.array2string(np.array(r), separator=",") for r in right_chunk_ids]
                    r_chunk_ids = [s_id + "_r_" + r[1:-1].strip() for r in r_chunk_ids]
                    right_chunk_emb_ids = self.emb_reader.vocab_to_index(r_chunk_ids)

                    l_chunk_ids = [np.array2string(np.array(l), separator=",") for l in left_chunk_ids]
                    l_chunk_ids = [s_id + "_l_" + l[1:-1].strip() for l in l_chunk_ids]
                    left_chunk_emb_ids = self.emb_reader.vocab_to_index(l_chunk_ids)

                    sentence_alignment.left_chunk_emb_ids = left_chunk_emb_ids
                    sentence_alignment.right_chunk_emb_ids = right_chunk_emb_ids

                    left_content_mask = [[STOP_WORDS.get(t, 1.0) for t in lc] for lc in left_chunks]
                    sentence_alignment.left_content_mask = left_content_mask

                    right_content_mask = [[STOP_WORDS.get(t, 1.0) for t in rc] for rc in right_chunks]
                    sentence_alignment.right_content_mask = right_content_mask
                    
                    sentence_alignment.left_chunks_tokenized = [self.emb_reader.vocab_to_index(tokens) for tokens in left_chunks]
                    sentence_alignment.right_chunks_tokenized = [self.emb_reader.vocab_to_index(tokens) for tokens in right_chunks]

                    for chunk in sentence_alignment.chunks:
                        types.add(chunk.type)
                        chunk1, chunk2 = list(chunk.chunk1), list(chunk.chunk2)
                        chunk.seq_len = [len(chunk1), len(chunk2)]

                        chunk1 = filter(lambda e : e != '-PAD-', chunk1)
                        chunk2 = filter(lambda e : e != '-PAD-', chunk2)

                        chunk1_idx = chunk_identifier(chunk1, self.reader.left_chunks[s_idx])
                        chunk2_idx = chunk_identifier(chunk2, self.reader.right_chunks[s_idx])
                        if chunk1_idx != -1 and chunk2_idx != -1:
                            chunk.idx = [chunk1_idx, chunk2_idx]
                            

                self.type_encoder = LabelEncoder()
                self.type_encoder.fit(list(types))
                wrapper.initialized = True

                # Load FOL resource(s)
                if self.cfg.constr_res_path:
                    with open(self.cfg.constr_res_path, 'r') as fp:
                        self.constr_res = json.load(fp)
            return f(self, *args, **kwargs)
        wrapper.initialized = False
        return wrapper


    def score_transform(self, score):
        if score == "NIL":
            return 0
        else:
            return int(score)

    @_init_training
    def __getitem__(self, item_id):
        aligned_input = self.reader.sentence_alignments[item_id]
        data_point = {}

        data_point["sidx"] = torch.tensor(aligned_input.sidx)
        data_point["left_chunks"] = torch.tensor(aligned_input.left_chunks_tokenized)
        data_point["right_chunks"] = torch.tensor(aligned_input.right_chunks_tokenized)
        data_point["left_chunk_emb_ids"] = torch.tensor(aligned_input.left_chunk_emb_ids)
        data_point["right_chunk_emb_ids"] = torch.tensor(aligned_input.right_chunk_emb_ids)

        data_point["left_chunk_ids"] = [torch.tensor(l) for l in aligned_input.left_chunk_ids]

        ll = [sum(l) for l,_ in zip(aligned_input.left_content_mask, aligned_input.left_chunk_ids)]
        ll.extend([1 for i in range(self.reader.max_chunks - len(ll))])
        ll = torch.tensor(ll).float()

        data_point["right_chunk_ids"] = [torch.tensor(l) for l in aligned_input.right_chunk_ids]

        data_point["left_content_mask"] = torch.tensor(aligned_input.left_content_mask)
        data_point["right_content_mask"] = torch.tensor(aligned_input.right_content_mask)
        
        rr = [sum(r) for r,_ in zip(aligned_input.right_content_mask, aligned_input.right_chunk_ids)]
        rr.extend([1 for i in range(self.reader.max_chunks - len(rr))])
        rr = torch.tensor(rr).float()
        
        data_point["num_left_chunks"] = torch.tensor(aligned_input.num_left_chunks)
        data_point["num_right_chunks"] = torch.tensor(aligned_input.num_right_chunks)

        lx,rx = data_point["left_chunks"],data_point["right_chunks"]
 
        left_l,right_l = data_point["num_left_chunks"],data_point["num_right_chunks"]
        lm = data_point["left_content_mask"]
        rm = data_point["right_content_mask"]

        if self.emb_type == "chunk":
            lx_chunk, rx_chunk = data_point["left_chunk_emb_ids"], data_point["right_chunk_emb_ids"]
        
            lx = self.chunk_embedding(lx)
            lx = torch.mean(lx, 1)

            rx = self.chunk_embedding(rx)
            rx = torch.mean(rx, 1)

            lx_chunk = self.chunk_embedding(lx_chunk)
            rx_chunk = self.chunk_embedding(rx_chunk)

            lx[:left_l] = lx_chunk
            rx[:right_l] = rx_chunk

        if self.emb_type == "glove":
            lx = self.chunk_embedding(lx)
            lx = torch.sum((lx * lm.unsqueeze(2)) , 1) / (torch.sum(lm, 1).unsqueeze(1) + 1e-15)
            # lx = torch.sum(lx, 1) / ll.unsqueeze(1)
            # lx = torch.mean(lx, 1)

            rx = self.chunk_embedding(rx)
            rx = torch.sum((rx * rm.unsqueeze(2)) , 1) / (torch.sum(rm, 1).unsqueeze(1) + 1e-15)
            
            # rx = torch.sum(rx, 1) / rr.unsqueeze(1)
            # rx = torch.mean(rx, 1)
 
        data_point["left_embedding"] = lx
        data_point["right_embedding"] = rx
        
        aligned = np.zeros(shape=(self.reader.max_chunks, self.reader.max_chunks))
        
        scores = np.zeros(shape=(self.reader.max_chunks, self.reader.max_chunks))
        scores[:] = -1

        types = np.zeros(shape=(self.reader.max_chunks, self.reader.max_chunks))
        types[:] = -1
        syn = SS.SyntacticSimilarity(aligned_input.s1, aligned_input.s2)

        if self.cfg.syn_scores:
            syn_scores = np.zeros(shape=(self.reader.max_chunks, self.reader.max_chunks))
            for ((ix1,sv1), (ix2,sv2)) in product(enumerate(aligned_input.left_chunk_ids), enumerate(aligned_input.right_chunk_ids)):
                sv1 = [i-1 for i in sv1]
                sv2 = [i-1 for i in sv2]
                syn_scores[ix1, ix2] = syn.compute_chunk_similarity(sv1, sv2)
            data_point["syn_scores"] = torch.tensor(syn_scores).float()

        for chunk in aligned_input.chunks:
            al_type = chunk.type
            al_score = chunk.score
            if chunk.idx:
                chunk.idx = tuple(chunk.idx)
                scores[chunk.idx] = self.score_transform(al_score)
                types[chunk.idx] = self.type_encoder.transform([al_type])[0]
                aligned[chunk.idx] = 1
            else:
                # print chunk.chunk1, chunk.chunk2
                # print chunk
                # print al_type
                pass
                
        data_point["is_aligned"] = torch.tensor(np.logical_not(np.any(aligned, axis = 1)).astype(np.int))
        data_point["aligned"] = torch.tensor(aligned)
        data_point["scores"] = torch.tensor(scores)
        data_point["types"] = torch.tensor(types)

        constr = self.constr_res.get(str(item_id+1)) # item_id+1 because resource self.reader.sentence_alignments is 0 indexed and constr_res 1 indexed
        data_point['constr'] = torch.tensor(constr) if constr else torch.Tensor()
        return data_point
    
