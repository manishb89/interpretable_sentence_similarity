import spacy
import numpy as np
from itertools import product

# Load the spacy english model
SPACY_NLP = spacy.load("en_core_web_sm")


def jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)

    if not s1 and not s2: return 0.0
    common = float(len(s1.intersection(s2)))
    union = float(len(s1.union(s2)))
    return common / union 
    
class TokenFeature(object):

    def __init__(self, tree_node):
        self.tree_node = tree_node

    def compute_feature(self):
        #pos = [self.tree_node.pos_]
        ancestors = [t.dep_ for t in self.tree_node.ancestors]
        children = [t.dep_ for t in self.tree_node.children]
        is_root = [self.tree_node.dep_ == 'ROOT']
        return [ancestors, children, is_root]
    
class SyntacticSimilarity(object):

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.d1 = SPACY_NLP(unicode(self.s1))
        self.d2 = SPACY_NLP(unicode(self.s2))

    def compute_token_similarity(self, t1_ix, t2_ix):
        if t1_ix >= len(self.d1) or t2_ix >= len(self.d2): return 0.0 
        t1 = self.d1[t1_ix]
        t2 = self.d2[t2_ix]

        t1_node = TokenFeature(t1)
        t2_node = TokenFeature(t2)

        fs1 = t1_node.compute_feature()
        fs2 = t2_node.compute_feature()

        total_sim = 0.0
        for f1, f2 in zip(fs1, fs2):
            sim = jaccard(f1, f2)
            total_sim += sim
        
        return total_sim / len(fs1)

    def compute_chunk_similarity(self, c1_ix, c2_ix):
        sim_matrix = np.zeros((len(c1_ix), len(c2_ix)))
        
        for ((i1,t1),(i2, t2)) in product(enumerate(c1_ix), enumerate(c2_ix)):
            sim = self.compute_token_similarity(t1, t2)
            sim_matrix[i1, i2] = sim

        #return sim_matrix
        return np.mean(np.max(sim_matrix, 1))

        

