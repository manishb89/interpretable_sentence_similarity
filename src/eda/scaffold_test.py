from __future__ import division
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree as NLTKTree
from tqdm import tqdm
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
import pprint
import codecs

core_nlp_server_path = "http://localhost"
core_nlp_server_port = 9000

class ScaffoldingCheck(object):
    """
    Scaffolding check for interpretable sentence similarity 
    with constituency parse trees
    """
    def __init__(self):
        self.nlp = StanfordCoreNLP(core_nlp_server_path, port=core_nlp_server_port)

    def check_chunk_alignment(self, s1, s2, chunks_alignment):
        """
        @param s1: sentence to chunk alignment 
        @param s2: sentence to chunk alignment
        @param chunks_alignment: chunk alignment object for the sentence pair (s1, s2)
        """
        # The CoreNLP library (or its python wrapper) misbehaves with utf-8
        # by passing by encoding to ascii for now, would revisit 

        s1 = s1.encode("ascii", errors = "ignore")
        parsed_s1 = self.nlp.parse(s1)
        parse_t1 = NLTKTree.fromstring(parsed_s1)

        s2 = s2.encode("ascii", errors = "ignore")
        parsed_s2 = self.nlp.parse(s2)
        parse_t2 = NLTKTree.fromstring(parsed_s2)

        s1_indx_chunk, s2_indx_chunk = {}, {}
        for t in parse_t1.subtrees():
            flat_tree = t.flatten()
            s1_indx_chunk[tuple(flat_tree.leaves())] = flat_tree

        for t in parse_t2.subtrees():
            flat_tree = t.flatten()
            s2_indx_chunk[tuple(flat_tree.leaves())] = flat_tree

        chunks = chunks_alignment.chunks
        common_subtrees = []
        uncommon_subtrees = []

        # Ignore the No Alignment chunks
        chunks = [chunk for chunk in chunks if chunk.type != "NOALI"]
        num_chunks = len(chunks)

        if num_chunks == 0: return [], [], 1.0
        
        for chunk in chunks:
            c1, c2 = tuple(chunk.chunk1.split()), tuple(chunk.chunk2.split())
            if c1 in s1_indx_chunk and c2 in s2_indx_chunk:
                t1 = s1_indx_chunk[c1]
                t2 = s2_indx_chunk[c2]
                l1,l2 = t1.label(), t2.label()

                #Subtree root label match at the top levele e.g P, N, V 
                l1_top, l2_top = l1[0], l2[0]
                if l1_top == l2_top: common_subtrees.append((t1, t2))
                else: uncommon_subtrees.append((t1, t2))

        return common_subtrees, uncommon_subtrees, len(common_subtrees) / num_chunks
    
        
    def check_constituency(self, sentence, chunks):
        """
        @param sentence: sentence to check constituency
        @param chunks: human labeled chunks of the sentence
        """
        # The CoreNLP library (or its python wrapper) misbehaves with utf-8
        # by passing by encoding to ascii for now, would revisit 

        sentence = sentence.encode("ascii", errors="ignore")
        parsed_sentence = self.nlp.parse(sentence)
        parse_tree = NLTKTree.fromstring(parsed_sentence)

        subtree_as_leafs = [t.flatten().leaves() for t in parse_tree.subtrees()]
        chunk_total = len(chunks)
        matched_chunks = 0
        
        for chunk in chunks:
            if chunk in subtree_as_leafs:
                matched_chunks += 1

        return matched_chunks / chunk_total



# Utility Methods 
def compute_constituency_agreement(fp1, fp2, fp1_chunk, fp2_chunk, align_file):
    reader = DatasetReader(fp1, fp2, fp1_chunk, fp2_chunk, align_file)
    reader.read()
    scaffold_chck = ScaffoldingCheck()

    overall_alignment_left,overall_alignment_right = 0,0
    enm_dataset = list(enumerate(reader.dataset))

    try:
        for i, (l, r) in tqdm(enm_dataset):
            l_c = reader.left_chunks[i]
            r_c = reader.right_chunks[i]
            alignment_fraction_left = scaffold_chck.check_constituency(l, l_c)
            alignment_fraction_right = scaffold_chck.check_constituency(r, r_c)

            overall_alignment_left += alignment_fraction_left
            overall_alignment_right += alignment_fraction_right

        print "Total Left Alignment {}".format(overall_alignment_left / len(reader.dataset))
        print "Total Right Alignment {}".format(overall_alignment_right / len(reader.dataset))
        print "Overall Alignment {}".format((overall_alignment_left + overall_alignment_right) / (2*len(reader.dataset)))
    finally:
        scaffold_chck.nlp.close()
    
def compute_alignment_agreement(fp1, fp2, fp1_chunk, fp2_chunk, align_file):
    reader = DatasetReader(fp1, fp2, fp1_chunk, fp2_chunk, align_file)
    reader.read()
    scaffold_chck = ScaffoldingCheck()
    overall_alignment = 0
    total_unaligned_chunks = 0
    unaligned = []
    
    enm_dataset = list(enumerate(reader.dataset))
    try:
        for i,(l, r) in tqdm(enm_dataset):
            alignment = reader.sentence_alignments[i]
            trees, unc, score = scaffold_chck.check_chunk_alignment(l, r, alignment)
            overall_alignment += score
            total_unaligned_chunks += len(unc)
            if len(unc) > 0: unaligned.append(i)

        print "Total Alignment Agreement {}".format(overall_alignment / len(reader.dataset))
        print "Total Number of Unaligned Chunks {}".format(total_unaligned_chunks)
        return unaligned
    finally:
        scaffold_chck.nlp.close()
