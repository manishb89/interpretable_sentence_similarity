import json, sys
from copy import deepcopy
import itertools

from corpus import corpus as CC

class AlignmentEval():
    def __init__(self, reader):
        self.reader = reader
        self.non_aligned_sids = []
        self.gold_empty_sid = []
        self.nonali = []
        self.gold_alignments, self.gold_alignment_types, self.chunk_not_match_mask = self.get_gold_ali()
    
    def get_gold_ali(self, debug=False):
        chunk_alignments = {}
        alignment_types = {}
        chunk_not_match_mask = {}
        # sentence ids start from 1 and sentence_alignments is 0 indexed
        nested_chunk_sids = set()
        for sid, sa in enumerate(self.reader.sentence_alignments, start=1):
            if sa.chunks == []:
                self.non_aligned_sids.append(sid)
                chunk_alignments[sid] = []
                continue
            try:
                chunk_idx_l = []
                chunk_idx_r = []
                chunks1 = [chk.c1 for chk in sa.chunks]
                chunks2 = [chk.c2 for chk in sa.chunks]

                for chk in chunks1:
                    if len(chk) == 1 and isinstance(chk[0], list): # because [[('convicted', 4), ('nazi', 5), ('death', 6), ('camp', 7), ('guard', 8)]],
                        chunk_idx_l.append(sa.lc.index(chk[0]))
                        nested_chunk_sids.add(sid)
                    else:
                        chunk_idx_l.append(sa.lc.index(chk))

                for chk in chunks2:
                    if len(chk) == 1 and isinstance(chk[0], list): # because [[('convicted', 4), ('nazi', 5), ('death', 6), ('camp', 7), ('guard', 8)]],
                        chunk_idx_r.append(sa.rc.index(chk[0]))
                        nested_chunk_sids.add(sid)
                    else:
                        chunk_idx_r.append(sa.rc.index(chk))

            except:
                if debug:
                    print 'ERROR getting gold alignments for sid',sid

            chunk_alignments[sid] = [list(x) for x in zip(chunk_idx_l, chunk_idx_r)]
            # chunk_types is incorrect for the same 9 sentences where gold aligmnents are empty (parse issue in corpus.py)
            alignment_types[sid] = [chk.type.split('_')[0] for chk in sa.chunks]
            # True for matching chunks
            chunk_words1 = [[w[0] for w in chk] for chk in chunks1]
            chunk_words2 = [[w[0] for w in chk] for chk in chunks2]
            chunk_not_match_mask[sid] = [chunk_words1[i] != chunk_words2[i] for i in xrange(len(chunk_words1))]
        
        if debug:
            print '\nnested_chunk_sids: ', nested_chunk_sids

        return chunk_alignments, alignment_types, chunk_not_match_mask

    
    def calc_coverage(self, cn_alignments, rel_set=None, ignore_exact_chunk_match=True, debug=False):
        correct_alignments = 0
        incorrect_alignments = 0
        missing_alignments = 0

        correct_alignments_rel = 0
        incorrect_alignments_rel = 0
        missing_alignments_rel = 0

        self.nonali = []
        
        for sid in range(1, len(self.gold_alignments) + 1):
            gold_alignment = deepcopy(self.gold_alignments[sid])

            if len(gold_alignment) == 0:
                if sid not in self.non_aligned_sids:
                    self.gold_empty_sid.append(sid)
                continue
            
            
            if rel_set:
                gold_alignment_type = deepcopy(self.gold_alignment_types[sid])

            if ignore_exact_chunk_match:
                gold_alignment = list(itertools.compress(gold_alignment, self.chunk_not_match_mask[sid]))
                if rel_set:
                    gold_alignment_type = list(itertools.compress(gold_alignment_type, self.chunk_not_match_mask[sid]))

            
            # filter out alignments of other types in gold alignments, if specified
            if rel_set:
                gold_alignment_rel =  [ali for i, ali in enumerate(gold_alignment) if gold_alignment_type[i] in rel_set]
            else:
                gold_alignment_rel = deepcopy(gold_alignment)

            if sid in cn_alignments:
                cn_alignment = deepcopy(cn_alignments[sid])

                while cn_alignment:
                    ali_cn = cn_alignment.pop()
                    
                    if ali_cn in gold_alignment:
                        correct_alignments += 1
                        gold_alignment.remove(ali_cn)
                    else:
                        incorrect_alignments += 1
                        # print sid, ali_cn
                        # print self.gold_alignments[sid]
                        # print self.chunk_not_match_mask[sid]
                        # print

                    if ali_cn in gold_alignment_rel:
                        correct_alignments_rel += 1
                        gold_alignment_rel.remove(ali_cn)
                    else:
                        incorrect_alignments_rel += 1

                missing_alignments += len(gold_alignment)
                missing_alignments_rel += len(gold_alignment_rel)
            else:
                missing_alignments += len(gold_alignment)
                missing_alignments_rel += len(gold_alignment_rel)
                self.nonali.append(sid)
        if debug:
            print 'Cannnot calc gold_alignment for {} sentences (another {} do not have alignments)'.format(len(self.gold_empty_sid), len(self.non_aligned_sids))
            print 'sids: {}'.format(self.gold_empty_sid)
            print 'ignoring these for coverage calculation'
                

        # print correct_alignments, incorrect_alignments, missing_alignments
        p = round(1.0*correct_alignments/(correct_alignments+incorrect_alignments+1e-10),3)
        r = round(1.0*correct_alignments/(correct_alignments+missing_alignments+1e-10),3)

        p_rel = round(1.0*correct_alignments_rel/(correct_alignments_rel+incorrect_alignments_rel+1e-10),3)
        r_rel = round(1.0*correct_alignments_rel/(correct_alignments_rel+missing_alignments_rel+1e-10),3)

        print 'Precision: {}, Recall: {}'.format(p, r)
        print 'Correct: {}, Incorrect: {}, Missing: {}'.format(correct_alignments,incorrect_alignments, missing_alignments)

        if rel_set:
            print '\nRelation {} specific metrics:'.format(str(rel_set))
            print 'Precision: {}, Recall: {}'.format(p_rel, r_rel)
            print 'Correct: {}, Incorrect: {}, Missing: {}, Total: {}'.format(correct_alignments_rel,incorrect_alignments_rel, missing_alignments_rel,sum((correct_alignments_rel, missing_alignments_rel)))


if __name__ == '__main__':

    train_path = '../datasets/sts_16/train_2015_10_22.utf-8/'
    fp1 = train_path + 'STSint.input.headlines.sent1.txt'
    fp2 = train_path + 'STSint.input.headlines.sent2.txt'
    fp1_chunk = train_path + 'STSint.input.headlines.sent1.chunk.txt'
    fp2_chunk = train_path + 'STSint.input.headlines.sent2.chunk.txt'
    align_file = train_path + 'STSint.input.headlines.wa'

    rel_map = {
        'Synonym': {'equi'},
        'Antonym': {'oppo'},
        'IsA': {'spe1', 'spe2'},
        'SimilarTo': {'simi'},
        'RelatedTo': {'rel'}
    }

    relation_types = ['Synonym', 'Antonym', 'IsA', 'SimilarTo', 'RelatedTo', 'DistinctFrom', 'FormOf']
    match_types = ['chunk', 'bigram', 'unigram']

    print 'Loading and processing resources'
    cn_align_path = train_path + 'STSint.input.headlines.fol.cn_all_resources.json'
    with open(cn_align_path) as fp:
        cn_alignments = json.load(fp)
    for rel in relation_types:
        for match_type in match_types:
            cn_alignments[rel][match_type] = {int(k):v for k,v in cn_alignments[rel][match_type].items()}

    reader = CC.DatasetReader(fp1, fp2, fp1_chunk, fp2_chunk, align_file)
    reader.read()
    a = AlignmentEval(reader)

    # # CN Individial Relations
    # for rel in relation_types:
    #     print '\nRel: {}'.format(rel)
        
    #     for match_type in match_types:
            
    #         print '\nMatch Type: {}'.format(match_type)
    #         a.calc_coverage(cn_alignments[rel][match_type], rel_set=rel_map[rel])

    #     print '\n'

    # # CN Relations Combined
    # print 'Rel: Combined'
    # for match_type in match_types:
    #     print '\nMatch Type: {}'.format(match_type)
    #     cn_align_path = train_path + 'STSint.input.headlines.fol.cn_combined_{}_content_only.json'.format(match_type)
    #     with open(cn_align_path) as fp:
    #         cn_alignments = json.load(fp)
    #         cn_alignments = {int(k):v for k,v in cn_alignments.items()}
                    
    #     a.calc_coverage(cn_alignments, ignore_exact_chunk_match=True)

    # PPDB
    print 'Rel/Match Type: PPDB'
    cn_align_path = train_path + 'STSint.input.headlines.fol.ppdb_tldr_unigram_content_only.json'
    with open(cn_align_path) as fp:
        ppdb_alignments = json.load(fp)
        ppdb_alignments = {int(k):v for k,v in ppdb_alignments.items()}
                
    a.calc_coverage(ppdb_alignments, ignore_exact_chunk_match=True)
        