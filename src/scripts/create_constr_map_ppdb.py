from collections import defaultdict
import json, codecs, sys
import spacy, re
import ppdb
from corpus import corpus as CC
# from corpus.util import process_sentence

chunk_regex = re.compile(r'''('s| %|--|["`',;:?$+%.&])( )?''')
spacy_nlp = spacy.load('en')


def process_sentence_ppdb(s, lemmatize=False):
    if lemmatize:
        s = ' '.join([x.lemma_ for x in spacy_nlp(s)]) #todo: check if lemmatization on non-chunked sentence is any better
    s = s.strip().lower()
    s = s[2:-2].split(' ] [ ')
    s = [chunk_regex.sub('', x) for x in s]
    unigrams = [chunk.split() for chunk in s]
    bigrams = [[' '.join(wrd) for wrd in zip(chk[:-1],chk[1:])] for chk in unigrams]
    return (s, bigrams, unigrams)

def check_ppdb(ppdb_rules, l, r):
    r_in_l = l in ppdb_rules and r in [' '.join(x) for x in ppdb_rules.get_rhs(l)]
    l_in_r = r in ppdb_rules and l in [' '.join(x) for x in ppdb_rules.get_rhs(r)]
    return l_in_r or r_in_l

def get_content_mask(s, alignments, debug=False):
    content_pos = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'NUM'}
    content_chunks = []
    content_words_mask = []

    s_pos = [x.pos_ for x in spacy_nlp(s)]

    for chk in alignments:
        content_chunks.append(s_pos[:len(chk)])
        s_pos = s_pos[len(chk):]
    
    for chk in content_chunks:
        content_words_mask.append([w in content_pos for w in chk])
    
    if debug:
        print s_pos
        print content_chunks

    return content_words_mask


def ppdb_constr_map(chunk1_path, chunk2_path, ppdb_rules, match_type='chunk', content_words_mask=None):
    rel_matches = defaultdict(set)
    for sid, (s1, s2) in enumerate(zip(open(chunk1_path), open(chunk2_path)), start=1):
        s1_chunks, s1_bigrams, s1_unigrams = process_sentence_ppdb(s1, lemmatize=False)
        s2_chunks, s2_bigrams, s2_unigrams = process_sentence_ppdb(s2, lemmatize=False)
        
        if content_words_mask:
            s1_content_mask = content_words_mask[sid][0]
            s2_content_mask = content_words_mask[sid][1]
        
        for ni, ci in enumerate(s1_chunks):
            bi = s1_bigrams[ni]
            wi = s1_unigrams[ni]
            for nj, cj in enumerate(s2_chunks):
                bj = s2_bigrams[nj]
                wj = s2_unigrams[nj]
                
                if ci == cj:
                    continue
                
                if match_type == 'chunk':
                    if check_ppdb(ppdb_rules, ci, cj):
                        rel_matches[sid].add((ni,nj))
                        
                elif match_type == 'bigram':
                    if len(bi) >= 1 and len(bj) >= 1:
                        for bii in bi:
                            for bjj in bj:
                                if check_ppdb(ppdb_rules, bii, bjj):
                                    rel_matches[sid].add((ni,nj))
                                    
                elif match_type == 'unigram':
                    if len(wi) >= 1 and len(wj) >= 1:
                        for nii, wii in enumerate(wi):
                            for njj, wjj in enumerate(wj):
                                if check_ppdb(ppdb_rules, wii, wjj):
                                    try:
                                        if content_words_mask:
                                            if s1_content_mask[ni][nii] and s2_content_mask[nj][njj]:
                                                rel_matches[sid].add((ni,nj))
                                        else:
                                            rel_matches[sid].add((ni,nj))
                                    except Exception as e:
                                        print sid, wi, ci, s1_content_mask
                                        print sid, wj, cj, s2_content_mask
                                        print str(e)
                else:
                    print('Wrong match type!')

    rel_matches = {k:sorted(list(x) for x in v) for k,v in rel_matches.items()}
    return rel_matches

if __name__ == '__main__':
    data_dir = '../datasets'
    content_only = True


    train_dir = '{}/sts_16/train_2015_10_22.utf-8'.format(data_dir)
    chunk1_path = '{}/STSint.input.headlines.sent1.chunk.txt'.format(train_dir)
    chunk2_path = '{}/STSint.input.headlines.sent2.chunk.txt'.format(train_dir)
    fp1 = train_dir + '/STSint.input.headlines.sent1.txt'
    fp2 = train_dir + '/STSint.input.headlines.sent2.txt'
    align_file = train_dir + '/STSint.input.headlines.wa'

    reader = CC.DatasetReader(fp1, fp2, chunk1_path, chunk2_path, align_file)
    reader.read()

    ppdb_rules = ppdb.load_ppdb('../datasets/ppdb-2.0-tldr', force=True)
    print 'PPDB loaded. Number of rules:', len(ppdb_rules)

    # TRAIN
    cont_mask = {}
    for sid, (s1, s2) in enumerate(zip(codecs.open(fp1, mode='rt', encoding='utf-8'), codecs.open(fp2, mode='rt', encoding='utf-8')), start=1):
        sa = reader.sentence_alignments[sid-1] #sentence alignments is 0 indexed
        s1_content_mask = get_content_mask(s1, sa.lc)
        s2_content_mask = get_content_mask(s2, sa.rc)
        cont_mask[sid] = (s1_content_mask, s2_content_mask)
    print '\nCreated mask for content words'

    ppdb_alignments = ppdb_constr_map(chunk1_path, 
                                      chunk2_path, 
                                      ppdb_rules, 
                                      'unigram', 
                                      content_words_mask=cont_mask if content_only else None)
    
    out_path = '{}/STSint.input.headlines.fol.ppdb_tldr_unigram_content_only.json'.format(train_dir)
    with open(out_path, 'w') as fp:
        json.dump(ppdb_alignments, fp)


    # TEST
    test_dir = '{}/sts_16/test'.format(data_dir)
    chunk1_path = '{}/STSint.testinput.headlines.sent1.chunk.txt'.format(test_dir)
    chunk2_path = '{}/STSint.testinput.headlines.sent2.chunk.txt'.format(test_dir)
    fp1 = test_dir + '/STSint.testinput.headlines.sent1.txt'
    fp2 = test_dir + '/STSint.testinput.headlines.sent2.txt'
    align_file = test_dir + '/STSint.testinput.headlines.wa'

    
    # content words
    reader_test = CC.DatasetReader(fp1, fp2, chunk1_path, chunk2_path, align_file)
    reader_test.read()

    
    cont_mask_test = {}
    for sid, (s1, s2) in enumerate(zip(codecs.open(fp1, mode='rt', encoding='utf-8'), codecs.open(fp2, mode='rt', encoding='utf-8')), start=1):
        sa = reader_test.sentence_alignments[sid-1] #sentence alignments is 0 indexed
        s1_content_mask = get_content_mask(s1, sa.lc)
        s2_content_mask = get_content_mask(s2, sa.rc)
        cont_mask_test[sid] = (s1_content_mask, s2_content_mask)
    print '\nCreated mask for content words'

    ppdb_alignments = ppdb_constr_map(chunk1_path, 
                                      chunk2_path, 
                                      ppdb_rules, 
                                      'unigram', 
                                      content_words_mask=cont_mask_test if content_only else None)
    
    out_path = '{}/STSint.testinput.headlines.fol.ppdb_tldr_unigram_content_only.json'.format(test_dir)
    with open(out_path, 'w') as fp:
        json.dump(ppdb_alignments, fp)
