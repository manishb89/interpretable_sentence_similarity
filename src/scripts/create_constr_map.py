from collections import defaultdict
import json, codecs, sys
import spacy

from corpus import corpus as CC
# from corpus.util import process_sentence

import re
import argparse

chunk_regex = re.compile(r'''('s| %|--|["`',;:?$+%.&])( )?''')
spacy_nlp = spacy.load('en')


def process_sentence(s, lemmatize=False):
    if lemmatize:
        s = ' '.join([x.lemma_ for x in spacy_nlp(s)]) #todo: check if lemmatization on non-chunked sentence is any better
    s = s.strip().lower()
    s = s[2:-2].split(' ] [ ')
    s = [chunk_regex.sub('', x) for x in s]
    chunks = [chk.replace(' ', '_').replace('-', '_') for chk in s]
    s = [chunk.replace('-', ' ').split() for chunk in s] #unigrams
    # s = [chunk.split() for chunk in s] #unigrams
    bigrams = [['_'.join(wrd) for wrd in zip(chk[:-1],chk[1:])] for chk in s]
    return (chunks, bigrams, s)

def check_dict(res, l, r):
    r_in_l = l in res and r in res[l]
    l_in_r = r in res and l in res[r]
    return r_in_l or l_in_r

def check_dict_any(cache, l, r):
    return any(check_dict(cache[res], l, r) for res in cache)

def get_content_mask(s, alignments):
    content_pos = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'NUM'}
    content_chunks = []
    content_words_mask = []

    s_pos = [x.pos_ for x in spacy_nlp(s)]

    for chk in alignments:
        content_chunks.append(s_pos[:len(chk)])
        s_pos = s_pos[len(chk):]
    
    for chk in content_chunks:
        content_words_mask.append([w in content_pos for w in chk])

    return content_words_mask

def create_constr_map(chunk1_path, chunk2_path, rel_res, match_type='chunk', content_words_mask=None):
    rel_matches = defaultdict(set)
    for sid, (s1, s2) in enumerate(zip(codecs.open(chunk1_path, mode='rt', encoding='utf-8'), codecs.open(chunk2_path, mode='rt', encoding='utf-8')), start=1):
        s1_chunks, s1_bigrams, s1_unigrams = process_sentence(s1, lemmatize=True)
        s2_chunks, s2_bigrams, s2_unigrams = process_sentence(s2, lemmatize=True)

        s1_chunks_, s1_bigrams_, s1_unigrams_ = process_sentence(s1, lemmatize=False)
        s2_chunks_, s2_bigrams_, s2_unigrams_ = process_sentence(s2, lemmatize=False)

        if content_words_mask:
            s1_content_mask = content_words_mask[sid][0]
            s2_content_mask = content_words_mask[sid][1]


        for ni, ci in enumerate(s1_chunks):
            bi = s1_bigrams[ni]
            wi = s1_unigrams[ni]

            wi_ = s1_unigrams_[ni]

            #tmp ignore 6 s1 chunks
            if len(wi) != len(wi_):
                continue
 
            for nj, cj in enumerate(s2_chunks):
                bj = s2_bigrams[nj]
                wj = s2_unigrams[nj]

                wj_ = s2_unigrams_[nj]
                
                #tmp ignore 8 s2 chunks
                if len(wj) != len(wj_):
                    continue

                # ignore exact chunk matches
                if s1_chunks_[ni] == s2_chunks_[nj]:
                    continue

                if match_type == 'chunk':
                    if check_dict(rel_res, ci, cj):
                            rel_matches[sid].add((ni,nj))
                        
                elif match_type == 'bigram':
                    if len(bi) >= 1 and len(bj) >= 1:
                        for bii in bi:
                            for bjj in bj:
                                if check_dict(rel_res, bii, bjj):
                                        rel_matches[sid].add((ni,nj))
                                    
                elif match_type == 'unigram':
                    if len(wi) >= 1 and len(wj) >= 1:
                        for nii, wii in enumerate(wi):
                            wii_ = wi_[nii]
                            for njj, wjj in enumerate(wj):
                                wjj_ = wj_[njj]
                                if check_dict(rel_res, wii, wjj):
                                    if content_words_mask:
                                        if s1_content_mask[ni][nii] and s2_content_mask[nj][njj]:
                                            rel_matches[sid].add((ni,nj))
                                    else:
                                        rel_matches[sid].add((ni,nj))
                else:
                    print('Wrong match type!')

    rel_matches = {k:sorted(v) for k,v in rel_matches.items()}
    return rel_matches


def create_constr_map_multiple(chunk1_path, chunk2_path, cache, match_type='chunk', content_words_mask=None):
    rel_matches = defaultdict(set)
    counter = 0
    for sid, (s1, s2) in enumerate(zip(codecs.open(chunk1_path, mode='rt', encoding='utf-8'), codecs.open(chunk2_path, mode='rt', encoding='utf-8')), start=1):
        s1_chunks, s1_bigrams, s1_unigrams = process_sentence(s1, lemmatize=True)
        s2_chunks, s2_bigrams, s2_unigrams = process_sentence(s2, lemmatize=True)

        s1_chunks_, s1_bigrams_, s1_unigrams_ = process_sentence(s1, lemmatize=False)
        s2_chunks_, s2_bigrams_, s2_unigrams_ = process_sentence(s2, lemmatize=False)

        if content_words_mask:
            s1_content_mask = content_words_mask[sid][0]
            s2_content_mask = content_words_mask[sid][1]


        for ni, ci in enumerate(s1_chunks):
            bi = s1_bigrams[ni]
            wi = s1_unigrams[ni]

            wi_ = s1_unigrams_[ni]

            #tmp ignore 6 s1 chunks
            if len(wi) != len(wi_):
                continue
 
            for nj, cj in enumerate(s2_chunks):
                bj = s2_bigrams[nj]
                wj = s2_unigrams[nj]

                wj_ = s2_unigrams_[nj]
                
                #tmp ignore 8 s2 chunks
                if len(wj) != len(wj_):
                    continue

                # ignore exact chunk matches
                if s1_chunks_[ni] == s2_chunks_[nj]:
                    continue

                if match_type == 'chunk':
                    if check_dict_any(cache, ci, cj):
                            rel_matches[sid].add((ni,nj))
                        
                elif match_type == 'bigram':
                    if len(bi) >= 1 and len(bj) >= 1:
                        for bii in bi:
                            for bjj in bj:
                                if check_dict_any(cache, bii, bjj):
                                        rel_matches[sid].add((ni,nj))
                                    
                elif match_type == 'unigram':
                    if len(wi) >= 1 and len(wj) >= 1:
                        for nii, wii in enumerate(wi):
                            wii_ = wi_[nii]
                            for njj, wjj in enumerate(wj):
                                wjj_ = wj_[njj]
                                if check_dict_any(cache, wii, wjj):
                                    try:
                                        if content_words_mask:
                                            if s1_content_mask[ni][nii] and s2_content_mask[nj][njj]:
                                                rel_matches[sid].add((ni,nj))
                                        else:
                                            rel_matches[sid].add((ni,nj))
                                    except Exception as e:
                                        # print sid, wi, ci, s1_content_mask
                                        # print sid, wj, cj, s2_content_mask
                                        # print str(e)
                                        # sys.exit()
                                        counter +=1

                else:
                    print('Wrong match type!')

    rel_matches = {k:sorted(list(x) for x in v) for k,v in rel_matches.items()}
    print '{} sentence pairs ignored'.format(counter)
    return rel_matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create constraint map of chunks")
    parser.add_argument('-d', '--data_dir', default="../datasets/sts_16", help='path to STS 16 datasets directory', required=False)

    args = parser.parse_args()
    
    relation_types = ['Synonym', 'Antonym', 'IsA', 'SimilarTo', 'RelatedTo', 'DistinctFrom', 'FormOf']
    match_types = ['unigram', 'bigram', 'chunk']

    
    cache = {}
    for rel in relation_types:
        cache_path = '{}/combined_cn_resource_{}_processed.json'.format(args.data_dir, rel)
        with open(cache_path, 'r') as fp:
            cache[rel] = json.load(fp)



    # TRAIN
    train_dir = '{}/train_2015_10_22.utf-8'.format(args.data_dir)
    chunk1_path = '{}/STSint.input.headlines.sent1.chunk.txt'.format(train_dir)
    chunk2_path = '{}/STSint.input.headlines.sent2.chunk.txt'.format(train_dir)
    fp1 = '{}/STSint.input.headlines.sent1.txt'.format(train_dir)
    fp2 = '{}/STSint.input.headlines.sent2.txt'.format(train_dir)
    align_file = '/STSint.input.headlines.wa'.format(train_dir)


    # content words
    reader = CC.DatasetReader(fp1, fp2, chunk1_path, chunk2_path, align_file)
    reader.read()

    cont_mask = {}
    for sid, (s1, s2) in enumerate(zip(codecs.open(fp1, mode='rt', encoding='utf-8'), codecs.open(fp2, mode='rt', encoding='utf-8')), start=1):
        sa = reader.sentence_alignments[sid-1] #sentence alignments is 0 indexed
        s1_content_mask = get_content_mask(s1, sa.lc)
        s2_content_mask = get_content_mask(s2, sa.rc)
        cont_mask[sid] = (s1_content_mask, s2_content_mask)
    print 'Created mask for content words\n'


    # res = {}
    # for rel in relation_types:
    #     print 'Creating constraint map for {}'.format(rel)
    #     cache_rel = {k:v for k,v in cache.items() if k == rel}
    #     res[rel] = {}

    #     for match_type in match_types:
    #         print 'Type: {}'.format(match_type)

    #         res[rel][match_type] = create_constr_map_multiple(chunk1_path, chunk2_path, cache_rel, match_type, cont_mask)

    # out_path = train_dir + '/STSint.input.headlines.fol.cn_all_resources_content_only.json'
    # with open(out_path, 'w') as fp:
    #     json.dump(res, fp)

    # Combined
    for match_type in match_types:
        comb_matches = create_constr_map_multiple(chunk1_path, chunk2_path, cache, match_type, content_words_mask=cont_mask)
        
        out_path = '{}/STSint.input.headlines.fol.cn_combined_{}_content_only.json'.format(train_dir, match_type)
        with open(out_path, 'w') as fp:
            json.dump(comb_matches, fp)




    # TEST
    test_dir = '{}/test'.format(args.data_dir)
    chunk1_path = '{}/STSint.testinput.headlines.sent1.chunk.txt'.format(test_dir)
    chunk2_path = '{}/STSint.testinput.headlines.sent2.chunk.txt'.format(test_dir)
    fp1 = '{}/STSint.testinput.headlines.sent1.txt'.format(test_dir)
    fp2 = '{}/STSint.testinput.headlines.sent2.txt'.format(test_dir)
    align_file = '{}/STSint.testinput.headlines.wa'.format(test_dir)

    
    # content words
    reader_test = CC.DatasetReader(fp1, fp2, chunk1_path, chunk2_path, align_file)
    reader_test.read()

    cont_mask_test = {}
    for sid, (s1, s2) in enumerate(zip(codecs.open(fp1, mode='rt', encoding='utf-8'), codecs.open(fp2, mode='rt', encoding='utf-8')), start=1):
        sa = reader_test.sentence_alignments[sid-1] #sentence alignments is 0 indexed
        s1_content_mask = get_content_mask(s1, sa.lc)
        s2_content_mask = get_content_mask(s2, sa.rc)
        cont_mask_test[sid] = (s1_content_mask, s2_content_mask)
    print 'Created mask for test set content words\n'


    # res = {}
    # for rel in relation_types:
    #     print 'Creating constraint map for {}'.format(rel)
    #     cache_rel = {k:v for k,v in cache.items() if k == rel}
    #     res[rel] = {}

    #     for match_type in match_types:
    #         print 'Type: {}'.format(match_type)

    #         res[rel][match_type] = create_constr_map_multiple(chunk1_path, chunk2_path, cache_rel, match_type, cont_mask_test)

    # out_path = test_dir + '/STSint.testinput.headlines.fol.cn_all_resources_content_only.json'
    # with open(out_path, 'w') as fp:
    #     json.dump(res, fp)

    #Combined
    for match_type in match_types:
        comb_matches = create_constr_map_multiple(chunk1_path, chunk2_path, cache, match_type, content_words_mask=cont_mask_test)
        
        out_path = '{}/STSint.testinput.headlines.fol.cn_combined_{}_content_only.json'.format(test_dir, match_type)
        with open(out_path, 'w') as fp:
            json.dump(comb_matches, fp)
