import json
import itertools
import requests
from time import sleep
import codecs
from copy import deepcopy
import spacy, re
import argparse

from corpus.util import process_sentence

chunk_regex = re.compile(r'''('s| %|--|["`',;:?$+%.&])( )?''')
spacy_nlp = spacy.load('en')

def process_result(i):
    o = set()
    for s in i:
        s = [x.lemma_.strip().lower() for x in spacy_nlp(s)]
        s = [chunk_regex.sub('', x) for x in s]
        s = '_'.join(x for x in s if x)
        o.add(s)
    return sorted(o)

def get_cn_rel(phrase, rel='Synonym'):
    assert rel in frozenset({'Synonym', 'Antonym', 'IsA', 'SimilarTo', 'RelatedTo', 'DistinctFrom', 'FormOf'})
    p = {
        'node': '/c/en/{}'.format(phrase),
        'rel': '/r/{}'.format(rel),
        'limit': '10000',
        'other': '/c/en'
    }
    obj = requests.get('http://api.conceptnet.io/search', params=p).json()['edges']
    
    rel_nodes = set([x['start']['label'] for x in obj])
    rel_nodes = rel_nodes.union(set([x['end']['label'] for x in obj]))
    if rel != 'Synonym':
        try:
            rel_nodes.remove(phrase)
        except:
            pass
    rel_nodes = sorted(rel_nodes) # to store as json later
    return rel_nodes

def create_resource(chunkset, rel='Synonym', saved_resource=None):
    chunkset = deepcopy(chunkset)
    print 'Chunkset size: {}'.format(len(chunkset))
    if not saved_resource:
        cn_rel = {}
    else:
        with open(saved_resource, 'r') as fp:
            cn_rel = json.load(fp)
        chunkset.difference_update(set(cn_rel.keys()))
        print 'Chunkset size after removing existing chunks: {}'.format(len(chunkset))
    
    counter = 1
    while chunkset:
        chunk = chunkset.pop()
        cn_rel[chunk] = get_cn_rel(chunk, rel)
        counter += 1
        if counter % 100 == 0:
            print 'Done:', counter
        sleep(1)
    return cn_rel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create conceptnet cache")
    parser.add_argument('-d', '--data_dir', default="../datasets/sts_16", help='path to STS 16 datasets directory', required=False)

    args = parser.parse_args()

    print 'Preparing chunkset'
    chunkset = set()
    for s1 in codecs.open('{}/train_2015_10_22.utf-8/STSint.input.headlines.sent1.chunk.txt'.format(args.data_dir), mode='rt', encoding='utf-8'):
        chunks, bigrams, unigrams = process_sentence(s1, lemmatize=True)
        chunkset.update(chunks, itertools.chain(*bigrams), itertools.chain(*unigrams))
    for s1 in codecs.open('{}/test/STSint.testinput.headlines.sent1.chunk.txt'.format(args.data_dir), mode='rt', encoding='utf-8'):
        chunks, bigrams, unigrams = process_sentence(s1, lemmatize=True)
        chunkset.update(chunks, itertools.chain(*bigrams), itertools.chain(*unigrams))

    for s2 in codecs.open('{}/train_2015_10_22.utf-8/STSint.input.headlines.sent2.chunk.txt'.format(args.data_dir), mode='rt', encoding='utf-8'):
        chunks, bigrams, unigrams = process_sentence(s2, lemmatize=True)
        chunkset.update(chunks, itertools.chain(*bigrams), itertools.chain(*unigrams))
    for s2 in codecs.open('{}/test/STSint.testinput.headlines.sent2.chunk.txt'.format(args.data_dir), mode='rt', encoding='utf-8'):
        chunks, bigrams, unigrams = process_sentence(s2, lemmatize=True)
        chunkset.update(chunks, itertools.chain(*bigrams), itertools.chain(*unigrams))



    chunkset.difference_update(set([x for x in chunkset if len(x) <= 1]))
    chunkset = {x.encode('ascii', 'ignore') for x in chunkset} # for EUR symbols


    rels = ['Synonym', 'Antonym', 'IsA', 'SimilarTo', 'RelatedTo', 'DistinctFrom', 'FormOf']
    # rels = []
    for rel in rels:
        print 'Creating resource for {}'.format(rel)
        
        cn_resource = create_resource(chunkset, rel)
        with open('{}/combined_cn_resource_{}.json'.format(args.data_dir,rel), 'w') as fp:
            json.dump(cn_resource, fp)

        # conceptnet resource canonicalization
        print 'Processing resource'
        for key in cn_resource:
                cn_resource[key] = process_result(cn_resource[key])
        
        with open('{}/combined_cn_resource_{}_processed.json'.format(args.data_dir,rel), 'w') as fp:
            json.dump(cn_resource, fp)
