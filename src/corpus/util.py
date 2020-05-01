import spacy
import re

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