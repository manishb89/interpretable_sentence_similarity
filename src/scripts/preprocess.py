import sys
import spacy
import argparse
import codecs

spacy_nlp = spacy.load('en_core_web_sm')

def process_spacy(all_sentences):
	# all_sent1_lemma = [[process_spacy(chunk) for chunk in sent1] for sent1 in all_sent1]
	for sentence in all_sentences:
		for chunk in sentence:
			tokenized = spacy_nlp(chunk)
			lemma = [tok.lemma_.replace(' ','') for tok in tokenized if not tok.is_space]

	return lemma


def read_text(fpath):
        lines = None
        with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
            lines = fp.readlines()
            lines = [line.split("\n")[0] for line in lines]
        return lines


def read_chunks(fpath):
    chunks = None
    with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
        chunks = fp.readlines()
        chunks = [line.split("\n")[0].strip() for line in chunks]
        chunks = [l[1:-1].split("] [") for l in chunks]
        chunks = [[e.strip().decode() for e in l] for l in chunks]
        
        # chunks = [[e.split() for ix,e in enumerate(l)] for l in chunks]
    return chunks


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sent1', help="Path to sent1 file", default='../../datasets/sts_16/train_2015_10_22.utf-8/STSint.input.headlines.sent1.chunk.txt')
parser.add_argument('--sent2', help="Path to sent2 file", default='../../datasets/sts_16/train_2015_10_22.utf-8/STSint.input.headlines.sent2.chunk.txt')
parser.add_argument('--out1', help="Path to sent1 lemmatized output", default='../../datasets/sts_16/train_2015_10_22.utf-8/STSint.input.headlines.sent1.chunk.lemma.txt')
parser.add_argument('--out2', help="Path to sent2 lemmatized output", default='../../datasets/sts_16/train_2015_10_22.utf-8/STSint.input.headlines.sent2.chunk.lemma.txt')


def main(args):
	opt = parser.parse_args(args)
	all_sent1 = read_chunks(opt.sent1)
	all_sent2 = read_chunks(opt.sent2)
	
	all_sent1_lemma = [[process_spacy(chunk) for chunk in sent1] for sent1 in all_sent1]
	all_sent2_lemma = [[process_spacy(chunk) for chunk in sent2] for sent2 in all_sent2]

	all_sent1_lemma = [[' '.join(chunk) for chunk in sent] for sent in all_sent1_lemma]
	all_sent1_lemma = [' ] [ '.join(sent) for sent in all_sent1_lemma]
	all_sent1_lemma = ['[ ' + sent + ' ]' for sent in all_sent1_lemma]

	all_sent2_lemma = [[' '.join(chunk) for chunk in sent] for sent in all_sent2_lemma]
	all_sent2_lemma = [' ] [ '.join(sent) for sent in all_sent2_lemma]
	all_sent2_lemma = ['[ ' + sent + ' ]' for sent in all_sent2_lemma]
	# print all_sent1_lemma[:5]

	with codecs.open(opt.out1, 'w+', 'utf-8') as f1, codecs.open(opt.out2, 'w+', 'utf-8') as f2:
		f1.writelines([sent1	 + '\n' for sent1	 in all_sent1_lemma])
		f2.writelines([sent2 + '\n' for sent2 in all_sent2_lemma])

if __name__ == '__main__':
	main(sys.argv[1:])