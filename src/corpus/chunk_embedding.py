import codecs
import xml.etree.ElementTree as ET
from gensim.models import KeyedVectors
import torch
from itertools import product, chain
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
import datetime
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors
import argparse

never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return [t.lower() if t not in never_split else t for t in tokens]


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


def __read_chunks(fpath):
    with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
        chunks = fp.readlines()
        chunks = [line.split("\n")[0].strip().lower() for line in chunks]
        chunks = [l[1:-1].split("] [") for l in chunks]
        chunks = [[e.strip() for e in l] for l in chunks]

        chunks = [[e.split() for ix, e in enumerate(l)] for l in chunks]
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
    return chunks_with_id


def __read_alignments(fpath, left_chunks, right_chunks):
    # Read the alignments file as XML, this is not straightforward
    # some preprocessing is needed
    overall_alignments = {}
    with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
        alignments = fp.read()
        # This to escape failure of XML parsing
        alignments = alignments.replace("<==>", "==").replace("&", "&amp;")
        alignments = alignments.encode("ascii", errors="ignore")
        tree = ET.fromstring(alignments)
        sentences = tree.findall("sentence")
        total_sentence_counter = 0
        skip_counter = 0
        for idx, sentence in enumerate(sentences):
            left_sentence = (sentence.text.split("\n")[1].split("//")[1]).strip()
            right_sentence = (sentence.text.split("\n")[2].split("//")[1]).strip()

            total_sentence_counter += 1
            l_chunks = left_chunks[idx]
            r_chunks = right_chunks[idx]

            sid = sentence.attrib['id']

            left_chunk_emb_ids = set()
            right_chunk_emb_ids = set()
            chunk_emb_mapping = {}

            sentence_info = {}
            alignment_text = sentence.findall("alignment")[0].text.lower()
            chunks_text = alignment_text.split("\n")
            for chunk_text in chunks_text:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                if chunk_text.split("//")[1] == " noali " or chunk_text.split("//")[1] == " noali_fact ":
                    continue

                chunk_token_id1, chunk_token_id2 = chunk_text.split("//")[0].split("==")

                left_chunk_ids = [int(idx) for idx in chunk_token_id1.split()]
                right_chunk_ids = [int(idx) for idx in chunk_token_id2.split()]

                left_text, right_text = chunk_text.split("//")[-1].split("==")
                left_text, right_text = left_text.strip(), right_text.strip()

                left_seg = zip(left_text.split(), left_chunk_ids)
                right_seg = zip(right_text.split(), right_chunk_ids)

                expanded_left_chunk = []
                expanded_right_chunk = []

                chunk_tiler(left_seg, l_chunks, expanded_left_chunk)
                chunk_tiler(right_seg, r_chunks, expanded_right_chunk)

                if expanded_left_chunk is None or expanded_right_chunk is None:
                    print "Skipped Sentence with id : ", sid
                    continue

                expanded_left_chunk_ids = []
                expanded_right_chunk_ids = []

                for c1, c2 in product(expanded_left_chunk, expanded_right_chunk):
                    l_chunk_ids = [str(lid) for lw, lid in c1]
                    r_chunk_ids = [str(rid) for rw, rid in c2]
                    l_chunk_emb_identifier = sid + "_" + "l" + "_" + ",".join(l_chunk_ids)
                    r_chunk_emb_identifier = sid + "_" + "r" + "_" + ",".join(r_chunk_ids)

                    if l_chunk_emb_identifier in chunk_emb_mapping:
                        r_emb_set = chunk_emb_mapping[l_chunk_emb_identifier]
                        r_emb_set.add(r_chunk_emb_identifier)
                        chunk_emb_mapping[l_chunk_emb_identifier] = r_emb_set
                    else:
                        chunk_emb_mapping[l_chunk_emb_identifier] = {r_chunk_emb_identifier}

                    expanded_left_chunk_ids.append(l_chunk_emb_identifier)
                    expanded_right_chunk_ids.append(r_chunk_emb_identifier)

                left_chunk_emb_ids.update(expanded_left_chunk_ids)
                right_chunk_emb_ids.update(expanded_right_chunk_ids)

            if len(left_chunk_emb_ids) == 0 or len(right_chunk_emb_ids) == 0:
                skip_counter += 1
                continue

            sentence_info['left_chunk_emb_ids'] = left_chunk_emb_ids
            sentence_info['right_chunk_emb_ids'] = right_chunk_emb_ids
            sentence_info['chunk_emb_mapping'] = chunk_emb_mapping
            sentence_info['left_sentence'] = left_sentence
            sentence_info['right_sentence'] = right_sentence

            overall_alignments[sid] = sentence_info
    print "Out of {} sentences, {} are skipped".format(str(total_sentence_counter), str(skip_counter))
    return overall_alignments


def get_bert_sentence_embedding(sentence, bert_tokenizer, bert_model):
    marked_text = "[CLS] " + sentence + " [SEP]"

    token_sub_word_mapping = {}
    tokenized_text = []
    basic_tokens = whitespace_tokenize(marked_text)
    for token in basic_tokens:
        split_tokens = []
        for sub_token in bert_tokenizer.wordpiece_tokenizer.tokenize(token):
            split_tokens.append(sub_token)
            tokenized_text.append(sub_token)
        token_sub_word_mapping[token] = split_tokens

    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
        batch_i = 0
        token_embeddings = []

        # For each token in the sentence...
        for token_i in range(len(tokenized_text)):

            # Holds 12 layers of hidden states for each token
            hidden_layers = []

            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]

                hidden_layers.append(vec)

            token_embeddings.append(hidden_layers)

        sub_word_embeddings = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]

        final_token_embeddings = []
        idx_counter = 0

        for token in basic_tokens:
            sub_words = token_sub_word_mapping[token]

            if len(sub_words) == 1:
                final_token_embeddings.append(sub_word_embeddings[idx_counter])
                idx_counter += 1
            else:
                sub_words_emb_list = []
                for i in range(len(sub_words)):
                    sub_words_emb_list.append(sub_word_embeddings[idx_counter + i])
                final_token_embeddings.append(torch.mean(torch.stack(sub_words_emb_list), dim=0))
                idx_counter += len(sub_words)

    return basic_tokens, final_token_embeddings


def get_embeddings(embedding, chunk_emb_ids, bert=False):
    if not bert:
        temp_list = []
        for emb_id in chunk_emb_ids:
            temp_list.append(torch.Tensor(embedding.vocab[emb_id]))
        return torch.stack(temp_list, dim=0)


def get_bert_chunks_embedding(tokenized_text, sentence_embedding, chunk_emb_ids, actual_chunk_tokens):
    tokens_from_chunk = [w for w, idx in list(chain(*actual_chunk_tokens))]
    emb_vector_list = []
    for emb_id in chunk_emb_ids:
        clean_emb_id = emb_id.split("_")[2]
        clean_emb_id = map(int, clean_emb_id.split(","))
        start_index, end_index = clean_emb_id[0], clean_emb_id[-1]
        start_token, end_token = tokens_from_chunk[start_index - 1], tokens_from_chunk[end_index - 1]
        start_bert_token, end_bert_token = tokenized_text[start_index], tokenized_text[end_index]
        if start_token != start_bert_token or end_token != end_bert_token:
            print "Something is wrong somewhere !!", chunk_emb_ids, tokenized_text, tokens_from_chunk

        start_chunk_embedding = sentence_embedding[start_index]
        end_chunk_embedding = sentence_embedding[end_index]
        chunk_embedding = torch.cat((start_chunk_embedding, end_chunk_embedding), 0)
        emb_vector_list.append(chunk_embedding)
    return torch.stack(emb_vector_list, dim=0)


def vocab_to_index(seq, stoi, unk_token_idx):
    indices = []
    for s in seq:
        idx = stoi.get(s, unk_token_idx)
        indices.append(idx)
    return indices


def get_glove_embeddings(embedding, chunk_emb_ids, stoi, chunk_tokens, unk_token_idx):
    tokens_from_chunk = [w for w, idx in list(chain(*chunk_tokens))]
    vocab_indices = vocab_to_index(tokens_from_chunk, stoi, unk_token_idx)

    emb_vector_list = []

    for emb_id in chunk_emb_ids:
        clean_emb_id = emb_id.split("_")[2]
        clean_emb_id = map(int, clean_emb_id.split(","))
        chunk_vocab_indices = [vocab_indices[i - 1] for i in clean_emb_id]

        temp_list = []

        for idx in chunk_vocab_indices:
            temp_list.append(torch.Tensor(embedding[idx]))

        emb_vector_list.append(torch.mean(torch.stack(temp_list, dim=0), dim=0))

    return torch.stack(emb_vector_list, dim=0)


def _write_vectors(vectors, output_file):
    emb_vectors = Word2VecKeyedVectors(vector_size=1536)
    emb_vectors.vocab = vectors
    emb_vectors.vectors = np.array(vectors.values())
    emb_vectors.save(output_file)


def unwrap_emb(chunk_emb_ids, emb_matrix, vectors):
    for i in range(len(chunk_emb_ids)):
        vectors[chunk_emb_ids[i]] = emb_matrix[i].numpy().ravel()


def chunk_cosine_sim(gold_alignments, embedding_file, left_chunks_file, right_chunks_file, bert_tokenizer=None,
                     bert_model=None, emb_type="chunk", output_file=None):
    left_chunks = __read_chunks(left_chunks_file)
    right_chunks = __read_chunks(right_chunks_file)

    sentence_alignments = __read_alignments(gold_alignments, left_chunks, right_chunks)

    if emb_type == "glove":
        tmp_file = get_tmpfile("_temp_w2v_{}.txt".format(datetime.datetime.now().strftime('%H_%M_%S')))
        _ = glove2word2vec(embedding_file, tmp_file)
        embedding = KeyedVectors.load_word2vec_format(tmp_file)
        stoi = {w: i for i, w in enumerate(embedding.index2word)}
        itos = {i: w for i, w in enumerate(embedding.index2word)}
        num_words = len(embedding.index2word)
        unk_token_idx = num_words
        stoi["-UNK-"] = unk_token_idx
        itos[unk_token_idx] = "-UNK-"
        emb_matrix = embedding.vectors.copy()
        unk_symbol_emb = np.zeros((1, embedding.vector_size))
        emb_matrix = np.concatenate((emb_matrix, unk_symbol_emb), axis=0)

    total_alignments_counter = 0
    match_counter = 0
    vectors = {}

    for sid, emb_ids_map in tqdm(sentence_alignments.items()):
        try:
            left_chunk_emb_ids = list(emb_ids_map['left_chunk_emb_ids'])
            right_chunk_emb_ids = list(emb_ids_map['right_chunk_emb_ids'])

            if emb_type == "bert":
                # get bert sentence level embedding

                left_sentence = emb_ids_map['left_sentence']
                tokenized_left_text, left_sentence_embedding = get_bert_sentence_embedding(left_sentence,
                                                                                           bert_tokenizer, bert_model)
                right_sentence = emb_ids_map['right_sentence']
                tokenized_right_text, right_sentence_embedding = get_bert_sentence_embedding(right_sentence,
                                                                                             bert_tokenizer, bert_model)
                left_matrix = get_bert_chunks_embedding(tokenized_left_text, left_sentence_embedding,
                                                        left_chunk_emb_ids, left_chunks[int(sid) - 1])
                unwrap_emb(left_chunk_emb_ids, left_matrix, vectors)

                right_matrix = get_bert_chunks_embedding(tokenized_right_text, right_sentence_embedding,
                                                         right_chunk_emb_ids, right_chunks[int(sid) - 1])
                unwrap_emb(right_chunk_emb_ids, right_matrix, vectors)

            elif emb_type == "glove":
                left_matrix = get_glove_embeddings(emb_matrix, left_chunk_emb_ids, stoi, left_chunks[int(sid) - 1], unk_token_idx)
                right_matrix = get_glove_embeddings(emb_matrix, right_chunk_emb_ids, stoi, right_chunks[int(sid) - 1], unk_token_idx)

            left_norm = left_matrix / left_matrix.norm(dim=1)[:, None]
            right_norm = right_matrix / right_matrix.norm(dim=1)[:, None]
            cosine_sim = torch.mm(left_norm, right_norm.transpose(0, 1))
            values, indices = torch.max(cosine_sim, 1)
            indices = indices.numpy()
            selected_right_chunks = [right_chunk_emb_ids[i] for i in indices]
            total_alignments_counter += len(indices)

            for i in range(len(left_chunk_emb_ids)):
                selected_r_chunk = selected_right_chunks[i]
                current_l_chunk = left_chunk_emb_ids[i]
                r_emb_set = emb_ids_map['chunk_emb_mapping'][current_l_chunk]
                if selected_r_chunk in r_emb_set:
                    match_counter += 1
        except Exception as e:
            print e

    if emb_type == "bert" and output_file is not None:
        _write_vectors(vectors, output_file)

    print "Total alignment match percentage ratio : ", (float(match_counter) / total_alignments_counter) * 100.0


# Reference from Vivek's paper
def compute_chunk_to_sentence_score(gold_alignments, output_sts_format_file):
    with codecs.open(gold_alignments, mode="rt", encoding="utf-8") as fp:
        with codecs.open(output_sts_format_file, mode='w', encoding='utf-8') as out:
            out.write("pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment\n")
            alignments = fp.read()
            # This to escape failure of XML parsing
            alignments = alignments.replace("<==>", "==").replace("&", "&amp;")
            alignments = alignments.encode("ascii", errors="ignore")
            tree = ET.fromstring(alignments)
            sentences = tree.findall("sentence")
            for idx, sentence in enumerate(sentences):
                sentence_1 = (sentence.text.split("\n")[1].split("//")[1]).strip()
                sentence_2 = (sentence.text.split("\n")[2].split("//")[1]).strip()

                alignment_text = sentence.findall("alignment")[0].text.lower()
                chunks_text = alignment_text.split("\n")
                chunks_len_counter = 0
                sent_score = 0.0
                for chunk_text in chunks_text:
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue

                    chunks_len_counter += 1

                    chunk_align_label = (chunk_text.split("//")[1]).strip()
                    chunk_similarity_str = (chunk_text.split("//")[2]).strip()
                    if chunk_similarity_str == "nil":
                        chunk_similarity = 0
                    else:
                        chunk_similarity = int(chunk_similarity_str)

                    chunk_label_importance = 0.7

                    if chunk_align_label == "equi":
                        chunk_label_importance = 1
                    elif chunk_align_label == "oppo":
                        chunk_label_importance = -1

                    sent_score += (chunk_label_importance * chunk_similarity)
                sent_score /= chunks_len_counter
                sent_score = abs(sent_score)
                out.write("1" + "\t" + sentence_1 + "\t" + sentence_2 + "\t" + str(sent_score) + "\t" + "NEUTRAL\n")


if __name__ == '__main__':
    emb_type = "bert"

    parser = argparse.ArgumentParser(description='BERT embeddings generation script')
    parser.add_argument('-ga', '--gold_alignments', default="../datasets/sts_16/train_2015_10_22.utf-8/STSint.input.images.wa", help='gold alignments file', required=False)
    parser.add_argument('-lc', '--left_chunks', default="../datasets/sts_16/train_2015_10_22.utf-8/STSint.input.images.sent1.chunk.txt", help='left chunks file', required=False)
    parser.add_argument('-rc', '--right_chunks', default="../datasets/sts_16/train_2015_10_22.utf-8/STSint.input.images.sent2.chunk.txt", help='right chunks file', required=False)
    parser.add_argument('-o', '--output_file', default="../src/resources/bert_base_uncased_input_images_1536._emb.bin", help='output file', required=False)

    args = parser.parse_args()

    if emb_type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()

        chunk_cosine_sim(
            gold_alignments=args.gold_alignments,
            embedding_file="",
            left_chunks_file=args.left_chunks,
            right_chunks_file=args.right_chunks,
            bert_tokenizer=tokenizer,
            bert_model=model,
            emb_type=emb_type,
            output_file=args.output_file)

    elif emb_type == "glove":
        chunk_cosine_sim(
            gold_alignments="datasets/sts_16/test/STSint.testinput.headlines.wa",
            embedding_file="glove.840B.300d.txt",
            left_chunks_file="datasets/sts_16/test/STSint.testinput.headlines.sent1.chunk.txt",
            right_chunks_file="datasets/sts_16/test/STSint.testinput.headlines.sent2.chunk.txt", emb_type=emb_type)
