import sys; sys.path = [''] + sys.path
import torch, os
from datetime import datetime
from torch.utils.data import DataLoader
from copy import deepcopy


from corpus import corpus as CC
from model import PointerNetwork as PN
from model import model as MM
from embed import vocab as VB
from embed import embedding as EM

from training.holder import Holder
from scripts.external_eval import evaluate_alignment
from training.configuration import parse_args


def identity_collate(batch):
    return batch


def train(fp1, fp2, fp1_chunk, fp2_chunk, align_file, glove_fp, cfg):
    """
    @param fp1: Aligned Corpus, first sentences
    @param fp2: Aligned Corpus, second sentences
    @param fp1_chunk: Chunk information file for fp1
    @param fp2_chunk: Chunk information file for fp2
    @param align_file: Chunk level alignment across sentences
    @param glove_fp: File path of the Glove embeddings 
    """
    print 'Reloaded train at: {}'.format(str(datetime.now()))
    reader = CC.DatasetReader(fp1, fp2, fp1_chunk, fp2_chunk, align_file)
    reader.read()
    emb_reader = VB.EmbeddingReader(glove_fp)
    chunk_embedding = EM.ChunkEmbedding(emb_reader)
    
    train_data_reader = CC.TrainerDatasetReader(reader, emb_reader, cfg)
    train_data_loader = DataLoader(train_data_reader, batch_size=32, collate_fn=identity_collate,
                                   shuffle=True)

    pointer_network = PN.PointerNetwork(chunk_embedding, cfg)
    model_trainer = MM.ModelTrainer(pointer_network, train_data_loader)
    return model_trainer


if __name__ == '__main__':

    args = parse_args()

    emb_type = args.emb_type

    train_path = args.train_path
    dataset_type = args.dataset_type
    fp1 = train_path + 'STSint.input.' + dataset_type + '.sent1.txt'
    fp2 = train_path + 'STSint.input.' + dataset_type + '.sent2.txt'
    fp1_chunk = train_path + 'STSint.input.' + dataset_type + '.sent1.chunk.txt'
    fp2_chunk = train_path + 'STSint.input.' + dataset_type + '.sent2.chunk.txt'
    align_file = train_path + 'STSint.input.' + dataset_type + '.wa'

    train_bert_fp = args.train_embedding
    test_bert_fp = args.test_embedding
    resource_suffix = args.resource_suffix

    enable_constraint = args.constraint

    cfg = Holder()

    if enable_constraint:
        cfg.syn_scores = True
        cfg.output_constr = 'C1'
    else:
        cfg.syn_scores = False
        cfg.output_constr = ''

    cfg.constr_res_path = train_path + 'STSint.input.' + dataset_type + '.fol.' + resource_suffix
    cfg.rho = args.rho
    cfg.gpuid = args.gpuid
    cfg.input_dim = args.input_dim
    cfg.hidden_dim = args.hidden_dim

    reader = CC.DatasetReader(fp1, fp2, fp1_chunk, fp2_chunk, align_file)
    reader.read()
    emb_reader = VB.EmbeddingReader(train_bert_fp, emb_type=emb_type)
    chunk_embedding = EM.ChunkEmbedding(emb_reader)
    
    train_data_reader = CC.TrainerDatasetReader(reader, emb_reader, cfg)
    train_data_loader = DataLoader(train_data_reader, batch_size=args.batch_size, collate_fn=identity_collate,
                                   shuffle=True)

    pointer_network = PN.PointerNetwork(cfg)
    model_trainer = MM.ModelTrainer(pointer_network, train_data_loader, cfg)

    test_path = args.test_path
    fp1_test = test_path + 'STSint.testinput.' + dataset_type + '.sent1.txt'
    fp2_test = test_path + 'STSint.testinput.' + dataset_type + '.sent2.txt'
    fp1_test_chunk = test_path + 'STSint.testinput.' + dataset_type + '.sent1.chunk.txt'
    fp2_test_chunk = test_path + 'STSint.testinput.' + dataset_type + '.sent2.chunk.txt'
    align_file_test = test_path + 'STSint.testinput.' + dataset_type + '.wa'
    script_file = args.eval_script

    cfg_test = Holder()
    cfg_test.syn_scores = cfg.syn_scores
    cfg_test.constr_res_path = test_path + 'STSint.testinput.' + dataset_type + '.fol.' + resource_suffix
    cfg_test.output_constr = cfg.output_constr
    cfg_test.rho = cfg.rho
    cfg_test.gpuid = cfg.gpuid
    cfg_test.input_dim = args.input_dim
    cfg_test.hidden_dim = args.hidden_dim

    # training parameters
    max_epoch = args.num_epoch
    check_start_epoch = args.check_start_epoch
    tol = args.tol
    patience = args.pat

    reload(CC)
    reader_test = CC.DatasetReader(fp1_test, fp2_test, fp1_test_chunk, fp2_test_chunk, align_file_test)
    reader_test.read()

    if emb_type == "chunk":
        emb_reader = VB.EmbeddingReader(test_bert_fp, emb_type=emb_type)
    test_data_reader = CC.TrainerDatasetReader(reader_test, emb_reader, cfg_test)
    test_data_loader = DataLoader(test_data_reader, batch_size=args.batch_size, collate_fn=identity_collate,
                                   shuffle=True)

    best_f1 = 0.0
    best_model = None
    best_epoch = None
    remaining_patience = patience
    stopped_early = False

    for i in xrange(1,max_epoch+1):
        model_trainer.train(i)
        
        if i >= check_start_epoch:
            f1 = evaluate_alignment(model_trainer, train_data_loader, script_file, align_file)

            test_file_path = test_path + "test_alignments_" + str(i)
            test_f1 = evaluate_alignment(model_trainer, test_data_loader, script_file, align_file_test,
                                         tmp_file_name=test_file_path)

            if f1 - best_f1 > tol:
                # print 'remaining_patience RESET, f1 diff {} > {}'.format(f1 - best_f1, tol)
                remaining_patience = patience
                print 'F1 Improvement! Saving model.'
                best_model = deepcopy(model_trainer)
                best_epoch = i
                torch.save(pointer_network.state_dict(), 'model.checkpoint')
                best_f1 = f1
                best_f1_test = test_f1
            else:
                remaining_patience -= 1

            print 'Train f1: {}, Test F1:{}\n'.format(f1, test_f1)
            # print 'remaining_patience', remaining_patience, '\n'

            if remaining_patience <= 0:
                print '\nNo improvement for {} epochs, f1 diff {} < {}.'.format(patience, f1 - best_f1, tol)
                print 'Early stopping at epoch {}. Train f1: {}'.format(i, f1)
                stopped_early = True
                break

    # check at end
    if not stopped_early and f1 - best_f1 > tol:
        print 'Saving model with train f1: {}\n'.format(f1)
        best_model = deepcopy(model_trainer)
        best_epoch = i
        torch.save(pointer_network.state_dict(), 'model.checkpoint')
        best_f1 = f1

    f1 = evaluate_alignment(model_trainer, test_data_loader, script_file, align_file_test)
    print 'Final(last epoch) test F1:', f1

    print '\nBest model obtained at epoch: {}. Train F1: {}, Test F1:{}'.format(best_epoch, best_f1, best_f1_test)
