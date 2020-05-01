import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="interpretable sentence similarity")

    parser.add_argument('--dataset_type', default='headlines', choices=['headlines', 'images'],
                        help='headlines or images')
    parser.add_argument('--constraint', action='store_true', help='use FOL constraints')
    parser.add_argument('--resource_suffix', default='cn_combined_unigram_content_only.json',
                        help='constraint file name after fol.')

    parser.add_argument('--train_path', default='../datasets/sts_16/train_2015_10_22.utf-8/',
                        help='base path to train dataset')
    parser.add_argument('--test_path', default='../datasets/sts_16/test/',
                        help='base path to test dataset')
    parser.add_argument('--emb_type', default='chunk', help='type of the embedding', choices=["chunk"])
    parser.add_argument('--train_embedding', default=os.getcwd() + '/resources/bert_base_uncased_input_headlines_1536._emb.bin', help='train embeddings')
    parser.add_argument('--test_embedding', default=os.getcwd() + '/resources/bert_base_uncased_testinput_headlines_1536._emb.bin', help='test embeddings')

    parser.add_argument('--rho', default=4.0, type=float, help='rho')
    parser.add_argument('--gpuid', default=0, type=int, help='gpuid')
    parser.add_argument('--check_start_epoch', default=5, type=int, help='')
    parser.add_argument('--tol', default=1e-3, help='tolerance')
    parser.add_argument('--pat', default=5, help='patience')

    parser.add_argument('--input_dim', default=1536, type=int, help='embedding dimension')  # 1536
    parser.add_argument('--hidden_dim', default=768, type=int, help='Pointer network hidden dimension')  # 150
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--num_epoch', default=50, type=int, help='number of epoch')
    parser.add_argument('--eval_script', default='../scripts/evalF1.pl', help='eval script path')

    args = parser.parse_args()
    return args
