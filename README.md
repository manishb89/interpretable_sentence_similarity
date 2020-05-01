Implementation of our IJCAI 20 paper on Logic Constrained Pointer Networks for Interpretable Sentence Similarity.



## Citation
```
@article{maji2020logic,
  title={Logic Constrained Pointer Networks for Interpretable Textual Similarity},
  author={Maji*, Subhadeep and Kumar*, Rohan and Bansal, Manish and Roy, Kalyani and Goyal, Pawan},
  year={2020},
}
```

## Prerequisites
```
pytorch
pytorch-pretrained-bert
gensim
numpy
spacy
requests
tqdm
nltk
lxml
```

## Setup/Installation

SemEval-2016 iSTS task dataset (headlines and images):
 - Download train/test [sets](http://alt.qcri.org/semeval2016/task2/) and unzip under ```datasets/sts_16``` directory.

Spacy:
```bash
pip install spacy
python -m spacy download en
```
pytorch-pretrained-bert:
```python
from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
```

Note: Downloading pre-trained BertModel (```bert-base-uncased```) might take longer depending on internet connectivity.

## Steps to reproduce
Note: Following scripts default to the headlines dataset but can be run for the images dataset as well.
### 1. Generate BERT embeddings
```bash
python src/corpus/chunk_embedding.py --gold_alignments STSint.input.headlines.wa 
                                     --left_chunks STSint.input.headlines.sent1.chunk.txt 
                                     --right_chunks STSint.input.headlines.sent2.chunk.txt 
                                     --output_file bert_base_uncased_input_headlines_1536._emb.bin 
```
Note: Generate embeddings for both train and test separately.

### 2. Generate resource files for FOL constraints
```bash
cd src
# Create ConceptNet cache of related chunks from all sentences
python scripts/create_cn_cache.py --data_dir ../datasets/sts_16

# Create mapping resource file from left & right sentences according to ConceptNet relations
python scripts/create_constr_map.py --data_dir ../datasets/sts_16
```

## 3. Train and Evaluate model

Set all resources paths e.g. iSTS chunk dataset files, FOL constraints & BERT embeddings generated as per above instructions) in ```training/configuration.py``` to appropriate variables. Other hyperparameters could also be controlled via ```configuration.py```
- ```output_constr``` can be “C1” or “” to disable structured knowledge constraints (R1)  
- ```syn_scores``` is a boolean that enables/disables syntactic constraints (R2)  
- ```rho``` sets importance of constraints  
- ```gpuid``` sets the gpu id for pytorch
- ```max_epoch``` controls number of epochs
- ```patience``` parameter used for early stopping

At the end, ```train.py``` saves best model checkpoint in the ```model.checkpoint``` file and evaluates F1-score on test set.

Sample commands:
```bash
# To run in default settings i.e. without constraints.
python training/train.py

# Enable constraint
# Default resource is cn_combined_unigram_content_only.json in respective train/test path
python training/train.py --constraint

# If constraint resource needs to be changed
python training/train.py --constraint --resource cn_combined_bigram.json

# To run on another dataset
python training/train.py --dataset_type image

# To change hidden dimension in pointer network
python training/train.py --hidden_dim 150 --constraint

# Change rho
python training/train.py --hidden_dim 150 --constraint --rho 2.0
``` 
