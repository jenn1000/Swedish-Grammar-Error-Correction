import pandas as pd
import os
from simplet5 import SimpleT5
import argparse


###################argparse#############################
parser = argparse.ArgumentParser(description='Train the model!')

parser.add_argument('--train', type=str, default='train.tsv')
parser.add_argument('--valid', type=str, default='valid.tsv')
parser.add_argument('--test', type=str, default='test.tsv')
args = parser.parse_args()
#########################################################


train = pd.read_csv(args.train, sep='\t', header=None,  names=['source_text', 'target_text'])
valid = pd.read_csv(args.valid, sep='\t', header=None,  names=['source_text', 'target_text'])
test = pd.read_csv(args.test, sep='\t', header=None,  names=['source_text', 'target_text'])

model = SimpleT5()
model.from_pretrained('t5', 't5-base')

model.train(train_df=train,
            eval_df=valid,
            source_max_token_len=50,
            target_max_token_len=50,
            batch_size=8,
            max_epochs=5,
            use_gpu=True,
            outputdir='./t5model',
            early_stopping_patience_epochs=0)

