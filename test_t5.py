import pandas as pd
import os
from simplet5 import SimpleT5
import argparse
import nltk
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support



###################argparse#############################
parser = argparse.ArgumentParser(description='Train the model!')

parser.add_argument('--train', type=str, default='data/train_v2.tsv')
parser.add_argument('--valid', type=str, default='data/valid_v2.tsv')
parser.add_argument('--test', type=str, default='data/test.tsv')
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()
#########################################################


train = pd.read_csv(args.train, sep='\t', header=None,  names=['source_text', 'target_text'])
valid = pd.read_csv(args.valid, sep='\t', header=None,  names=['source_text', 'target_text'])
test = pd.read_csv(args.test, sep='\t', header=None,  names=['source_text', 'target_text'])

model = SimpleT5()
model.from_pretrained('t5', 't5-base')

model.load_model("t5",f"t5model/{args.model}", use_gpu=True)

predictions = []

precisions = []
recalls = []
f_scores = []

input = list(test['source_text'])
target = list(test['target_text'])

assert len(input) == len(target)

for i, sent in enumerate(input):
    prediction = model.predict(sent) #['abc']
    prediction = str(prediction)

    prediction_str = prediction
    print('prediction: ', prediction, 'target ', target[i])
    precision_,recall_, f_score = chrf_precision_recall_fscore_support(target[i].split(), prediction.split(),n=1,
	                                                                beta=0.5, epsilon=1e-16)
  
    
    f_scores.append(f_score)
    precisions.append(precision_)
    recalls.append(recall_)


final_f = sum(f_scores)/len(f_scores)
final_p = sum(precisions)/len(precisions)
final_r = sum(recalls)/len(recalls)

with open(f'./result/T5-{args.model}.txt', 'w') as f:
    f.write(f'f_score:   {final_f}')
    f.write(f'precision: {final_p}')
    f.write(f'recall: {final_r}')

with open(f'./case_analysis/T5-{args.model}_prediction.txt', 'w') as f:
    for text in prediction:
        t = " ".join(text)
	f.write(t)
	f.write('\n')

with open(f'./case_analysis/T5-{args.model}_target.txt', 'w') as f:
    for text in target:
        t = " ".join(text)
        f.write(t)
        f.write('\n')

