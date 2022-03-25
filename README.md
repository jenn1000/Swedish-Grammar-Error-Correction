# Swedish-Grammar-Error-Correction
Neural Grammar Error Correction using GRU and T5

## Abstract 
Grammar Error Correction (GEC) refers to an automatic process of correcting grammatical errors in given sentences. GEC is being actively studied as a sister field of machine translation(MT), but there exists a fatal limitation, namely the paucity of the parallel dataset for most of the languages including Swedish. This project, therefore, presents a neural baseline for Swedish GEC, using two neural models - bi-directional Gated Recurrent Unit (GRU) and the pre-trained T5 model, which are proven to be effective on text-to-text setting. We further experimented with two data supplementation methods to bolster the small-scale Swedish error-annotated dataset. Our baseline consists of a bi-directional GRU model trained on the seed corpus (F0.5 score = 40.37).The GRU model on the train data combined with synthetic data with a proportion of 1:10 achieved the best result (F0.5 score = __72.95__)

## Data Supplementation Methods
<img src = "https://user-images.githubusercontent.com/77832890/152796450-42e28a93-b10e-4965-81fb-54033e57f55a.jpg" width=40% height=40%>

In this study, we implement two data supplementation methods - spelling normalization and synthetic data augmentation using pre-defined rules. The flow of our supplementation techniques are shown above. 

1. Spelling normalization 


    We use SWEGRAM, the web-based linguistic annotation tool for Swedish to normalize spelling on erroneous sentences. The dataset used in this experiment are three SweLL-pilot sub-corpora containing L2 Swedish learner essays. We used a basic spell checking function to normalize learners' sentences that contain lexical-level errors as shown below.
    
      + ___peretar__ med_ -> ___pratar__ med_
      + _läser __sveniska___ -> _läser __svenska___

2. Synthetic data augmentation

    We create pre-defined rules to generate character-level, morphological-level, and word-level errors. To generate realistic grammatical errors that L2 Swedish learners would make, we propose four pre-defined rules - _insertion, deletion, transposition, and lemmatization_. To maximize the randomness of error generation, our proposed augmentation function randomly applies four rules with a probability of 0.05 respectively. The examples are written below. 
    
    + _fortsätta_ -> _fortssätta_
    + _partiet -> partien_
    + _hästar -> häster_
    + _De höga priserna -> De hög pris på hus_


## GEC Model
1. Baseline

    A ```bi-directional GRU```. Only the seed corpus (SweLL-gold corpus) is used for the baseline
    
2. T5
  
   We chose ```T5-base``` for further experiment on the Transformer-based model, due to its reknowned achievement on various text generation tasks. 

## Training Details
1. Preprocessing
    Tokenization using spaCy tokenizer for Swedish. The texts are tokenized and fed into the model through Torchtext 0.11.0 framework
    
2. Libraries
    The recurrent seq2seq model used in this experiment was built on Pytorch 1.10.0. The transformer model used in this experiment is built on simpleT5 library, constructed on top of PyTorch-lightening and Huggingface Transformer. 

    
_Further details are written in the GEC.pdf_

