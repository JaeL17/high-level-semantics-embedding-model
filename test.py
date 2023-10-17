import json
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import create_annoy, annoy_search
import os
import argparse
from datasets import load_dataset

def parse_testdata():
    dataset = load_dataset("mteb/mtop_intent")

    train_text = [i['text'] for i in dataset['train']]
    train_lab = [i['label_text'] for i in dataset['train']]

    test_text = [i['text'] for i in dataset['test']]
    test_lab = [i['label_text'] for i in dataset['test']]
    return train_text, train_lab, test_text, test_lab
    
def parse_prediction(pred, test_text, test_lab, train_text, train_lab, topn):
    acc_count = 0
    pred_dict = []
    for i in range(len(pred)):
        temp_dict = {'test_query': test_text[i],
                    'test_label': test_lab[i],
                    'train_query': train_text[pred[i][0]],
                    'train_label': train_lab[pred[i][0]]}

        label_ch = [train_lab[j] for j in pred[i][:topn]]
        if test_lab[i] in label_ch:
            acc_count +=1
            temp_dict['correct'] = 'O'
        else:
            temp_dict['correct'] = 'X'
        pred_dict.append(temp_dict)
    return acc_count/len(pred), pred_dict

def top3_accuracy(model, model_name, test_text, test_lab, train_text, train_lab, test_name):
    idx_vec = model.encode(train_text)
    test_vec = model.encode(test_text)
    create_annoy(idx_vec, f'test.ann', args.annoy_tree_num, 'annoy_index')
    pred, dist = annoy_search(f'annoy_index/test.ann', test_vec, args.annoy_tree_num)
    
    print(f'***** {test_name} test result *****')
    score1, pred_dict1 = parse_prediction(pred, test_text, test_lab, train_text, train_lab, 1)
    print(f"Top1 accuracy score: {score1}\n")
    
    score2, pred_dict = parse_prediction(pred, test_text, test_lab, train_text, train_lab, 2)
    print(f'Top2 accuracy score: {score2}\n')
    
    score3, pred_dict = parse_prediction(pred, test_text, test_lab, train_text, train_lab, 3)
    print(f'Top3 accuracy score: {score3}\n')
    
    
def run_test(args):
    # Load test model
    test_model = SentenceTransformer(args.test_model_path)
    
    # Load test dataset
    train_text, train_lab, test_text, test_lab = parse_testdata()
    top3_accuracy(model = test_model, model_name = args.test_model_path, test_text = test_text,test_lab = test_lab, train_text=train_text, train_lab=train_lab, test_name= 'mteb/mtop_intent')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model_path", type=str, required=True)
    parser.add_argument("--annoy_tree_num", type=int, required=True)
    args = parser.parse_args()
    
    run_test(args)