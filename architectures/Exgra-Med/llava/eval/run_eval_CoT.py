import os
import argparse
import json
import collections
import random
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from eval_metrics.glossary import *
from sklearn.metrics import f1_score as f1_score_eval
import warnings
import numpy as np
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--candidate', type=str, default="candidate.json", help='path to candidate answer file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred, candidate, criterion=None):    
    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    open_hit_scores = collections.defaultdict(list)
    corr = 0
    total = 0
    wrong_label = 0
    gt_dict = {"moderate dementia": 0, "mild dementia": 1, "non dementia": 2}
    gt_list = []
    pred_list = []

    corr_json = []
    incorr_json = []
    step_by_step_acc = {}
    for gt_item, pred_item in zip(gt, pred):
        try:
            gt_results = gt_item['conversations']
        except:
            gt_results = gt_item['conversatons']
        gt_value = gt_results[1]['value'].lower().split("+ final answer:")[-1].strip()
        pred_value = pred_item['text'].lower().split("+ final answer:")[-1].split("assistant:")[-1].strip()
        # print("=======Groundtruth=========")
        # print(gt_value)
        # print("=======Prediction=========")
        # print(pred_value)
        # print("##################################################")
        ############################
        gt_list_ans_cot = [normalize_word(item.strip().split("\n- step")[0].split("\n\n+ final")[0].replace("\n", "").strip()) for item in gt_results[1]['value'].lower().split("answer:")][1:-1]
        pred_list_ans_cot = [normalize_word(item.strip().split("\n- step")[0].split("+ final")[0].replace("\n", "").strip()) for item in pred_item['text'].lower().split("answer:")][1:-1]
        print(gt_list_ans_cot)
        # print(pred_item['text'].lower())
        print(pred_list_ans_cot)
        step_begin = len(gt_list_ans_cot) - len(pred_list_ans_cot)
        print("-------------------------")
        for item_index in range(len(pred_list_ans_cot)):
            if f"Step {item_index+step_begin+1}" not in step_by_step_acc:
                step_by_step_acc[f"Step {item_index+step_begin+1}"] = []

            step_by_step_acc[f"Step {item_index+step_begin+1}"].append(pred_list_ans_cot[item_index]==gt_list_ans_cot[item_index+step_begin])
        ############################
        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        if gt_value == pred_value:
            corr += 1
            ele = {}
            ele["image"] = gt_item['image']
            ele["question"] = gt_results[0]['value']
            ele['prediction'] = pred_item['text']
            ele['groundtruth'] = gt_results[1]['value']
            corr_json.append(ele)
        else:
            ele = {}
            ele["image"] = gt_item['image']
            ele["question"] = gt_results[0]['value']
            ele['prediction'] = pred_item['text']
            ele['groundtruth'] = gt_results[1]['value']
            incorr_json.append(ele)
            print("=======Groundtruth=========")
            print(gt_value)
            print("=======Prediction=========")
            print(pred_value)
            print("##################################################")
        total += 1

        try:
            pred_list.append(gt_dict[pred_value])
            gt_list.append(gt_dict[gt_value])
        except:
            print("=======Groundtruth=========")
            print(gt_value)
            print("=======Prediction=========")
            print(pred_value)
            print("##################################################")
            gt_list.append(gt_dict[gt_value])
            pred_list.append([pre for pre in [0,1,2] if pre != gt_dict[gt_value]][0])
            wrong_label += 1
            # raise Exception("No exist in groundtruth dictionary")

        open_hit_scores['hit'].append(calculate_appearance_with_normalization(pred_value, gt_value, candidate))
        open_hit_scores['q_id'].append(pred_item['question_id'])

        exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
        exact_scores['q_id'].append(pred_item['question_id'])


        f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
        f1_scores['f1'].append(f1_score)
        f1_scores['precision'].append(precision)
        f1_scores['recall'].append(recall)
        f1_scores['q_id'].append(pred_item['question_id'])

        # if isinstance(f1_scores['hit'][-1], str):
        #     # import pdb; pdb.set_trace()

        b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                hypothesis=str(pred_value).lower().split())
        b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
        b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
        b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
        
        bleu_scores['q_id'].append(pred_item['question_id'])
        bleu_scores['bleu_score'].append(b_score)
        bleu_scores['bleu_score_1'].append(b_score_1)
        bleu_scores['bleu_score_2'].append(b_score_2)
        bleu_scores['bleu_score_3'].append(b_score_3)

    # import pdb; pdb.set_trace()
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
    precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
    recall = sum(f1_scores['recall']) / len(f1_scores['recall'])

    bleu_score   = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score'])
    bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1'])
    bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2'])
    bleu_score_3 = sum(bleu_scores['bleu_score_3']) / len(bleu_scores['bleu_score_3'])

    open_hit_score = sum(open_hit_scores['hit']) / len(open_hit_scores['hit'])
    closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0

    num_open, num_close = len(open_hit_scores['hit']), len(closed_scores['hit'])
    print(f'num_open {num_open} || num_close {num_close}')

    f1_classification = f1_score_eval(gt_list, pred_list, average='macro', labels=[0, 1, 2])

    final_results = [
            ['exact match score', exact_score*100], 
            ['f1 score', f1_score*100], 
            ['precision', precision*100], 
            ['recall', recall*100], 
            ['bleu_score', bleu_score*100], 
            ['bleu_score_1', bleu_score_1*100], 
            ['bleu_score_2', bleu_score_2*100], 
            ['bleu_score_3', bleu_score_3*100], 
            ['open accuracy', open_hit_score*100],
            ['From Scratch Accuracy', corr/total*100],
            ['F1 classification', f1_classification*100],
            ["Wrong label", wrong_label]
        ]
    
    for step in step_by_step_acc:
        final_results.append([step, np.average(step_by_step_acc[step])*100])
    return tabulate(
        final_results, 
        headers=['Metric', 'Performance']
    ), corr_json, incorr_json

if __name__ == '__main__':
    args = parse_option()

    dataset = args.gt.split("/")[-2]
    print(f"\n========\n {dataset}")

    gt = json.load(open(args.gt, 'r'))
    candidate = json.load(open(args.candidate, 'r'))
    pred = load_jsonl(args.pred)

    gt_ids = [item['id'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
    print(f'num_gt_ids: {num_gt_ids} || num_pred_ids: {num_pred_ids}')
    # import pdb; pdb.set_trace()
    assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results, corr_json, incorr_json = evaluate(gt, pred, candidate)
    print(results)
    filename = os.path.splitext(os.path.basename(args.pred))[0] + '.txt'
    tabulate_file = os.path.join(os.path.dirname(args.pred),filename)
    with open(tabulate_file, 'w') as f: 
        f.write(results)


    with open(args.pred.replace(".jsonl", "_correct.json"), 'w') as f:
        json.dump(corr_json, f, indent=4)

    with open(args.pred.replace(".jsonl", "_incorrect.json"), 'w') as f:
        json.dump(incorr_json, f, indent=4)
