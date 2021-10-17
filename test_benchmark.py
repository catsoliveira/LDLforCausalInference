##Test Benchmark
import numpy as np
import pandas as pd
#import random
import matplotlib.pyplot as plt
import os
#import re
from UDC import UDC
from dc.dc import dc
from HCR import HCR_algorithm


def get_ground_truth(): #1 if A->B, -1 if B->A and 0 otherwise
    pairs=[]
    values={}
    final_pairs = []
    with open('benchmark_data/categorical_pairs.txt', 'r') as file:
        for pair in file.readlines():
            pairs.append(str(int(pair)))
    with open('benchmark_data/CEfinal_train_target.csv', 'r') as f:
        for info in f.readlines():
            inf = info.split(',')
            if any('train' + string == inf[0] for string in pairs):
                if inf[1] == '1' or inf[1] =='-1':
                    train = inf[0]
                    final_pairs.append(train.split('train')[1])
                    values[train.split('train')[1]] = inf[1]      
    return final_pairs, values

def load_data(pair):
    d = np.loadtxt('benchmark_data/' + pair)
    data = pd.DataFrame(d)
    data[0] = data[0].replace(sorted(data[0].unique()), [i for i in range(len(data[0].unique()))])
    data[1] = data[1].replace(sorted(data[1].unique()), [i for i in range(len(data[1].unique()))])
    return data


def load_pairs(pairs_file):
    entries = os.listdir('benchmark_data/')
    for entry in entries:
        val = entry.split('.')[0]
        if entry.startswith('train') and any('train' + string == val for string in pairs_file):    
            yield load_data(entry)               

def compact_X(X):
    return [[x] for x in X]


def test_benchmark():
    truths = get_ground_truth()
    all_pairs = truths[0]
    truth = truths[1]
    nsample = 0
    res_dsc= []
    res_dc = []
    res_hcr = []
    diff_dsc = []
    diff_dc = []
    diff_hcr = []
    
    for i, data in enumerate(load_pairs(all_pairs)):
        nsample+=1
        score_dsc = UDC(data).find_best_direction(data, ['no','no'])
        score_dc = dc(compact_X(data[0]), compact_X(data[1]))       
        score_hcr = HCR_algorithm(data)
        
        diff_dsc.append(abs(score_dsc[0] - score_dsc[1]))
        diff_dc.append(abs(score_dc[0] - score_dc[1]))
        diff_hcr.append(abs(score_hcr[0] - score_hcr[1]))
        
        
        if score_dsc[0] < score_dsc[1]:
            cause_dsc = '1'
        elif score_dsc[0] > score_dsc[1]:
            cause_dsc = '-1'
        else:
            cause_dsc = ''
            
        if score_dc[0] < score_dc[1]:
            cause_dc = '1'
        elif score_dc[0] > score_dc[1]:
            cause_dc = '-1'
        else:
            cause_dc = ''
            
        if score_hcr[0]> score_hcr[1]:
            cause_hcr = '1'
        elif score_hcr[0]< score_hcr[1]:
            cause_hcr = '-1'    
        else:
            cause_hcr = ''
        
        true_cause = truth[all_pairs[i]]
        if cause_dsc == true_cause:
            res_dsc.append(True)
        else:
            res_dsc.append(False)     
        if cause_dc == true_cause:
            res_dc.append(True)
        else:
            res_dc.append(False) 
        if cause_hcr == true_cause:
            res_hcr.append(True)
        else:
            res_hcr.append(False) 
            
    ind_dsc = np.argsort(diff_dsc)[::-1]
    ind_dc = np.argsort(diff_dc)[::-1]
    ind_hcr = np.argsort(diff_hcr)[::-1]
    res_dsc = [res_dsc[i] for i in ind_dsc]
    res_dc = [res_dc[i] for i in ind_dc]
    res_hcr = [res_hcr[i] for i in ind_hcr]
    
    dec_rate = np.arange(0.01, 1.01, 0.01)
    accuracy_dsc= []
    accuracy_dc =[]
    accuracy_hcr = []
    for r in dec_rate:
        maxindex = int(r*nsample)
        rate_dsc = res_dsc[:maxindex]
        rate_dc = res_dc[:maxindex]
        rate_hcr = res_hcr[:maxindex]
        accuracy_dsc.append(sum(rate_dsc)/ len(rate_dsc))
        accuracy_dc.append(sum(rate_dc)/ len(rate_dc))
        accuracy_hcr.append(sum(rate_hcr)/ len(rate_hcr))
    print(dec_rate)
    print(accuracy_dsc)
    print(accuracy_dc)
    print(accuracy_hcr)
    plt.plot(dec_rate, accuracy_dsc, label = 'LDL')
    plt.plot(dec_rate, accuracy_dc, label = 'DC')
    plt.plot(dec_rate, accuracy_hcr, label = 'HCR')
    plt.plot(dec_rate, [0.5 for i in range(len(dec_rate))], 'r--')
    plt.xlabel('Decision Rate', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 11})
    plt.show()
    
    
print(test_benchmark())
  