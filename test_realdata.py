##test real data
import pandas as pd
from UDC import UDC
from dc.dc import dc
from HCR import HCR_algorithm
from scipy.stats import chi2_contingency

def compact_X(X):
    return [[x] for x in X]

def test_temperature():
    data = pd.read_csv('real_data/pair0042.txt', sep='\t', header=None)
    data[0] = pd.cut(x = data[0], bins = [0,31,59,90,120,151,181,212,243,273,304,334,366],labels = [0, 1, 2, 3,4,5,6,7,8,9,10,11])
    data[1] = data[1].round()
    data[1] = data[1].astype(int)
    data[1] = data[1].replace(sorted(data[1].unique()), [i for i in range(len(data[1].unique()))])
    score = UDC(data).find_best_direction(data, ['cyclic', 'no'])
    return score


def test_adult():
    data = pd.read_csv('real_data/adult.csv', sep=',')
    data = data[['income', 'workclass', 'occupation']]
    nan_value = float("NaN")
    data.replace("?", nan_value, inplace=True)
    data.dropna(subset = ["occupation", 'workclass'], inplace=True)
    data['occupation'] = data['occupation'].map({"Adm-clerical": "admin", "Armed-Forces": "armed_forces","Craft-repair": "blue_collar",
       "Handlers-cleaners": "blue_collar","Machine-op-inspct": "blue_collar","Farming-fishing": "blue_collar","Transport-moving": "blue_collar",
       "Other-service": "service","Priv-house-serv": "service","Sales": "sales", "Exec-managerial": "white_collar","Prof-specialty": "professional",
       "Tech-support": "other_occupations","Protective-serv": "other_occupations"})
    data['workclass']= data['workclass'].map({"Private": "private","Self-emp-not-inc": "self_employed",
        "Self-emp-inc": "self_employed","Federal-gov": "public_servant","Local-gov": "public_servant",
        "State-gov": "public_servant","Without-pay": "unemployed","Never-worked": "unemployed"})
    data['income']= data['income'].replace(['<=50K', '>50K'], [0,1])
    data['occupation'] = data['occupation'].replace(["admin", "armed_forces", "blue_collar", "white_collar",
         "service", "sales", "professional", "other_occupations"], [0,1,2,3,4,5,6,7])
    data['workclass'] = data['workclass'].replace(["private", "self_employed", "public_servant", "unemployed"], [0,1,2,3])
    pairs = [['workclass','income'], ['occupation','income']]
    scores=[]
    for pair in pairs:
        dataa = data[pair]
        dataa.columns = range(dataa.shape[1])
        score = UDC(dataa).find_best_direction(dataa, ['no', 'no'])
        #score = HCR_algorithm(dataa)
        scores.append(score)
    return scores


def test_inflammation():
    #0 - Temperature of patient, 1 - Occurrence of nausea { yes, no }
    #2 - Lumbar pain { yes, no }, 3 - Urine pushing  { yes, no }
    #4 Micturition pains { yes, no }, 5 Burning of urethra
    #6Inflammation of urinary bladder, 7 Nephritis of renal pelvis origin 
    df = pd.read_csv('real_data/diagnosis.txt', delimiter = "\t", header = None)
    pairs = [[6,1], [6,2], [6,3], [6,4], [6,5],[7,1],[7,2],[7,3],[7,4],[7,5]]
    scores= []
    for pair in pairs:
        data = df[pair]     
        data.columns = range(data.shape[1])
        data.loc[data[0]=='yes', 0] = 1
        data.loc[data[0]=='no', 0] = 0
        data.loc[data[1]=='yes', 1] = 1
        data.loc[data[1]=='no', 1] = 0
        #score = HCR_algorithm(data)
        score = UDC(data).find_best_direction(data, ['no', 'no'])
        scores.append(score)
    return scores

def test_horsecolic():
    #data[1] - surgery not surgery
    data = pd.read_csv('real_data/horse_colic.csv', delimiter = ";", header = None)
    data[0] = data[0].replace(sorted(data[0].unique()), [i for i in range(len(data[0].unique()))])
    data[1] = data[1].replace(sorted(data[1].unique()), [i for i in range(len(data[1].unique()))])
    #score = HCR_algorithm(data)
    score = UDC(data).find_best_direction(data, ['no', 'no'])
    return score

def test_bridges():
    df = pd.read_csv('real_data/bridges.txt', delimiter = ",", header = None)
    ## 6 - LANES / 9 - MATERIAL /  12 - Type / 4- PURPOSE
    pairs = [[9,6], [4,12]]
    scores = []
    nan_value = float("NaN")
    for pair in pairs:
        data = df[pair]
        data.columns = range(data.shape[1])
        data.replace("?", nan_value, inplace=True)
        data.dropna(subset = [0,1], inplace=True)
        data[0] = data[0].replace(sorted(data[0].unique()), [i for i in range(len(data[0].unique()))])
        data[1] = data[1].replace(sorted(data[1].unique()), [i for i in range(len(data[1].unique()))])
        #score = HCR_algorithm(data)
        score = UDC(data).find_best_direction(data, ['no', 'no'])
        scores.append(score)
    return scores
        
  
#print(test_adult())
#print(test_inflammation())
#print(test_bridges())
#print(test_horsecolic())
print(test_temperature())