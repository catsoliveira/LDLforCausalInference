import copy
import numpy as np
import math

def states_variables(data):
    rs= [sorted(data[column].unique()) for column in list(data.columns)]
    columns= [i for i in range(len(data.columns))]
    states = dict(zip(columns, rs))
    return states

def count_matrixN(data, ii, jj):
    states = states_variables(data)
    #print(states)
    instances = data.values.tolist()
    matrix = np.zeros((len(states[ii]), len(states[jj])))
    for i in states[ii]:
        for j in states[jj]:
            for instance in instances:
                if instance[ii]==i and instance[jj]==j:
                    matrix[states[ii].index(i)][states[jj].index(j)]+=1
    return matrix

def count_variable(data, i): 
    counts=[]
    instances = data.values.tolist()
    states = states_variables(data)
    for x in range(len(states[i])):
        count_val=0
        for instance in instances:
            if instance[i]==x:
                count_val +=1
        counts.append(count_val)
    return counts


def L_score(data, score_type):
    freqx = count_variable(data, 0)
    px = [freq/sum(freqx) for freq in freqx]
    Nyy = count_matrixN(data, 2, 1)
    pyy = (Nyy.T/Nyy.sum(axis=1)).T
    setx = states_variables(data)[0]
    sety = states_variables(data)[1]
    setyp = states_variables(data)[2]    
    value = 0 
    for i in range(len(setx)):
        if px[i]!= 0 and freqx[i]!=0:
            value += freqx[i]*math.log(px[i])
    for j in range(len(setyp)):
        for k in range(len(sety)):
            if Nyy[j][k]!= 0 and pyy[j][k]!=0:
                value += Nyy[j][k] * math.log(pyy[j][k])
    if score_type == 'bic':
        d = len(setyp)*(len(sety)-1)+(len(setx)-1)
        value -= d/2 * math.log(len(data))
    return value
        

def HCR(dataa, score_type='bic', max_iteration=1000): 
    data = copy.deepcopy(dataa)
    N = count_matrixN(data, 0, 1)
    print(N)
    setx = states_variables(data)[0]
    setyp = copy.deepcopy(setx)
    Yp = []
    for i in range(len(setx)):
        Yp.append(np.argmax(N[i,:]))
    fx = dict(zip(setx, Yp))
    data[2] = data[0].map(fx)
    newscore = L_score(data, score_type)
    oldscore = -float('Inf')
    iteration = 0 
    bestdata = copy.deepcopy(data)
    while newscore > oldscore:
        if iteration > max_iteration:
            break
        iteration +=1
        print(iteration)
        oldscore = copy.deepcopy(newscore)
        for i in setx:
            for j in setyp:
                temp = copy.deepcopy(data)
                temp.loc[temp[0] == i, 2] = j
                score = L_score(temp, score_type)
                #print(score)
                if score>newscore:
                    newscore = copy.deepcopy(score)
                    bestdata = copy.deepcopy(temp)
        data = copy.deepcopy(bestdata)
    newscore = copy.deepcopy(oldscore)
    return(bestdata, newscore)

def HCR_algorithm(data, score_type = 'bic', max_iteration=1000): 
    dataXY = data
    scoreXY = HCR(dataXY, score_type, max_iteration)[1]
    dataYX = data[[1,0]]
    dataYX.columns = range(dataYX.shape[1])
    scoreYX = HCR(dataYX, score_type, max_iteration)[1]
    return scoreXY, scoreYX
    
    
    