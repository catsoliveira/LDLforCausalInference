##test synthetic data
import math
from synthetic_data import DataGenerator
import matplotlib.pyplot as plt
from UDC import UDC



def kl_divergence(p, q, prob): #average kl distance
    sum_value = 0
    prob_ind = []
    for elem in prob.keys():
        if len(elem)==2:
            prob_ind.append(prob[elem]) 
    for i in range(len(p)):
        summ= 0 
        for j in range(len(p[0])):
            if p[i][j]!=0 and q[i][j]!=0: 
                summ += (p[i][j] * math.log2(p[i][j]/q[i][j]) + q[i][j] * math.log2(q[i][j]/p[i][j]))/2
        sum_value += summ * prob_ind[i]     
    final = round(sum_value,3)
    return final
       

def test_performance():
    N_instances = [50,100,200,300,400,500,1000,1500,2000,2500,3000]
    support = [2,3,4,5]
    KL = {k:{} for k in support}
    for val in support:
        average_KL ={k:0 for k in N_instances} 
        for N in N_instances:
            score_KL = 0
            for i in range(50):
                generator = DataGenerator().getinstances(N, [val,val])
                data = generator[0]
                condprob_true = generator[1]
                probabilities = generator[2]
                while any(len(t)!=val for t in [data[column].unique() for column in list(data.columns)]):
                    generator = DataGenerator().getinstances(N, [val,val])
                    data = generator[0]
                    condprob_true = generator[1]
                    probabilities = generator[2]
                condprob_estimate = UDC(data).estimate_UDC(data, ['no','no'])
                score_KL += kl_divergence(condprob_true, condprob_estimate, probabilities)
                print(score_KL)
            average_KL[N] = score_KL/50
        KL[val] = average_KL
    return KL

def plot_performance():
    results = test_performance()
    print(results)
    colors = ['#C71585','#1f77b4', 'green', "#ff8533"]
    x = [50,100,200,300,400,500,1000,1500,2000,2500,3000]
    sup = [2,3,4,5]
    for support in sup:
        plt.loglog(x, list(results[support].values()), marker="o", color = colors[sup.index(support)], label=r'$ |\mathcal{X}| = |\mathcal{Y}| =$' + str(support))
    plt.xlabel(r'$ \mathit{N} $', fontsize=12)
    plt.ylabel('KL', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 11})
    plt.show()
    

def test_causality():
    N=[50,100,200,300,400,500, 1000,1500,2000,2500,3000]
    sup = [2,3,4,5]
    prob ={k:{} for k in sup}
    for val in sup:
        prob_N = {k:0 for k in N}
        for instances in N:
            times_rightdirection=0
            for i in range(100):
                generator = DataGenerator().getinstances(instances, [val,val])
                data = generator[0]
                while any(len(t)!=val for t in [data[column].unique() for column in list(data.columns)]):
                    generator = DataGenerator().getinstances(instances, [val,val])
                    data = generator[0]
                scores = UDC(data).find_best_direction(data, ['no','no']) #change to cyclic to run CUDC
                if scores[0]<scores[1]:
                    times_rightdirection+=1
            prob_right = times_rightdirection/100
            print(prob_right)
            prob_N[instances] = prob_right
        prob[val] = prob_N
    return prob

def plot_causality():
    results = test_causality()
    print(results)
    colors = ['#C71585','#1f77b4', 'green', "#ff8533"]
    x = [50,100,200,300,400,500,1000,1500,2000,2500,3000]
    sup = [2,3,4,5]
    for support in sup:
        plt.plot(x, list(results[support].values()), marker="o", color = colors[sup.index(support)], label=r'$ |\mathcal{X}| = |\mathcal{Y}| =$' + str(support))
    plt.xlabel(r'$ \mathit{N} $', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', prop={'size': 11})
    plt.show()
    

def test_causality_difsup():
    N=[50,100,200,300,400,500, 1000,1500,2000,2500,3000]
    sup = [[3,2], [2,3], [4,2], [2,4], [5,2], [2,5]]
    prob ={tuple(k):{} for k in sup}
    for val in sup:
        prob_N = {k:0 for k in N}
        for instances in N:
            times_rightdirection=0
            for i in range(100):
                generator = DataGenerator().getinstances(instances, [val[0],val[1]])
                data = generator[0] 
                col = [len(data[column].unique()) for column in list(data.columns)]
                while any(col[t]!=val[t] for t in range(len(col))):
                    generator = DataGenerator().getinstances(instances, [val[0],val[1]])
                    data = generator[0]
                    col = [len(data[column].unique()) for column in list(data.columns)]
                scores = UDC(data).find_best_direction(data, ['no','no']) #change to cyclic to run CUDC
                if scores[0]<scores[1]:
                    times_rightdirection+=1
            prob_right = times_rightdirection/100
            print(prob_right)
            prob_N[instances] = prob_right
        prob[tuple(val)] = prob_N
    return prob

def plot_causality_difsup():
    results = test_causality_difsup()
    print(results)
    x = [50,100,200,300,400,500,1000,1500,2000,2500,3000]
    sup = [[3,2], [2,3], [4,2], [2,4], [5,2], [2,5]]
    for support in sup:
        if support == [3,2] or support == [2,3]:
            col = '#C71585'
        elif support == [4,2] or support == [2,4]: 
            col = '#1f77b4'
        else:
            col = 'green'
        if support == [3,2] or support == [4,2] or support == [5,2]:
            plt.plot(x, list(results[tuple(support)].values()), marker="o", color = col, linestyle='dashed', label=r'$ |\mathcal{X}| = $' + str(support[0]) + r'$, |\mathcal{Y}| =$' + str(support[1]))
        else:
            plt.plot(x, list(results[tuple(support)].values()), marker="o", color = col, label=r'$ |\mathcal{X}| =$' + str(support[0]) + r'$, |\mathcal{Y}| =$' + str(support[1]))
    plt.xlabel(r'$ \mathit{N} $', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', prop={'size': 11})
    plt.show()
    
    
def test_performance1():
    N_instances = [50,100,200,300,400,500,1000,1500,2000,2500,3000]
    support = [[3,2], [2,3], [4,2], [2,4], [5,2], [2,5]]
    KL = {tuple(k):{} for k in support}
    for val in support:
        average_KL ={k:0 for k in N_instances} 
        for N in N_instances:
            score_KL = 0
            for i in range(50):
                generator = DataGenerator().getinstances(N, [val[0],val[1]])
                data = generator[0]
                condprob_true = generator[1]
                probabilities = generator[2]
                col = [len(data[column].unique()) for column in list(data.columns)]
                while any(col[t]!=val[t] for t in range(len(col))):
                    generator = DataGenerator().getinstances(N, [val[0],val[1]])
                    data = generator[0]
                    condprob_true = generator[1]
                    probabilities = generator[2]
                    col = [len(data[column].unique()) for column in list(data.columns)]
                print("true")
                print(condprob_true)
                condprob_estimate = UDC(data).estimate_UDC(data, ['no','no']) #change to cyclic to run CUDC
                score_KL += kl_divergence(condprob_true, condprob_estimate, probabilities)
            average_KL[N] = score_KL/50
        KL[tuple(val)] = average_KL
    return KL    

    


print(plot_causality_difsup())

