##GENERATE DATA
import pandas as pd
import numpy as np
import random
import networkx as nx


class DataGenerator:
    def __init__(self):
        self.graph = nx.DiGraph({0:[1],1:[]})        
             
    # Generates parameters for Graph g; Each attribute has s states.     
    def gen_probabilities(self, s):
        #save P(X|Y)
        h={}
        for i in range(len(self.graph.nodes)):
            if len(list(self.graph.predecessors(i)))>0:
                q = 1
                for x in list(self.graph.predecessors(i)):
                    q*= s[x]
                for j in range(q):
                    sum_value = 0
                    for k in range(s[i]):
                        t = random.random()
                        l=(i,j,k)
                        h[l] = t
                        sum_value+=t
                    for k in range(s[i]):
                        l=(i,j,k)
                        f = h[l]/ sum_value
                        h[l] = f
            else:
                sum_value2= 0 
                for k in range(s[i]):
                    t = random.random()
                    l=(i,k)
                    h[l] = t
                    sum_value2+=t
                for k in range(s[i]):
                    l=(i,k)
                    f = h[l]/ sum_value2
                    h[l]= f
        return h
 
    def generate_UDC(self, s):
        #matrix of conditional probabilities p(Y|X)
        matrix = np.zeros((s[0], s[1]))
        cond_prob =[]
        marg_prob=[]
        probabilities={}
        for i in range(s[1]):
            cond_prob.append(random.random())
        for x in range(s[0]):
            marg_prob.append(random.random())
        #generate symmetric matrix
        cond_prob = cond_prob/np.sum(cond_prob)
        matrix[0,:] = cond_prob
        for j in range(1, s[0]):
            new_line = np.random.permutation(cond_prob)
            matrix[j,:]= new_line
        while (matrix == matrix[0]).all():
            for m in range(1, len(matrix)):
                matrix[m,:] = np.random.permutation(cond_prob)
        #generate probabilities for X
        marg_prob = marg_prob/np.sum(marg_prob)
        #generate dictionary with all probabilities
        for k in range(len(marg_prob)):
            l=(0, k)
            probabilities[l] = marg_prob[k]
            for p in range(len(matrix[0])):
                ll= (1, k, p)
                probabilities[ll] = matrix[k][p]
        return probabilities, matrix
        
     
    def cyclic_permutations(self, lst): #returns all cyclic permutations of a list
        perm =[lst]
        for k in range(1, len(lst)):
            p = lst[k:] + lst[:k]
            if p == lst:
                break
            else:
                perm.append(p)
        return perm
    
    def generate_CUDC(self, s):
        matrix = np.zeros((s[0], s[1]))
        cond_prob =[]
        marg_prob=[]
        probabilities={}
        for i in range(s[1]):
            cond_prob.append(random.random())
        for x in range(s[0]):
            marg_prob.append(random.random())
        #generate symmetric matrix
        cond_prob = cond_prob/np.sum(cond_prob)
        matrix[0,:] = cond_prob
        for j in range(1, s[0]):
            cyclic_perm = self.cyclic_permutations(list(cond_prob))
            new_line = random.choice(cyclic_perm)
            matrix[j,:]= np.array(new_line)
        while (matrix == matrix[0]).all():
            for m in range(1, len(matrix)):
                matrix[m,:] = random.choice(cyclic_perm)
        #generate probabilities for X
        marg_prob = marg_prob/np.sum(marg_prob)
        #generate dictionary with all probabilities
        for k in range(len(marg_prob)):
            l=(0, k)
            probabilities[l] = marg_prob[k]
            for p in range(len(matrix[0])):
                ll= (1, k, p)
                probabilities[ll] = matrix[k][p]
        return probabilities, matrix
        
    #Samples n instances from Graph G with parameters h 
    def getinstances(self, n, s):
        G = self.graph
        uniform = self.generate_UDC(s) ##if we want to generate a CUDC switch to self.generate_CUDC
        h = uniform[0]
        instances=[]
        for m in range(n):
            instance = [0 for x in range(len(G.nodes))]
            for i in range(len(G.nodes)):
                if len(list(G.predecessors(i)))==0:
                    f = random.random()
                    summ = 0
                    for p in range(s[i]):
                        l=(i,p)
                        prob = h[l]
                        if summ <= f and f<= summ+prob:
                            instance[i]=p
                            break
                        summ+= prob
            for k in range(len(G.nodes)):
                if len(list(G.predecessors(k)))!=0:
                    value_j = instance[list(G.predecessors(k))[0]]
                    for j in range(1, len(list(G.predecessors(k)))):
                        value_j *= 2
                        value_j+= instance[list(G.predecessors(k))[j]]
                    f= random.random()
                    summ=0
                    for p in range(s[k]):
                        l2 =(k,value_j,p)
                        prob = h[l2]
                        if summ <= f and f<= summ+prob:
                            instance[k]=p
                            break
                        summ+=prob
            instances.append(instance)
            final_data = pd.DataFrame.from_records(instances)
        return final_data, uniform[1], h