import numpy as np
import copy
import math


class UDC:
    
    def __init__(self, data):
        self.instances = data.values.tolist()
        self.states = self.states_variables(data)
        NN = self.count_matrix(data)
        self.N = [NN, NN.transpose()]
        self.N_xy = [self.count_variable(data, 0), self.count_variable(data, 1)]
        self.cause = 0  #comment this to run find_best_direction
        self.effect = 1 #comment this to run find_best_direction
    
    def states_variables(self, data): #returns the states of each variable X and Y
        rs= [sorted(data[column].unique()) for column in list(data.columns)]
        columns= [i for i in range(len(data.columns))]
        states = dict(zip(columns, rs))
        return states
    
    def count_matrix(self, data): #return the matrix of counts of each X=x and Y=y
        matrix = np.zeros((len(self.states[0]), len(self.states[1])))
        for i in range(len(self.states[0])):
            for j in range(len(self.states[1])):
                for instance in self.instances:
                    if instance[0]==i and instance[1]==j:
                        matrix[i][j]+=1
        matrix = matrix + 0.001
        return matrix
    
    def count_variable(self, data, i): #return the counts of X=x or Y=y
        counts=[]
        for x in range(len(self.states[i])):
            count_val=0
            for instance in self.instances:
                if instance[i]==x:
                    count_val +=1
            counts.append(count_val)
        dep = [j for j in [0,1] if j!= i][0]
        counts = [x + len(self.states[dep])*0.001 for x in counts]
        return counts
    
    def permutation_sortN(self, mat): #returns the permutations that sort each matrix line
        perm= []
        for i in range(1, len(mat)):
            lst = np.argsort(mat[i,:])
            perm.append(lst)
        return perm
    
    def cond_probabilities(self, data): #returns the matrix of cond probabilities
        ind = self.cause
        dep = self.effect
        cond_prob = np.zeros((len(self.states[ind]), len(self.states[dep])))
        N = self.N[ind]
        N_ind = self.N_xy[ind]
        for i in range(len(self.states[ind])):
            for j in range(len(self.states[dep])):
                if N_ind[i]!=0:
                    cond_prob[i][j] = N[i][j] / N_ind[i] 
        return cond_prob
    
   
    def uniformly_dispersive(self, data):
        #preprocessing
        ind = self.cause
        dep = self.effect
        N = self.N[ind]
        rho = self.permutation_sortN(N)
        N_instances = len(data)
        tau1 = [i for i in range(len(self.states[dep]))]
        #initialization
        gamma = self.cond_probabilities(data)[0,:]
        eta_new = np.argsort(gamma)
        eta = np.zeros(len(self.states[dep]))
        while not np.array_equal(eta, eta_new):
            eta = copy.deepcopy(eta_new)
            tau = [tau1]
            inverse_eta = np.argsort(eta)
            for x in range(1, len(N)):
                taux = [rho[x-1][inverse_eta[k]] for k in range(len(self.states[dep]))]
                tau.append(taux)
            #update gamma
            for y in range(len(self.states[dep])):
                sum_N = 0
                for xx in range(len(self.states[ind])):
                    sum_N += N[xx][tau[xx][y]]
                gamma_y = sum_N / N_instances
                gamma[y]= gamma_y
            eta_new = np.argsort(gamma)
        return gamma, tau    
    
    def cyclic_permutations(self, lst): #returns all cyclic permutations of a list
        perm =[lst]
        for k in range(1, len(lst)):
            p = lst[k:] + lst[:k]
            if p == lst:
                break
            else:
                perm.append(p)
        return perm
       
    def cyclic_uniformly_dispersive(self, data):
        ind = self.cause
        dep = self.effect
        N = self.N[ind]
        N_instances = len(data)
        cyclic_perm = self.cyclic_permutations([k for k in range(len(self.states[dep]))])
        tau1 = [i for i in range(len(self.states[dep]))]
        gamma = self.cond_probabilities(data)[0,:]
        eta_new = np.argsort(gamma)
        eta = np.zeros(len(self.states[dep]))
        while not np.array_equal(eta, eta_new):
            eta = copy.deepcopy(eta_new)
            tau = [tau1]
            for x in range(1, len(N)):
                total_sums = []
                for perm in cyclic_perm:
                    summ= 0 
                    for y in range(len(N[0])):
                        if gamma[y]!=0:
                            summ += N[x][perm[y]]* math.log2(gamma[y])
                    total_sums.append(summ)
                max_index = total_sums.index(max(total_sums))
                taux = cyclic_perm[max_index]
                tau.append(taux)
            #update gamma
            for yy in range(len(self.states[dep])):
                sum_N = 0
                for xx in range(len(self.states[ind])):
                    sum_N += N[xx][tau[xx][yy]]
                gamma_y = sum_N / N_instances
                gamma[yy]= gamma_y
            eta_new = np.argsort(gamma)
            #print(self.LL(self.symmetric_condprob(gamma, tau)))
        return gamma, tau    
    
    def UDC_condprob(self, gamma, perm): 
        #returns the uniformly dispersive cond. probability matrix given gamma and all rows permutations
        aux_matrix = np.tile(gamma, (len(perm), 1))
        for i in range(1, len(perm)):
            aux_matrix[i, perm[i]] = aux_matrix[i, perm[0]]
        return aux_matrix  
    
    def LL(self, parameters): #calculate loglikelihood given parameters
        ind = self.cause
        dep = self.effect
        N = self.N[ind]
        ll= 0 
        for x in range(len(self.states.keys())):
            if x == dep:
                for i in range(len(N)):
                    for j in range(len(N[0])):
                        if N[i][j]!=0 and parameters[i][j]!=0:
                            ll+= N[i][j]*math.log2(parameters[i][j])
        return ll
            
    
    def estimate_UDC(self, data, var_type):#returns the UDC given data and type 
        dep = self.effect
        if var_type[dep] == 'cyclic':
            channel = self.cyclic_uniformly_dispersive(data)
        else: 
            channel = self.uniformly_dispersive(data)
        gamma = channel[0]
        perm = channel[1]
        uniform_condprob = self.UDC_condprob(gamma, perm)
        return uniform_condprob
        
    def find_best_direction(self, data, type_variables):
        ###X->Y###########
        self.cause = 0
        self.effect = 1
        LLa1 = self.LL(self.cond_probabilities(data))
        uniform_condprob = self.estimate_UDC(data, type_variables)
        LLa2 = self.LL(uniform_condprob)
        Da = abs(LLa1 - LLa2)/abs(LLa1)
        ##Y->X#################
        self.cause = 1
        self.effect = 0
        LLb1 = self.LL(self.cond_probabilities(data))
        uniform_condprob1 = self.estimate_UDC(data, type_variables)
        LLb2 = self.LL(uniform_condprob1)
        Db = abs(LLb1 - LLb2)/abs(LLb1)
        return Da, Db
    