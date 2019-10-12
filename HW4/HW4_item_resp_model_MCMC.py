#python 3 file created by Choi, Seokjun

#using Markov chain monte carlo,
#get samples of Item response model's parameters

import re
from math import exp, log
from random import normalvariate

import matplotlib.pyplot as plt

from HW4_indep_MCMC import IndepMcPost


#proposal: normal(last, 10^2)
#구현:
# 1. normal pdf's log ver < 아 다시보니까 indep MC (기존 simed sample이 proposal에 영향안주는) proposal이었음 없어도될듯
# 2. log - likelihood (input: recent parameters)



class IRM_IndepMcPost(IndepMcPost):
    def __init__(self, data):
        self.data = data
        self.num_row = len(data) #n
        self.num_col = len(data[0]) #p
        # n*p = 418*24

        self.posterior_sample = []

        self.num_total_iters = 0
        self.num_accept = 0

    #parameters: i: rows(0~417), j:cols(0~23)
    #(theta0, theta1, ..., theta417, beta0, beta1, ..., beta23) : 418+24 dim
    def proposal_sampler(self): #이거 린터가 자꾸 지랄인데 걍 돌아감. 냅두던가 아니면 함수명 바꾸고 sampler를 또 오버라이드하던가 하자
        return [normalvariate(0,1) for _ in range(418+24)] #분산 문제?

    def log_likelihood(self, param_vec):
        theta = param_vec[:self.num_row]
        beta = param_vec[self.num_row:]
        log_likelihood_val = 0
        for i in range(self.num_row):
            for j in range(self.num_col):
                param_sum = theta[i] + beta[j]
                log_likelihood_val += self.data[i][j]*param_sum - log(1+exp(param_sum))
        return log_likelihood_val
    
    def get_specific_dim_samples(self, dim_idx):
        if dim_idx >= self.num_row+self.num_col:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.posterior_sample]



#ex3
data = []
with open("c:/gitProject/statComputing2/HW4/drv.txt","r", encoding="utf8") as f:
    while(True):
        line = f.readline()
        split_line = re.findall(r"\w", line)
        if not split_line:
            break
        split_line = [int(elem) for elem in split_line]
        data.append(split_line)

initial = [20 for _ in range(418+24)]
# print(log_likelihood(initial))
OurMcSampler = IRM_IndepMcPost(data)
# print(OurMcSampler.proposal_sampler())


# print(OurMcSampler.log_likelihood(initial))
OurMcSampler.generate_samples(initial,2000)
theta1 = OurMcSampler.get_specific_dim_samples(0)
beta1 = OurMcSampler.get_specific_dim_samples(418)
plt.plot(range(len(theta1)), theta1)
plt.show()
plt.plot(range(len(theta1)), beta1)
plt.show()

#문제: 기존 sample과 좀 비슷한데서 generate되어야 리젝 기준 r이 unif 범위 [0,1]쪽으로 떨어지는데 그러질 못하고 있음

