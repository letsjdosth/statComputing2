#python 3 file created by Choi, Seokjun

#using rejection sampling method, 
#sampling poisson's parameter value
#with lognormal density

from math import log, exp, factorial
from statistics import mean
import random

import matplotlib.pyplot as plt


class Lognormal_Poisson_RejectionSampler:
    def __init__(self, data):
        self.data = data
        self.data_mean = mean(data)

    def pois_pmf(self, x, param_lambda):
        if not isinstance(x, int):
            raise ValueError("x should be integer.")
        return ((param_lambda**x)*exp(-param_lambda)/factorial(x))

    def thres_p_calculator(self, lognorm_sample):
        thres_p_upper = (self.pois_pmf(x, lognorm_sample) for x in self.data) 
        thres_p_lower = (self.pois_pmf(x, self.data_mean) for x in self.data) 
        thres_p = 1
        for up, low in zip(thres_p_upper, thres_p_lower):
            thres_p = thres_p * up/low
            #~수업 note~
            #이거 underflow 날거 걱정되면
            #log씌워서 sum으로 계산하고 다시 변환해오자 (나중에 해볼것)
            #아니면 통째로 알고리즘을 다 log버전으로 돌리자(sampler를 수정. Uniform sample에 log씌우고 rejection rule)
        return thres_p
        
    def sampler(self):
        #get one sample
        while(1):
            unif_sample = random.uniform(0,1)
            lognorm_sample = exp(random.normalvariate(log(4), 0.5))
            
            thres_p = self.thres_p_calculator(lognorm_sample)

            # print('p: ', thres_p ," and now uniform r.s : ", unif_sample)
            if unif_sample < thres_p:
                # print('accepted: ', lognorm_sample)
                yield lognorm_sample
            else:
                # print('rejected')
                pass
    
    def get_sample(self, number_of_smpl):
        result = []
        for _ in range(0, number_of_smpl):
            result.append(next(self.sampler()))
        return result



if __name__ == "__main__" :
    print('run as main')
    random.seed(2019-311-252)
    given_data = (8,3,4,3,1,7,2,6,2,7)
    LPsampler = Lognormal_Poisson_RejectionSampler(given_data)
    print(LPsampler.get_sample(10))
    plt.hist(LPsampler.get_sample(100000), bins=100)
    plt.show()
