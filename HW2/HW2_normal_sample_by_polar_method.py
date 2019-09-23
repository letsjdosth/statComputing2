#python 3 file created by Choi, Seokjun

#get normal samples by polar method

from math import sin, cos, log, pi, sqrt
from random import uniform, seed

import matplotlib.pyplot as plt


class NormalPolarSampler:
    def __init__(self, param_mean, param_std):
        self.param_mean = param_mean
        self.param_std = param_std

    def sampler(self):
        unif1 = uniform(0,1)
        unif2 = uniform(0,1)
        
        #polar coordinate
        R = sqrt(-2*log(unif1))
        theta = 2*pi*unif2
        return [R*sin(theta)*self.param_std + self.param_mean,
             R*cos(theta)*self.param_std + self.param_mean]
    
    def get_sample(self, number_of_smpl):
        result = []
        for _ in range(0, number_of_smpl//2):
            result += self.sampler()
        
        if(number_of_smpl%2==1):
            result.append(self.sampler()[0])
        
        return result

if __name__ == "__main__":
    print('run as main')

    seed(2019-311-252)

    NormalSampler = NormalPolarSampler(0,1)
    print(NormalSampler.get_sample(5))

    plt.xlim(-4,4)
    plt.hist(NormalSampler.get_sample(10000), bins=100)
    plt.show()
