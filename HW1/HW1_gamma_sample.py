#python 3 file created by Choi, Seokjun

#get gamma-distributed random samples
#using exponential samples achived by inverse-cdf method

from math import exp, log
from random import uniform, seed

import matplotlib.pyplot as plt


class ExponentialSampler:
    def __init__(self, param_scale):
        self.param_scale = param_scale

    def exponential_sampler(self):
        unif_sample = uniform(0,1)
        return (-self.param_scale*log(unif_sample))

    def get_exponential_sample(self, number_of_smpl):
        result = []
        for i in range(0, number_of_smpl):
            result.append(self.exponential_sampler())
        return result

class GammaSampler(ExponentialSampler):
    def __init__(self, param_alpha, param_beta):
        if param_alpha%1 != 0:
            raise ValueError("alpha should be integer")
        self.param_alpha = param_alpha
        self.param_scale = param_beta
    
    def gamma_sampler(self):
        exp_samples = self.get_exponential_sample(self.param_alpha)
        product = 1
        for smpl in exp_samples:
            product = product * smpl
        return (-1 * log(product) * self.param_scale)
    
    def get_gamma_sample(self, number_of_smpl):
        result = []
        for i in range(0, number_of_smpl):
            result.append(self.gamma_sampler())
        return result


if __name__ == "__main__":
    print('run as main')
    # EXPsampler = ExponentialSampler(0.5)
    # print(EXPsampler.get_exponential_sample(10))

    seed(2019-311-252)
    GAMMAsampler = GammaSampler(2,0.5)
    print(GAMMAsampler.get_gamma_sample(10))

    plt.hist(GAMMAsampler.get_gamma_sample(100000), bins=100)
    plt.show()
