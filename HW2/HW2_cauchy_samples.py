#python 3 file created by Choi, Seokjun

#get cauchy-distributed random samples
#using inverse-cdf method


from math import tan, pi
from random import uniform, seed

import matplotlib.pyplot as plt

class CauchySampler:
    def __init__(self, param_loc, param_scale):
        if param_scale <= 0:
            raise ValueError("scale parameter should be >0")
        self.param_loc = param_loc
        self.param_scale = param_scale

    def sampler(self):
        unif_sample = uniform(0,1)
        return (self.param_scale * tan(pi * (unif_sample - 0.5)) + self.param_loc)

    def get_sample(self, number_of_smpl):
        result = []
        for _ in range(0, number_of_smpl):
            result.append(self.sampler())
        return result


if __name__ == "__main__":
    print('run as main')

    seed(2019-311-252)
    Cauchy_sampler_instance = CauchySampler(0,1)
    print(Cauchy_sampler_instance.get_sample(10))

    plt.xlim(-20,20)
    plt.hist(Cauchy_sampler_instance.get_sample(10000), bins=50000)
    plt.show()
