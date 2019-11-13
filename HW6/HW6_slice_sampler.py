#python 3 file created by Choi, Seokjun

#using Slice sampler,
#get standard normal samples.

from math import exp, log, sqrt
from random import uniform, choices, seed
from statistics import mean, variance
import time

import matplotlib.pyplot as plt


class SliceSampler:
    def __init__(self, target_pdf, get_interval_func, initial):
        self.target_pdf = target_pdf
        self.get_interval = get_interval_func
        self.samples = [initial]
    
    def sampler(self, last):
        #slicing
        prob_lower_bound = uniform(0, self.target_pdf(last))
        interval_list = self.get_interval(prob_lower_bound)
        
        #draw from slice
        interval_weight = [interval[1]-interval[0] for interval in interval_list]
        chosen_interval = choices(interval_list, weights=interval_weight)[0]
        # print(chosen_interval)
        new_sample = uniform(chosen_interval[0], chosen_interval[1])
        return new_sample

    def generate_samples(self, num_samples):
        start_time = time.time()
        for _ in range(num_samples):
            last = self.samples[-1]
            self.samples.append(self.sampler(last))
        elap_time = time.time()-start_time
        print("iteration", num_samples, "/", num_samples, 
            " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")

    def get_samples(self):
        return self.samples
    
    def show_hist(self):
        plt.hist(self.samples, bins=100)
        plt.show()
    
    def get_moments(self):
        return (mean(self.samples), variance(self.samples))


if __name__ == "__main__":
    seed(2019-311252)    

    def standard_normal_pdf(x):
        return exp(-0.5*(x)**2)

    def standard_normal_get_interval(prob):
        upper_bound = sqrt(-2*log(prob))
        lower_bound = -upper_bound
        return [(lower_bound, upper_bound)]

    StdNormal_SliceSampler = SliceSampler(standard_normal_pdf, standard_normal_get_interval, 0)
    StdNormal_SliceSampler.generate_samples(200000)
    print("sample mean and variance:", StdNormal_SliceSampler.get_moments())
    StdNormal_SliceSampler.show_hist()
    
