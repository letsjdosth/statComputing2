#python 3 file created by Choi, Seokjun

# when X~Laplace(0,1), calcualte E(X^2)
# using Importance sampling with proposal pdf N(0,2^2)

from math import exp, pi, sqrt
from random import normalvariate, seed

class ImportanceSampler:
    def __init__(self, target_pdf, proposal_pdf, proposal_sampler):
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        
        self.sample = []
        self.weight = []

    def generate(self, num_samples_for_sim):
        unstandardized_weight_sum = 0
        for _ in range(num_samples_for_sim):
            proposed_sample = self.proposal_sampler()
            unstandardized_weight = self.target_pdf(proposed_sample) / self.proposal_pdf(proposed_sample)
            self.sample.append(proposed_sample)
            self.weight.append(unstandardized_weight)
            unstandardized_weight_sum += unstandardized_weight

        self.weight = [x/unstandardized_weight_sum for x in self.weight]

    def expectation(self, inner_func):
        expectation = 0
        for val, weight in zip(self.sample, self.weight):
            expectation += inner_func(val)*weight
        
        return expectation


if __name__ == "__main__":
    print('run as main')

    seed(2019-311-252)

    def Laplace01_pdf(x):
        return 0.5*exp(-abs(x))
    
    def squarefunc(x):
        return x**2
    
    def normal02_sampler():
        return normalvariate(0,2)
    
    def normal_pdf(x, mu, sigma):
        return (exp(-0.5*(x-mu)**2/(sigma**2))/(sqrt(2*pi)*sigma))

    def normal02_pdf(x):
        return normal_pdf(x,0,2)


    LaplaceExpectation = ImportanceSampler(Laplace01_pdf, normal02_pdf, normal02_sampler)
    LaplaceExpectation.generate(10000)
    result = LaplaceExpectation.expectation(squarefunc)
    print(result) #should be near 2