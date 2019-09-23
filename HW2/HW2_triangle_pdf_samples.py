#python 3 file created by Choi, Seokjun

# sample from below distribution!
# our cdf:
# F(x)
#   = 0 if x<0
#   = 4x^2 if 0<x<0.25
#   = 8x/3 - 4x^2/3 -1/3 if 0.25<x<1
#   = 1 if x>1

# correspond pdf:
# f(x)
#   = 8x if 0<x<0.25
#   = -8x/3 + 8/3 if 0.25<=x<1
#   = 0 otherwise


from math import sqrt
from random import uniform, seed
from functools import partial

import matplotlib.pyplot as plt


class InvCdfSampler:
    def __init__(self, inv_cdf):
        #inv_cdf should be function
        self.inv_cdf = inv_cdf

    def sampler(self):
        unif_sample = uniform(0,1)
        return self.inv_cdf(unif_sample)

    def get_sample(self, number_of_smpl):
        result = []
        for _ in range(0, number_of_smpl):
            result.append(self.sampler())
        return result


class RejectionSampler:
    def __init__(self, target_pdf, proposal_pdf, proposal_sampler, envelope_multiplier):
        # target pdf, proposal pdf, proposal sampler must be functions.
        # <caution> "envelop_multiplier" value satisfies "envelope_multiplier * proposal_pdf > target_pdf"
        # proposal sampler should not be the function that has arguments. 
        #     just from proposal_sampler() we should able to get 1 sample
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        self.envelope_multiplier = envelope_multiplier
        
    def envelope(self, x):
        return (self.proposal_pdf(x) * self.envelope_multiplier)
    
    def sampler(self):
        while(True):
            unif_sample = uniform(0,1)
            proposal_sample = self.proposal_sampler()

            thres = self.target_pdf(proposal_sample) / self.envelope(proposal_sample)
            if thres > unif_sample:
                return proposal_sample

    def get_sample(self, number_of_smpl):
        result = []
        for _ in range(0, number_of_smpl):
            result.append(self.sampler())
        return result



def triangle_inv_cdf(y):
    if (0 <= y < 0.25):
        return (0.5 * sqrt(y))
    elif (0.25 <= y <= 1):
        return (-sqrt(-0.75 * (y - 1)) + 1)
    else:
        raise ValueError('input of inverse cdf should be 0<=y<=1')


def triangle_pdf(x):
    if (0 <= x < 0.25):
        return (8*x)
    elif (0.25 <= x < 1):
        return (-8*x/3 + 8/3)
    else:
        return 0

if __name__ == "__main__":
    print('run as main')

    seed(2019-311-252)

    #test for inv_cdf
    assert triangle_inv_cdf(0.25) == 0.25 #should be 0.25
    assert triangle_inv_cdf(1) == 1 #should be 1

    #test for pdf
    assert triangle_pdf(0.25) == 2 #should be 2

    #Inv CDF sampler
    TriangleInvCdfSampler = InvCdfSampler(triangle_inv_cdf)
    print(TriangleInvCdfSampler.get_sample(10))
    plt.xlim(0,1)
    plt.hist(TriangleInvCdfSampler.get_sample(10000), bins=100)
    plt.show()

    #Rejection sampler
    #uniform proposal, envelop = 3*uniform(0,1)
    TriangleRejectionSampler = RejectionSampler(triangle_pdf, lambda x : 1, partial(uniform,0,1), 3)
    print(TriangleRejectionSampler.get_sample(10))
    plt.hist(TriangleRejectionSampler.get_sample(10000), bins=100)
    plt.show()