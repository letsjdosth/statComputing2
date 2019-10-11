#python 3 file created by Choi, Seokjun

#Gibbs sampler

from math import log
from random import seed, betavariate
from functools import partial

from numpy.random import negative_binomial
#음 직접 만들까하다가 귀찮아서


class GibbsSampler:
    def __init__(self, initial_val, full_conditional_sampler):
        if not len(initial_val)==len(full_conditional_sampler):
            raise ValueError("number of initial_val's dimension should be equal to number of conditional densities.")
        self.initial = initial_val
        self.up_to_date = list(initial_val)
        self.num_dim = len(initial_val)
        self.full_conditional_sampler = full_conditional_sampler
        self.samples = [initial_val]

    def sampler(self):
        new_sample = [None for _ in range(self.num_dim)]
        for dim_idx in range(self.num_dim):
            new_val = self.full_conditional_sampler[dim_idx](self.full_conditional_sampler, up_to_date=self.up_to_date)

            new_sample[dim_idx] = new_val
            self.up_to_date[dim_idx] = new_val
        new_sample = tuple(new_sample)
        self.samples.append(new_sample)
    
    def generate_samples(self, num_samples, num_burn_in=0):
        n = num_samples + num_burn_in
        for _ in range(1, n):
            self.sampler()
        self.samples = \
            self.samples[(len(self.samples)-num_samples):len(self.samples)]


class FurSealPupCapRecap_FullCondSampler:
    #parameter vector order : 
    # 0  1  2  3  4  5  6  7
    # N  a1 a2 a3 a4 a5 a6 a7
    NumberCaptured = (30,22,29,26,31,32,35)
    NumberNewlyCaught= (30,8,17,7,9,8,5)
    r = sum(NumberNewlyCaught) #84

    def N(self, up_to_date):
        prod = 1
        for alpha in up_to_date[1:]:
            prod *= 1-alpha
        return negative_binomial(self.r+1, 1-prod) + self.r
    
    def a(self, up_to_date, a_idx):
        c = self.NumberCaptured[a_idx-1]
        alpha = c + 0.5
        beta = up_to_date[0] - c + 0.5
        return betavariate(alpha, beta)

    full_cond = [N]
    for i in range(1,8):
        full_cond.append(partial(a, a_idx=i))

    def __getitem__(self, index):
        return self.full_cond[index]
    
    def __len__(self):
        return len(self.full_cond)

if __name__=="__main__":
    #test for neg.bin
    # negative_binomial(10)
    
    #ex4
    Seal_fullcond = FurSealPupCapRecap_FullCondSampler()
    print(len(Seal_fullcond)) #8
    Seal_initial_values = (150, 0.1,0.1,0.1,0.1,0.1,0.1,0.1)
    Seal_Gibbs = GibbsSampler(Seal_initial_values, Seal_fullcond)
    Seal_Gibbs.generate_samples(10000)
    print(Seal_Gibbs.samples[-5:-1]) #맞나? 물개쨩....90마리밖에없어?

