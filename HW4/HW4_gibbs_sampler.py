#python 3 file created by Choi, Seokjun

#Gibbs sampler

import time
from math import log
from random import seed, betavariate, normalvariate, gammavariate
from functools import partial
from statistics import mean
import re


import matplotlib.pyplot as plt
from numpy.random import negative_binomial
#음 negbin... 직접 만들까하다가 귀찮아서


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
    
    def generate_samples(self, num_samples):
        start_time = time.time()
        for i in range(1, num_samples):
            self.sampler()
            if i%10000==0:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec")


    def get_specific_dim_samples(self, dim_idx):
        if dim_idx >= self.num_dim:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.samples]
    
    def get_sample_mean(self):
        #burnin자르고 / thining 이후 쓸것
        mean_vec = []
        for i in range(self.num_dim):
            would_cal_mean = self.get_specific_dim_samples(i)
            mean_vec.append(mean(would_cal_mean))
        return mean_vec

    def show_hist(self):
        grid_column= int(self.num_dim**0.5)
        grid_row = int(self.num_dim/grid_column)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        if grid_column*grid_row < self.num_dim:
            grid_row +=1
        for i in range(self.num_dim):
            subplot_idx=str(grid_row)+str(grid_column)+str(i+1)
            plt.subplot(subplot_idx)
            dim_samples = self.get_specific_dim_samples(i)
            plt.ylabel(str(i)+"-th dim")
            plt.hist(dim_samples, bins=100)
        plt.show()

    def get_autocorr(self, dim_idx, maxLag):
        y = self.get_specific_dim_samples(dim_idx)
        acf = []
        y_mean = mean(y)
        y = [elem - y_mean  for elem in y]
        n_var = sum([elem**2 for elem in y])
        for k in range(maxLag+1):
            N = len(y)-k
            n_cov_term = 0
            for i in range(N):
                n_cov_term += y[i]*y[i+k]
            acf.append(n_cov_term / n_var)
        return acf

    def show_acf(self, maxLag):
        grid_column= int(self.num_dim**0.5)
        grid_row = int(self.num_dim/grid_column)
        if grid_column*grid_row < self.num_dim:
            grid_row +=1
        subplot_grid = [i for i in range(maxLag+1)]
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            subplot_idx=str(grid_row)+str(grid_column)+str(i+1)
            plt.subplot(subplot_idx)
            acf = self.get_autocorr(i, maxLag)
            plt.ylabel(str(i)+"-th dim")
            plt.ylim([-1,1])
            plt.bar(subplot_grid, acf, width=0.3)
            plt.axhline(0, color="black", linewidth=0.8)
        plt.show()

    def burnin(self, num_burn_in):
        self.samples = self.samples[num_burn_in-1:]

    def thinning(self, lag):
        self.samples = self.samples[::lag]

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

class BayesianSimpleReg_FullCondSampler:
    #regression model
    #population
    # y ~ Normal(b0+b1*x, sigma^2)
    #hyperprior
    # b_j ~ Normal(mu_j, tau_j^2), j=0,1
    # sigma^2 ~ Inv.Gamma(a,b)
    #parameter of hyperprior
    # tau_j=10^2, a=0.001, b=0.001, mu0=?, mu1=?

    #parameter vector order :
    # 0  1  2
    # b0 b1 sigma**2

    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y
        self.n = len(data_x)

        self.mu0 = 0
        self.mu1 = 0
        self.tau0 = 10
        self.tau1 = 10
        self.invGam_a = 0.001
        self.invGam_b = 0.001
    
    def b0(self, up_to_date):
        cond_post_precision = self.n/up_to_date[2] + 1/(self.tau0**2)
        cond_post_mu_upperpart = sum([self.y[i]-up_to_date[1]*self.x[i] for i in range(self.n)])/up_to_date[2] + self.mu0/(self.tau0**2)
        return normalvariate(cond_post_mu_upperpart/cond_post_precision, 1/cond_post_precision)

    def b1(self, up_to_date):
        cond_post_precision = sum([self.x[i]**2 for i in range(self.n)])/up_to_date[2] + 1/(self.tau1**2)
        cond_post_mu_upperpart = sum([(self.y[i]-up_to_date[0])*self.x[i] for i in range(self.n)])/up_to_date[2] + self.mu1/(self.tau1**2)
        return normalvariate(cond_post_mu_upperpart/cond_post_precision, 1/cond_post_precision)

    def inv_gamma_generator(self, param_a, param_rate):
        return 1/gammavariate(param_a, 1/param_rate)

    def sigma_square(self, up_to_date):
        #inv_gamma(a,b)
        scale = self.n/2 + self.invGam_a
        rate = sum([((self.y[i]-up_to_date[0]-up_to_date[1]*self.x[i])**2)/2 for i in range(self.n)]) + self.invGam_b
        return self.inv_gamma_generator(scale, rate)

    full_cond = [b0, b1, sigma_square]

    def __getitem__(self, index):
        return self.full_cond[index]
    
    def __len__(self):
        return len(self.full_cond)


if __name__=="__main__":
    #ex4
    Seal_fullcond = FurSealPupCapRecap_FullCondSampler()
    # print(len(Seal_fullcond)) #8
    Seal_initial_values = (150, 0.1,0.1,0.1,0.1,0.1,0.1,0.1)
    Seal_Gibbs = GibbsSampler(Seal_initial_values, Seal_fullcond)
    Seal_Gibbs.generate_samples(100000)
    Seal_Gibbs.show_hist()
    Seal_Gibbs.show_acf(5)
    
    #음 좀 자르자
    Seal_Gibbs.burnin(25000)
    Seal_Gibbs.thinning(2)

    print(Seal_Gibbs.get_sample_mean())
    Seal_Gibbs.show_hist()
    Seal_Gibbs.show_acf(5)


    #ex5
    teen_birth = []
    poverty = []
    with open("c:/gitProject/statComputing2/HW4/index.txt","r", encoding="utf8") as f:
        f.readline() #header 한번 밀어야함
        while(True):
            line = f.readline()
            split_line = re.findall(r"[\w.]+", line)
            if not split_line:
                break
            teen_birth.append(float(split_line[5]))
            poverty.append(float(split_line[1]))


    Reg_fullcond = BayesianSimpleReg_FullCondSampler(poverty, teen_birth)
    # print(len(Reg_fullcond)) #3
    Reg_initial_value = (0,0,1)
    Reg_Gibbs = GibbsSampler(Reg_initial_value, Reg_fullcond)
    Reg_Gibbs.generate_samples(100000)
    # print(Reg_Gibbs.samples[-5:-1]) #맞나??2
    Reg_Gibbs.show_hist()
    Reg_Gibbs.show_acf(30)
    
    #여기도좀자르자-_-
    Reg_Gibbs.burnin(25000)
    Reg_Gibbs.thinning(20)

    print(Reg_Gibbs.get_sample_mean())
    Reg_Gibbs.show_hist()
    Reg_Gibbs.show_acf(30)

