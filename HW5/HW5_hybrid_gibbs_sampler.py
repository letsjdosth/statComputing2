#python 3 file created by Choi, Seokjun

#Hybrid Gibbs sampler on Seal data

import time
from math import log, gamma
from random import seed, betavariate, expovariate
from functools import partial
from statistics import mean

import matplotlib.pyplot as plt
from numpy.random import negative_binomial
#음 negbin... 직접 만들까하다가 귀찮아서

from HW5_MC_Core import MC_MH


class Seal_MC_MH_onlylast2dim(MC_MH):
    # MH로 돌릴 parameter를 위한 MH class
    #parameter vector order : 
    #    [data              ][param0 1    ] for hyperprior's view
    # 0  1  2  3  4  5  6  7  8      9
    # N  a1 a2 a3 a4 a5 a6 a7 theta1 theta2
 

    def seal_proposal_sampler(self, last):
        #do not depend on last : indep mcmc
        proposed_thetas = [expovariate(1000) for i in range(2)]
        #문제: theta1>0 theta2>0이어야함 - > exp쓰자(이유:posterior에 gamma - exp kernel꼴이 뒤에 곱해져있음))
        return proposed_thetas

    def seal_log_proposal_pdf(self, from_smpl, to_smpl):
        #do not depend on from_smpl
        #exp쓰면 target(posterior)에서도 같이 없애버리고 proposal pdf를 구현 안 해도 되나, 코드 직관성을 위해 일단 뒀음
        logval = 0
        # logval = -sum(to_smpl)/1000 #여기가 exp(상쇄 부분)
        return logval

    def seal_log_target_pdf(self, param_vec):
        #exp쓰면 target(posterior)에서도 해당 term을 없애버리고 proposal pdf를 구현 안 해도 되나, 코드 직관성을 위해  일단 뒀음
        thetas = param_vec
        logval = 7*log(gamma(thetas[0]+thetas[1]) / (gamma(thetas[0])+gamma(thetas[1])))
        for alpha in self.data:
            logval += (thetas[0]*log(alpha) + thetas[1]*log(1-alpha))
            # logval -= (thetas[0]+thetas[1])/1000 #여기가 exp (상쇄 부분)
        return logval

    def __init__(self, data, initial):
        #proposal_sampler_sd : 418+24 dim iterable object
        super().__init__(log_target_pdf=None, log_proposal_pdf=None, proposal_sampler=None, data=data, initial=initial)
        # self, log_target_pdf, log_proposal_pdf, proposal_sampler, data, initial
        self.proposal_sampler = self.seal_proposal_sampler
        self.log_proposal_pdf = self.seal_log_proposal_pdf
        self.log_target_pdf = self.seal_log_target_pdf
    
    def get_thetas_mean(self):
        #burnin자르고 / thining 이후 쓸것
        theta1 = []
        theta2 = []
        for smpl in self.MC_sample:
            theta1.append(smpl[0])
            theta2.append(smpl[1])
        return (mean(theta1), mean(theta2))



class Seal_HybridGibbsSampler: 
    #앞 N, alpha1~alpha7은 일반적인 gibbs로, 그리고 theta1~2는 MH로 돌린다
    
    #다른데이터에도 일반적으로쓰려면 어떻게만들어야할지잘감이안온다
    #방법1: 아예 HW4의 gibbs sampler를 좀더 추상적으로 고쳐야할듯 
    # 일반 gibbs + conddist로 업데이트할 dim 과 MH쓸 dim을 밖에서 받고
    # 해당 dim을위한 fullcond iterable object(element:callable, in:now param, out:proposal sample from sampler)를 받고
    # 또 해당 dim을위한 MH update instance를 받고
    # (묶어돌릴애들은 또 어떻게 받나....)
    # 이후 sampler를 아래와같이 오버라이드

    def __init__(self, initial_val, gibbsdim_full_conditional_sampler):
        self.initial = initial_val
        self.up_to_date = list(initial_val)
        self.num_dim = len(initial_val)
        self.full_conditional_sampler = gibbsdim_full_conditional_sampler
        self.samples = [initial_val]

    def sampler(self, num_MCiter):
        new_sample = [None for _ in range(self.num_dim)]
        
        # ordinary gibbs using full-conditional distribution
        for dim_idx in range(self.num_dim-2):
            new_val = self.full_conditional_sampler[dim_idx](self.full_conditional_sampler, up_to_date=self.up_to_date)

            new_sample[dim_idx] = new_val
            self.up_to_date[dim_idx] = new_val
        
        # hybrid part using MH
        MH_object = Seal_MC_MH_onlylast2dim(data=self.up_to_date[1:8], initial=self.up_to_date[8:10]) 
        #이걸 외부에서 받게만들어야할까? (나중에고치자)
        
        #pi(theta1,theta2)는 hyperprior이고 alpha가 거기서 뽑혀나오므로
        # alpha를 (실제 전체모델상에서 data는 아니지만 MH부분상에서는 data 역할임) data자리에 집어넣자
        #(코딩방식을 여러가지로 할 수 있는데... 
        # param_vec을 떼는 작업을 어느 위치에서 처리하냐의 차이임
        # 다 파라메터로보고 data를 None을 넘긴다음 MC sampler에서 처리하느냐 아니면 그냥 위처럼 넘기느냐..
        
        MH_object.generate_samples(num_MCiter, verbose=False) #True로 두고 Mcmc 도는걸 구경할수있다(콘솔에 찍느냐 느려지므로 비추천)
        MH_object.burnin(num_MCiter//2) #일괄적으로 절반 자른다
        new_thetas = MH_object.get_thetas_mean()
        new_sample[-2:] = new_thetas
        self.up_to_date[-2:] = new_thetas

        new_sample = tuple(new_sample)
        self.samples.append(new_sample)
    
    def generate_samples(self, num_samples, num_MCiter=10000):
        start_time = time.time()
        for i in range(1, num_samples):
            self.sampler(num_MCiter=num_MCiter)
            if i%10==0:
                print("in gibbs step iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        print("in gibbs step iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec")


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
            plt.subplot(grid_row, grid_column, i+1)
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
            plt.subplot(grid_row, grid_column, i+1)
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

class FurSealPupCapRecap_FullCondSampler_with_thetas:
    #parameter vector order : 
    # 0  1  2  3  4  5  6  7  8      9
    # N  a1 a2 a3 a4 a5 a6 a7 theta1 theta2
    NumberCaptured = (30,22,29,26,31,32,35)
    NumberNewlyCaught= (30,8,17,7,9,8,5)
    r = sum(NumberNewlyCaught) #84

    def N(self, up_to_date):
        prod = 1
        for alpha in up_to_date[1:8]:
            prod *= 1-alpha
        # return negative_binomial(self.r+1, 1-prod) + self.r
        return negative_binomial(self.r, 1-prod) + self.r
    
    def a(self, up_to_date, a_idx):
        c = self.NumberCaptured[a_idx-1]
        alpha = c + up_to_date[8]
        beta = up_to_date[0] - c + up_to_date[9]
        return betavariate(alpha, beta)

    full_cond = [N]
    for i in range(1,8):
        full_cond.append(partial(a, a_idx=i))

    def __getitem__(self, index):
        return self.full_cond[index]
    
    def __len__(self):
        return len(self.full_cond)


if __name__=="__main__":
    
    seed(2019-311252)
    #ex1
    Seal_fullcond = FurSealPupCapRecap_FullCondSampler_with_thetas()
    # print(len(Seal_fullcond)) #8
    Seal_initial_values = (150, 0.1,0.1,0.1,0.1,0.1,0.1,0.1, 0.5, 0.5)
    Seal_Gibbs = Seal_HybridGibbsSampler(Seal_initial_values, Seal_fullcond)
    Seal_Gibbs.generate_samples(5000, num_MCiter=30000) #보고서쓸땐 좀 많이돌리자
    Seal_Gibbs.show_hist()
    Seal_Gibbs.show_acf(5)
    # print(Seal_Gibbs.get_sample_mean())
    
    #자르자
    Seal_Gibbs.burnin(1000)
    Seal_Gibbs.thinning(2)

    print(Seal_Gibbs.get_sample_mean())
    Seal_Gibbs.show_hist()
    Seal_Gibbs.show_acf(5)
