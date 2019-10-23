#python 3 file created by Choi, Seokjun

#using Markov chain monte carlo,
#get samples of Item response model's parameters

#feedback
    # (1/2번 문제는 indep chain. 이건 indep chain 말고 그냥 MCMC로 뽑을것 <- 이러면 r값 계산시 proposal ratio도 넣어야함)
    # 음..그러면 r 계산할때 likelihood*prior 가 f임

#교수님 추가 요구사항 : 
    # beta_i 는 posterior mean을 보일것
    # theta_i는 histogram 그릴것


import re
import time
import os
import multiprocessing as mp

from math import exp, log
from random import normalvariate, uniform
from statistics import mean

import matplotlib.pyplot as plt

from HW4_MC_Core import MC_MH


class IRM_McPost(MC_MH):
    def IRM_proposal_sampler(self, last):
            #sd를 각 parameter마다 다르게 잡을수 있도록 tuple로 받자
            sd = self.proposal_sampler_sd
            return [normalvariate(last[i], sd[i]) for i in range(418+24)]

    def IRM_log_proposal_pdf(self, from_smpl, to_smpl):
        #When we calculate log_r(MH ratio's log value), just canceled.
        #since normal distribution is symmetric.
        #so we do not need implement this term.
        return 0

    def IRM_log_likelihood(self, param_vec):
        theta = param_vec[:self.num_row]
        beta = param_vec[self.num_row:]
        log_likelihood_val = 0
        for i in range(self.num_row):
            for j in range(self.num_col):
                param_sum = theta[i] + beta[j]
                log_likelihood_val += self.data[i][j]*param_sum - log(1+exp(param_sum))
        return log_likelihood_val
    
    def IRM_log_prior(self, param_vec):
        #When we calculate log_r(MH ratio's log value), just canceled.
        #so we do not need implement this term.
        return 0

    def IRM_log_target(self, param_vec):
        return self.IRM_log_likelihood(param_vec) + self.IRM_log_prior(param_vec)

    def __init__(self, proposal_sampler_sd, data, initial):
        #proposal_sampler_sd : 418+24=442 dim iterable object
        super().__init__(log_target_pdf=None, log_proposal_pdf=None, proposal_sampler=None, data=data, initial=initial)
        # self, log_target_pdf, log_proposal_pdf, proposal_sampler, data, initial
        self.num_row = len(data) #n
        self.num_col = len(data[0]) #p
        # n*p = 418*24

        self.proposal_sampler_sd = proposal_sampler_sd
        self.proposal_sampler = self.IRM_proposal_sampler
        self.log_proposal_pdf = self.IRM_log_proposal_pdf
        self.log_target_pdf = self.IRM_log_target


        #parameters: i: rows(0~417), j:cols(0~23)
        #(theta0, theta1, ..., theta417, beta0, beta1, ..., beta23) : 418+24 dim
    
    def get_specific_dim_samples(self, dim_idx):
        if dim_idx >= self.num_row+self.num_col:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.MC_sample]

    def get_acceptance_rate(self):
        return self.num_accept/self.num_total_iters

    def get_sample_mean(self):
        #burnin자르고 / thining 이후 쓸것
        mean_vec = []
        for i in range(self.num_row+self.num_col):
            would_cal_mean = self.get_specific_dim_samples(i)
            mean_vec.append(mean(would_cal_mean))
        return mean_vec

    def show_hist(self, dim_idx, show=True):
        hist_data = self.get_specific_dim_samples(dim_idx)
        plt.ylabel(str(dim_idx)+"th dim")
        plt.hist(hist_data, bins=100)
        if show:
            plt.show()
    
    def show_traceplot(self, dim_idx, show=True):
        traceplot_data = self.get_specific_dim_samples(dim_idx)
        plt.ylabel(str(dim_idx)+"th dim")
        plt.plot(range(len(traceplot_data)), traceplot_data)
        if show:
            plt.show()
    
    def show_betas_traceplot(self):
        grid_column= 6
        grid_row = 4
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(24):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_traceplot(418+i,False)
        plt.show()

    def show_betas_hist(self):
        grid_column= 6
        grid_row = 4
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(24):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_hist(418+i, False)
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

    def show_acf(self, dim_idx, maxLag, show=True):
        grid = [i for i in range(maxLag+1)]
        acf = self.get_autocorr(dim_idx, maxLag)
        plt.ylim([-1,1])
        plt.ylabel(str(dim_idx)+"th dim")
        plt.bar(grid, acf, width=0.3)
        plt.axhline(0, color="black", linewidth=0.8)
        if show:
            plt.show()

    def show_betas_acf(self, maxLag):
        grid_column= 6
        grid_row = 4
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(24):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_acf(418+i, maxLag, False)
        plt.show()


#################################################################

#for multiprocessing
def multiproc_1unit_do(result_queue, prop_sd, data, initial,num_iter):
    func_pid = os.getpid()
    print("pid: ", func_pid, "start!")
    UnitMcSampler = IRM_McPost(prop_sd, data, initial)
    
    UnitMcSampler.generate_samples(num_iter, func_pid)
    # UnitMcSampler.burnin(num_iter//2)
    acc_rate = UnitMcSampler.get_acceptance_rate()
    result_queue.put(UnitMcSampler)
    print("pid: ", func_pid, " acc_rate:",acc_rate)


#ex3
if __name__ == "__main__":
    
    data = []
    with open("c:/gitProject/statComputing2/HW4/drv.txt","r", encoding="utf8") as f:
        while(True):
            line = f.readline()
            split_line = re.findall(r"\w", line)
            if not split_line:
                break
            split_line = [int(elem) for elem in split_line]
            data.append(split_line)


    core_num = 8 #띄울 process 수
    num_iter = 100000 #each MCMC chain's
    prop_sd = [0.07 for _ in range(418+24)] #0.07정도에서 acc.rate가 맘에들게나옴
    
    proc_vec = []
    proc_queue = mp.Queue()

    #여기에서 initial을 만들고 process 등록
    for i in range(core_num):
        unit_initial = [uniform(-10,10) for _ in range(418+24)]
    
        unit_proc = mp.Process(target = multiproc_1unit_do, args=(proc_queue, prop_sd, data, unit_initial,num_iter))
        proc_vec.append(unit_proc)
    
    
    for unit_proc in proc_vec:
        unit_proc.start()
    
    mp_result_vec = []
    for _ in range(core_num):
        each_result = proc_queue.get()
        # print("mp_result_vec_object:", each_result)
        mp_result_vec.append(each_result)

    for unit_proc in proc_vec:
        unit_proc.join()
    print("exit.mp")

    #cheak traceplot
    for chain in mp_result_vec:
        chain.show_betas_traceplot()
    #음 그냥 맞춰주는 조합이...여기저기있나봄.....
    #아님 분포가 modal이 많거나...(빠져서못나오나?)

    
    ##########################
    OurMcSampler = mp_result_vec[-1] #마지막 체인
    print("acc rate: ", OurMcSampler.get_acceptance_rate())
    # OurMcSampler.show_traceplot(0) #theta1
    # OurMcSampler.show_hist(0) #theta1
    

    OurMcSampler.show_betas_hist()
    OurMcSampler.show_betas_traceplot()
    OurMcSampler.show_betas_acf(200)
    # print("meanvec(before burn-in): ", OurMcSampler.get_sample_mean())

    
    OurMcSampler.burnin(10000)
    OurMcSampler.thinning(200)
    OurMcSampler.show_betas_traceplot()
    OurMcSampler.show_betas_hist()
    OurMcSampler.show_betas_acf(30)
    print("mean vec(after burn-in): ", OurMcSampler.get_sample_mean())