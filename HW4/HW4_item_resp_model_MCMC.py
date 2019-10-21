#python 3 file created by Choi, Seokjun

#using Markov chain monte carlo,
#get samples of Item response model's parameters

#feedback
    # (1/2번 문제는 indep chain. 이건 indep chain 말고 그냥 MCMC로 뽑을것 <- 이러면 r값 계산시 proposal ratio도 넣어야함)
    # r 계산할때 likelihood*prior 가 f임

#교수님 추가 요구사항 : 
    # beta_i 는 posterior mean을 보일것
    # theta_i는 histogram 그릴것

#구현할것:
# acf plot (ㅡㅡ음 짜증)
# convergence verifier (multiprocessing?)

import re
from math import exp, log
from random import normalvariate, uniform
from statistics import mean

import matplotlib.pyplot as plt

import time
import multiprocessing as mp
import os



class MC_MH:
    def __init__(self, log_target_pdf, log_proposal_pdf, proposal_sampler, data, initial):
        self.log_target_pdf = log_target_pdf #arg (smpl)
        self.log_proposal_pdf = log_proposal_pdf #arg (from_smpl, to_smpl)
        self.proposal_sampler = proposal_sampler #function with argument (smpl)
        
        self.data = data
        self.initial = initial
        
        self.MC_sample = [initial]

        self.num_total_iters = 0
        self.num_accept = 0
        

    def log_r_calculator(self, candid, last):
        log_r = (self.log_target_pdf(candid) - self.log_proposal_pdf(from_smpl=last, to_smpl=candid) - \
             self.log_target_pdf(last) + self.log_proposal_pdf(from_smpl=candid, to_smpl=last))
        return log_r

    def sampler(self):
        last = self.MC_sample[-1]
        candid = self.proposal_sampler(last) #기존 state 집어넣게
        unif_sample = uniform(0, 1)
        log_r = self.log_r_calculator(candid, last)
        # print(log(unif_sample), log_r) #for debug
        if log(unif_sample) < log_r:
            self.MC_sample.append(candid)
            self.num_total_iters += 1
            self.num_accept += 1
        else:
            self.MC_sample.append(last)
            self.num_total_iters += 1

    def generate_samples(self, num_samples, pid=None, verbose=True):
        start_time = time.time()
        for i in range(1, num_samples):
            self.sampler()
            if i%500 == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%500 == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        print()
        elap_time = time.time()-start_time
        if pid is not None:
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec")
        else:
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec")


    def burnin(self, num_burn_in):
        self.MC_sample = self.MC_sample[num_burn_in-1:]

    def thinning(self, lag):
        self.MC_sample = self.MC_sample[::lag]


class IRM_McPost(MC_MH): #<-고쳐야함
    def IRM_proposal_sampler(self, last):
            #sd를 각 parameter마다 다르게 잡을수 있도록 tuple로 받자
            sd = self.proposal_sampler_sd
            return [normalvariate(last[i], sd[i]) for i in range(418+24)] #분산 문제?

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
        #proposal_sampler_sd : 418+24 dim iterable object

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
        plt.hist(hist_data, bins=100)
        if show:
            plt.show()
    
    def show_traceplot(self, dim_idx, show=True):
        traceplot_data = self.get_specific_dim_samples(dim_idx)
        plt.plot(range(len(traceplot_data)), traceplot_data)
        if show:
            plt.show()

#################################################################

#for multiprocessing
def multiproc_1unit_do(result_queue, prop_sd, data, initial,num_iter):
    func_pid = os.getpid()
    print("pid: ", func_pid, "start!")
    UnitMcSampler = IRM_McPost(prop_sd, data, initial)
    
    UnitMcSampler.generate_samples(num_iter, func_pid)
    UnitMcSampler.burnin(num_iter//2)
    meanvec = UnitMcSampler.get_sample_mean()
    # result_queue.put(meanvec) #아예 instance자체를 큐에넣으면안되나? 될듯 나중에해보자
    result_queue.put(UnitMcSampler)
    print("pid: ", func_pid, "meanvec head ", meanvec[0:5])



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


    core_num = 4
    num_iter = 1000 #each MCMC chain's
    prop_sd = [0.05 for _ in range(418+24)] #0.1정도에서 수렴하는거같음<아님 더작아야할듯
    
    proc_vec = []
    meanvec_queue = mp.Queue()

    #initial을 만들고 등록하자
    for i in range(core_num):
        unit_initial = [i-1.5 for _ in range(418+24)]
    
        unit_proc = mp.Process(target = multiproc_1unit_do, args=(meanvec_queue, prop_sd, data, unit_initial,num_iter))
        proc_vec.append(unit_proc)
    
    
    for unit_proc in proc_vec:
        unit_proc.start()
    
    mp_result_vec = []
    for _ in range(core_num):
        each_result = meanvec_queue.get()
        print("mp_result_vec_object:", each_result)
        mp_result_vec.append(each_result)

    for unit_proc in proc_vec:
        unit_proc.join()
    print("exit.mp")

    #뒤에뭐 히스토그램을 그리던지 traceplot을 그리던지 하셈

    
    
    # print(meanvec_queue.get()[:5])

    # # print(log_likelihood(initial))
    # OurMcSampler = IRM_McPost(prop_sd, data, initial)
    # # print(OurMcSampler.proposal_sampler())


    # # print(OurMcSampler.IRM_log_target(initial))
    # OurMcSampler.generate_samples(1000)
    # print("acc rate: ", OurMcSampler.get_acceptance_rate())
    # OurMcSampler.show_traceplot(0) #theta1
    # OurMcSampler.show_hist(0) #theta1
    # OurMcSampler.show_traceplot(417+1) #beta1
    # OurMcSampler.show_hist(417+1) #beta1

    # OurMcSampler.show_traceplot(400) #beta1
    # OurMcSampler.show_hist(400) #beta1

    # print("meanvec(after burn-in): ", OurMcSampler.get_sample_mean())
    # OurMcSampler.burnin(20000)
    # print("meanvec(after burn-in): ", OurMcSampler.get_sample_mean())