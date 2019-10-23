#python 3 file created by Choi, Seokjun

#using "independent" Markov chain monte carlo,
#draw posterior samples


from math import log, exp, pi, factorial, tan
from random import seed, normalvariate, uniform
from statistics import mean
import multiprocessing as mp
import os

import matplotlib.pyplot as plt

from HW4_MC_Core import MC_MH

class MC_MH_1dim_withUtil(MC_MH):
    def get_autocorr(self, maxLag):
        y = self.MC_sample
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

    def show_traceplot(self):
        plt.plot(range(len(self.MC_sample)),self.MC_sample)
        plt.show()

    def show_hist(self):
        plt.hist(self.MC_sample, bins=100)
        plt.show()
    
    def show_acf(self, maxLag):
        grid = [i for i in range(maxLag+1)]
        acf = self.get_autocorr(maxLag)
        plt.ylim([-1,1])
        plt.bar(grid, acf, width=0.3)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.show()


    def get_sample_mean(self, startidx=None):
        if startidx is not None:
            return mean(self.MC_sample[startidx:])
        else:
            return mean(self.MC_sample)

    def get_acceptance_rate(self):
        return self.num_accept/self.num_total_iters



class EX1(MC_MH_1dim_withUtil):
    def EX1_proposal_sampler(self, last):
        #do not depend last
        param_mu=4
        param_sigma=0.5 
        return exp(normalvariate(log(param_mu), param_sigma))

    def EX1_log_proposal_pdf(self, from_smpl, to_smpl):
        # Indep MCMC: do not depend on from_smpl
        # prior를 proposal로 쓰기때문에, r 계산시 proposal과 target의 prior가 나눠져서 날아감. 할필요가없음
        #(거꾸로 여길 구현하면, prior도 구현해야함)
        # param_mu=4
        # param_sigma=0.5
        # x=to_smpl
        # const = 1/(x*param_sigma*((2*pi)**0.5))
        # ker = exp(-(log(x)-param_mu)**2/(2*(param_sigma**2)))
        # return log(const*ker)
        return 0

    def EX1_log_likelihood(self, param_lambda):
        log_likelihood_val = 0
        for val in self.data:
            log_likelihood_val += log((param_lambda**val)*exp(-param_lambda)/factorial(val))
        return log_likelihood_val
    
    def EX1_log_prior(self, param_vec):
        #When we calculate log_r(MH ratio's log value), just canceled.
        #so we do not need implement this term.
        return 0

    def EX1_log_target(self, param_vec):
        return self.EX1_log_likelihood(param_vec) + self.EX1_log_prior(param_vec)
  

    def __init__(self, data, initial):
        #proposal_sampler_sd : 418+24 dim iterable object
        super().__init__(log_target_pdf=None, log_proposal_pdf=None, proposal_sampler=None, data=data, initial=initial)
        # self, log_target_pdf, log_proposal_pdf, proposal_sampler, data, initial
        self.proposal_sampler = self.EX1_proposal_sampler
        self.log_proposal_pdf = self.EX1_log_proposal_pdf
        self.log_target_pdf = self.EX1_log_target


#from HW2,(좀 개조하긴함. __call__부분)
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
    
    def __call__(self):
        return self.sampler()


class EX2(MC_MH_1dim_withUtil):
    def EX2_proposal_sampler(self, last):
        #do not depend last
        param_loc=0
        param_sigma=1
        EX2_proposal_sampler = CauchySampler(param_loc, param_sigma)
        return EX2_proposal_sampler()

    def EX2_log_proposal_pdf(self, from_smpl, to_smpl):
        # Indep MCMC: do not depend on from_smpl
        # prior를 proposal로 쓰기때문에, r 계산시 proposal과 target의 prior가 나눠져서 날아감. 할필요가없음
        #(거꾸로 여길 구현하면, prior도 구현해야함)
        return 0

    def EX2_log_likelihood(self, param_mu):
        param_sigma = 1
        log_likelihood_val = 0
        for val in self.data:
            const = 1/(param_sigma*((2*pi)**0.5))
            ker = exp(-(val-param_mu)**2/(2*(param_sigma**2)))
            pdfval = const*ker
            if pdfval==0: #cauchy가 너무 꼬리가길어서 튄점에서 underflow나서 0나옴 log(0)==끔찍
                pdfval = 0.00000000000000001 #python 64bit minimum value(>0)(before floating point expression)
            log_likelihood_val += log(pdfval)
        return log_likelihood_val
    
    def EX2_log_prior(self, param_vec):
        #When we calculate log_r(MH ratio's log value), just canceled.
        #so we do not need implement this term.
        return 0

    def EX2_log_target(self, param_vec):
        return self.EX2_log_likelihood(param_vec) + self.EX2_log_prior(param_vec)
  

    def __init__(self, data, initial):
        #proposal_sampler_sd : 418+24 dim iterable object
        super().__init__(log_target_pdf=None, log_proposal_pdf=None, proposal_sampler=None, data=data, initial=initial)
        # self, log_target_pdf, log_proposal_pdf, proposal_sampler, data, initial
        self.proposal_sampler = self.EX2_proposal_sampler
        self.log_proposal_pdf = self.EX2_log_proposal_pdf
        self.log_target_pdf = self.EX2_log_target



#for multiprocessing
#each proc's details are adjusted after see 1 test chain. (see main part's code first)
def multiproc_1unit_do_EX1(result_queue, data, initial, num_iter):
    func_pid = os.getpid()
    print("pid: ", func_pid, "start!")
    UnitMcSampler = EX1(data, initial)
    
    UnitMcSampler.generate_samples(num_iter, func_pid)
    UnitMcSampler.burnin(num_iter//2) #걍 반 자르자 파라메터로 또 받기 귀찮
    acc_rate = UnitMcSampler.get_acceptance_rate()
    result_queue.put(UnitMcSampler)
    print("pid: ", func_pid, " acc_rate:",acc_rate)

def multiproc_1unit_do_EX2(result_queue, data, initial, num_iter):
    func_pid = os.getpid()
    print("pid: ", func_pid, "start!")
    UnitMcSampler = EX2(data, initial)
    
    UnitMcSampler.generate_samples(num_iter, func_pid)
    UnitMcSampler.burnin(num_iter//2) #여기도..반자르자..
    UnitMcSampler.thinning(80) #하나돌려서그려보니까 autocorr이 잘 안죽음
    acc_rate = UnitMcSampler.get_acceptance_rate()
    result_queue.put(UnitMcSampler)
    print("pid: ", func_pid, " acc_rate:",acc_rate)



if __name__ == "__main__":
    seed(2019311252)
    #ex1
    ex1_data = (8,3,4,3,1,7,2,6,2,7)     
    
    Pois_Lognorm_model = EX1(ex1_data, 10)
    Pois_Lognorm_model.generate_samples(30000)
    print("acceptance rate: ", Pois_Lognorm_model.get_acceptance_rate())
    Pois_Lognorm_model.show_traceplot()
    Pois_Lognorm_model.show_hist()
    Pois_Lognorm_model.show_acf(15)

    Pois_Lognorm_model.burnin(3000) #짧게잘라도될듯
    Pois_Lognorm_model.thinning(6)
    Pois_Lognorm_model.show_traceplot()
    Pois_Lognorm_model.show_hist()
    Pois_Lognorm_model.show_acf(15)

    ########verify convergence with random initial points
    core_num = 8 #띄울 process 수
    EX1_num_iter = 30000 #each MCMC chain's
    
    EX1_proc_vec = []
    EX1_proc_queue = mp.Queue()

    for _ in range(core_num):
        unit_initial = uniform(0,100) #random으로 뽑자
        unit_proc = mp.Process(target = multiproc_1unit_do_EX1, args=(EX1_proc_queue, ex1_data, unit_initial, EX1_num_iter))
        EX1_proc_vec.append(unit_proc)
    
    
    for unit_proc in EX1_proc_vec:
        unit_proc.start()
    
    EX1_mp_result_vec = []
    for _ in range(core_num):
        each_result = EX1_proc_queue.get()
        # print("EX1_mp_result_vec_object:", each_result)
        EX1_mp_result_vec.append(each_result)

    for unit_proc in EX1_proc_vec:
        unit_proc.join()
    print("exit multiprocessing")
    
    print("**EX1: compare mean vec**")
    EX1_comp_mean_vec = []
    for chain in EX1_mp_result_vec:
        chain_mean = chain.get_sample_mean()
        print(chain_mean)
        EX1_comp_mean_vec.append(chain_mean)
    print("max diff: ", max(EX1_comp_mean_vec)-min(EX1_comp_mean_vec))
    print("**")
    
    ####################################
    
    ########################################################################
    #ex2
    ex2_data = (2.983, 1.309, 0.957, 2.16, 0.801, 1.747, -0.274, 1.071, 2.094, 2.215,
        2.255, 3.366, 1.028, 3.572, 2.236, 4.009, 1.619, 1.354, 1.415, 1.937)
    Norm_Cauchy_model = EX2(ex2_data, 10)
    Norm_Cauchy_model.generate_samples(num_samples=100000)
    print("acceptance rate: ", Norm_Cauchy_model.get_acceptance_rate())
    Norm_Cauchy_model.show_traceplot()
    Norm_Cauchy_model.show_hist()
    Norm_Cauchy_model.show_acf(100) #음 문제가많군
    Norm_Cauchy_model.burnin(3000)
    Norm_Cauchy_model.thinning(80) #과감하게 자릅시다
    Norm_Cauchy_model.show_traceplot()
    Norm_Cauchy_model.show_hist()
    Norm_Cauchy_model.show_acf(100)



    ########verify convergence with random initial points
    core_num = 8 #띄울 process 수
    EX2_num_iter = 100000 #each MCMC chain's
    
    EX2_proc_vec = []
    EX2_proc_queue = mp.Queue()

    for _ in range(core_num):
        unit_initial = uniform(0,100) #random으로 뽑자
        unit_proc = mp.Process(target = multiproc_1unit_do_EX2, args=(EX2_proc_queue, ex2_data, unit_initial, EX2_num_iter))
        EX2_proc_vec.append(unit_proc)
    
    
    for unit_proc in EX2_proc_vec:
        unit_proc.start()
    
    EX2_mp_result_vec = []
    for _ in range(core_num):
        each_result = EX2_proc_queue.get()
        # print("EX2_mp_result_vec_object:", each_result)
        EX2_mp_result_vec.append(each_result)

    for unit_proc in EX2_proc_vec:
        unit_proc.join()
    print("exit multiprocessing")
    
    print("**EX2: compare mean vec**")
    EX2_comp_mean_vec = []
    for chain in EX2_mp_result_vec:
        chain_mean = chain.get_sample_mean()
        print(chain_mean)
        EX2_comp_mean_vec.append(chain_mean)
    print("max diff: ", max(EX2_comp_mean_vec)-min(EX2_comp_mean_vec))
    print("**")
    
    ####################################