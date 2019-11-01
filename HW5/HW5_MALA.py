#python 3 file created by Choi, Seokjun

#using Metropolis-Adjusted Langevin Algorithm (MALA),
#get samples!


import time
from math import log
from random import uniform, normalvariate, seed
from statistics import mean
from os import getpid
import multiprocessing as mp

import matplotlib.pyplot as plt


class MC_MALA_1dim:
    def __init__(self, log_target_pdf, derivative_of_log_target, sigma_square, data, initial):
        self.log_target_pdf = log_target_pdf #arg (smpl)
        self.derivative_of_log_target = derivative_of_log_target #arg (smpl)
        #caution : it means grad(log(f(x)))!! (not log(grad(f(x))))

        self.sigma_square = sigma_square
        self.sigma = sigma_square**0.5

        self.data = data
        self.initial = initial
        
        self.MC_sample = [initial]

        self.num_total_iters = 0
        self.num_accept = 0
        
    def proposal(self, last):
        discretizer = normalvariate(0,1)
        proposing_state = last + 0.5 * self.sigma_square * self.derivative_of_log_target(last) + self.sigma * discretizer
        return proposing_state


    def log_r_calculator(self, candid, last):
        log_r = (
            self.log_target_pdf(candid) - self.log_target_pdf(last)
            - (last - candid - 0.5*self.sigma_square*self.derivative_of_log_target(candid))**2 / (2*self.sigma_square)
            + (candid - last - 0.5*self.sigma_square*self.derivative_of_log_target(last))**2 / (2*self.sigma_square)
        )
        return log_r

    def sampler(self):
        last = self.MC_sample[-1]
        candid = self.proposal(last) #기존 state 집어넣게
        unif_sample = uniform(0, 1)
        log_r = self.log_r_calculator(candid, last)
        # print(candid, log(unif_sample), log_r) #for debug
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
            if i%10000 == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%10000 == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        if pid is not None and verbose: #여기 verbose 추가함
            print("pid:",pid, "iteration", num_samples, "/", num_samples, 
            " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif pid is None and verbose: #여기 verbose 추가함
            print("iteration", num_samples, "/", num_samples, 
            " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")

    def burnin(self, num_burn_in):
        self.MC_sample = self.MC_sample[num_burn_in-1:]

    def thinning(self, lag):
        self.MC_sample = self.MC_sample[::lag]

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

    def show_traceplot(self, show=True):
        plt.plot(range(len(self.MC_sample)),self.MC_sample)
        if show:
            plt.show()

    def show_hist(self):
        plt.hist(self.MC_sample, bins=100)
        plt.show()
    
    def show_acf(self, maxLag, show=True):
        grid = [i for i in range(maxLag+1)]
        acf = self.get_autocorr(maxLag)
        plt.ylim([-1,1])
        plt.bar(grid, acf, width=0.3)
        plt.axhline(0, color="black", linewidth=0.8)
        if show:
            plt.show()

    def get_sample_mean(self, startidx=None):
        if startidx is not None:
            return mean(self.MC_sample[startidx:])
        else:
            return mean(self.MC_sample)

    def get_acceptance_rate(self):
        return self.num_accept/self.num_total_iters

class MC_MALA_unif_normal_posterior_sampler(MC_MALA_1dim):
    def UM_log_target_pdf(self, mu):
        log_val = 0
        for data_val in self.data:
            log_val += (data_val - mu)**2
        return -0.5*log_val

    def UM_derivative_of_log_target(self, mu):
        deriv_val = 0
        for data_val in self.data:
            deriv_val += (data_val - mu)
        return deriv_val

    def __init__(self, sigma_square, data, initial):
        super().__init__(self.UM_log_target_pdf, 
        self.UM_derivative_of_log_target, 
        sigma_square, data, initial)


def multiproc_1unit_do_MALA(result_queue, sigma_square, data, initial, num_iter):
    func_pid = getpid()
    print("pid: ", func_pid, "start!")

    Unit_MALA_Sampler = MC_MALA_unif_normal_posterior_sampler(sigma_square, data, initial)
    
    Unit_MALA_Sampler.generate_samples(num_iter, func_pid)
    acc_rate = Unit_MALA_Sampler.get_acceptance_rate()
    result_queue.put(Unit_MALA_Sampler)
    print("pid: ", func_pid, " acc_rate:",acc_rate)

if __name__ == "__main__":
    seed(2019311252)
    data = []
    with open("c:/gitProject/statComputing2/HW5/hw5_norm.txt","r", encoding="utf8") as f:
        while(True):
            line = f.readline()
            if not line:
                break
            data.append(float(line))
    

    core_num = 8 #띄울 process 수
    num_iter = 200000 #each MCMC chain's
    
    proc_vec = []
    proc_queue = mp.Queue()

    for _ in range(core_num):
        unit_initial = uniform(-100,100) #random으로 뽑자
        unit_sigma_square = 0.03 #그냥 일괄 설정  0.01~0.03
        #feedback: 너무 작으면, 수렴되고 나서 거의 움직이질 못하게 됨
        
        
        unit_proc = mp.Process(target = multiproc_1unit_do_MALA, args=(proc_queue, unit_sigma_square, data, unit_initial, num_iter))
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
    print("exit multiprocessing")
    

    #all traceplot
    grid_column= 2
    grid_row = int(core_num/2+0.5)
    plt.figure(figsize=(5*grid_column, 3*grid_row))
    for i, chain in enumerate(mp_result_vec):
        plt.subplot(grid_row, grid_column, i+1)
        chain.show_traceplot(False)
    plt.show()

    #all acf plot
    for i, chain in enumerate(mp_result_vec):
        plt.subplot(grid_row, grid_column, i+1)
        chain.show_acf(10, False)
    plt.show()

    #all cut
    for chain in mp_result_vec:
        chain.burnin(50000)
        chain.thinning(5)

    #mean compare
    comp_mean_vec = []
    for chain in mp_result_vec:
        chain_mean = chain.get_sample_mean()
        print(chain_mean)
        comp_mean_vec.append(chain_mean)
    print("max diff: ", max(comp_mean_vec)-min(comp_mean_vec))
    print("**")
    

    # details of last chain
    OurSampler = mp_result_vec[-1]
    OurSampler.show_traceplot()
    OurSampler.show_hist()
    OurSampler.show_acf(10)
 
