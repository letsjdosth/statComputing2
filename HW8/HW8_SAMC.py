#python 3 file created by Choi, Seokjun

#using SAMC,
#get standard normal samples.

from math import log, exp, pi
from random import normalvariate, uniform, seed
import time
from statistics import mean
import os
import multiprocessing as mp
from abc import abstractmethod

import matplotlib.pyplot as plt


class MC_StochasticApproximation:
    def __init__(self, log_target_pdf, partition_indicator, set_visiting_freq, start_const_on_exp_vec, proposal_sigma, initial):
        # code variable name : Lecturnote notation 
        
        self.initial = initial
        self.dim = len(initial)
        self.log_target_pdf = log_target_pdf #function. if input data point, return log-pdf value

        self.partition_indicator = partition_indicator #function. if input data point, then return i (index of partition)
        self.set_visiting_freq = set_visiting_freq #vector. for each partition i. (trivially, 1/n, n=#of partitions)

        self.proposal_sigma = proposal_sigma
        self.normalizing_const_exponent = [tuple(start_const_on_exp_vec)] # vector : theta
        self.MC_sample  = [tuple(initial)]
        self.MC_visiting_idx = [self.partition_indicator(initial)]
        
        self.num_total_iters = 0
        self.num_accept = 0
        self.pid = None
        
    def gain_factor(self):
        #gain factor 모양 보며 아래 parameter 2개 조정
        t0 = 2 #should be >1
        xi = 0.9 #shouwld be 0.5 < xi <= 1

        iter_num = self.num_total_iters
        gain_factor = t0 / max(t0, iter_num**xi)
        return gain_factor

    @abstractmethod    
    def proposal_sampler(self, last):
        # 차원에 맞게 multivariate normal d개 생선한 tuple 리턴
        # return normalvariate(last, self.proposal_sigma)
        pass
    
    def log_proposal_pdf(self, from_smpl, to_smpl):
        #When we calculate log_r(MH ratio's log value), just canceled.
        #since normal distribution is symmetric.
        #so we do not need implement this term.
        return 0

    def log_r_calculator(self, last, last_partition_idx, candid, candid_partition_idx):
        log_r = (self.log_target_pdf(candid) - self.log_proposal_pdf(from_smpl=last, to_smpl=candid) - \
             self.log_target_pdf(last) + self.log_proposal_pdf(from_smpl=candid, to_smpl=last))
        now_theta = self.normalizing_const_exponent[-1]
        log_r += now_theta[last_partition_idx] - now_theta[candid_partition_idx]
        return log_r

    def MH_rejection_step(self, last_sample_point, last_partition_idx, candid_sample_point, candid_partition_idx):
        unif_sample = uniform(0, 1)
        if candid_partition_idx is None:
            return False
        
        try:
            log_r = self.log_r_calculator(last_sample_point, last_partition_idx, candid_sample_point, candid_partition_idx)
        # print(log(unif_sample), log_r) #for debug
        except ValueError:
            return False
        
        if log(unif_sample) < log_r:
            return True
        else:
            return False
    
    def Weight_updating_step(self, candid_partition_idx):
        last_theta_vec = self.normalizing_const_exponent[-1]
        now_gain_factor = self.gain_factor()
        new_theta = []
        for i, theta_i in enumerate(last_theta_vec):
            new_theta.append(theta_i - now_gain_factor * self.set_visiting_freq[i])
        new_theta[candid_partition_idx] += now_gain_factor
        return new_theta

    def sampler(self):
        last_sample_point = self.MC_sample[-1]
        last_partition_idx = self.MC_visiting_idx[-1]
        proposal_sample_point = self.proposal_sampler(last_sample_point)
        proposal_partition_idx =self.partition_indicator(proposal_sample_point)

        #MH step
        accept_bool = self.MH_rejection_step(last_sample_point, last_partition_idx, proposal_sample_point, proposal_partition_idx)
        
        self.num_total_iters += 1
        if accept_bool:
            self.MC_sample.append(tuple(proposal_sample_point))
            # self.MC_visiting_idx.append(proposal_partition_idx)
            self.num_accept += 1
        else :
            self.MC_sample.append(tuple(last_sample_point))
            # self.MC_visiting_idx.append(last_partition_idx)
        
        #Weight updating step
        if proposal_partition_idx is not None:
            self.MC_visiting_idx.append(proposal_partition_idx)
            new_theta = self.Weight_updating_step(proposal_partition_idx)
            self.normalizing_const_exponent.append(new_theta)
        else:
            # proposal이 튀어나갔을시 그냥 reject하고, c* 초기화 대신 기존값 사용하게 구현함
            # (c*를 어떻게 잡아야할지...)
            new_theta = self.normalizing_const_exponent[-1] 
            self.normalizing_const_exponent.append(new_theta)

    def generate_samples(self, num_samples, verbose=True):
        start_time = time.time()
        
        for i in range(1, num_samples):
            self.sampler()
            if i%50000 == 0 and verbose and self.pid is not None:
                print("pid:",self.pid," iteration", i, "/", num_samples)
            elif i%50000 == 0 and verbose and self.pid is None:
                print("iteration", i, "/", num_samples)
        
        elap_time = time.time()-start_time
        
        if self.pid is not None and verbose:
            print("pid:",self.pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif self.pid is None and verbose:
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
    
    def burnin(self, num_burn_in):
        self.MC_sample = self.MC_sample[num_burn_in-1:]
    
    def thinning(self, lag):
        self.MC_sample = self.MC_sample[::lag]



class SAMC_withUtil_2dim(MC_StochasticApproximation):
    def __init__(self, log_target_pdf, partition_indicator, set_visiting_freq, start_const_on_exp_vec, proposal_sigma, initial):
        super().__init__(log_target_pdf, partition_indicator, set_visiting_freq, start_const_on_exp_vec, proposal_sigma, initial)
    
    #abstractmehtod override
    def proposal_sampler(self, last):
        return (normalvariate(last[0], self.proposal_sigma), normalvariate(last[1], self.proposal_sigma))


    def get_specific_dim_samples(self, dim_idx):
        #dim_idx 0부터 시작함
        if dim_idx >= self.dim:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.MC_sample]
    
    def show_scatterplot(self, show=True):
        x_vec = self.get_specific_dim_samples(0)
        y_vec = self.get_specific_dim_samples(1)
        # plt.plot(x_vec, y_vec, '-o', marker=".")
        plt.plot(x_vec, y_vec, 'o', marker=".")
        if show:
            plt.show()

    def get_accept_rate(self):
        try:
            acc_rate = self.num_accept / self.num_total_iters
        except ZeroDivisionError:
            acc_rate = 0
        return acc_rate

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
            try:
                k_lag_acf = n_cov_term / n_var
            except ZeroDivisionError:
                # raise ZeroDivisionError("n_var is too small (underflow raised.)") #전파 필요시
                k_lag_acf = 1
            acf.append(k_lag_acf)
        return acf

    def show_acf(self, dim_idx, maxLag, show=True):
        grid = [i for i in range(maxLag+1)]
        acf = self.get_autocorr(dim_idx, maxLag)
        plt.ylim([-1,1])
        plt.bar(grid, acf, width=0.3)
        plt.axhline(0, color="black", linewidth=0.8)
        if show:
            plt.show()
    
    def show_visiting_idx_hist(self, show=True):
        plt.hist(self.MC_visiting_idx, bins=10)
        if show:
            plt.show()
    
    def get_visiting_idx_count(self, max_idx):
        count_vec = [0 for x in range(max_idx+1)]
        for idx in self.MC_visiting_idx:
            count_vec[idx] += 1
        return count_vec


#our case
# (1/3)*N((0,0), I) + (1/3)*N((-8,-6), [1, 0.9; 0.9, 1]) + (1/3)*N((8,6), [1, -0.9; -0.9, 1]))
def bivariate_normal_pdf(x_2d, corr, mu_x1, mu_x2, sigmasq_1=1, sigmasq_2=1):
    x1, x2 = x_2d
    ker = (x1-mu_x1)**2/sigmasq_1 + (x2-mu_x2)**2/sigmasq_2 - 2*corr*(x1-mu_x1)*(x2-mu_x2)/(sigmasq_1*sigmasq_2)**0.5
    ker *= (-0.5)/(1-corr**2)
    const = 2*pi*(sigmasq_1*sigmasq_2*(1-corr**2))**0.5
    return exp(ker)/const

def mixture_log_pdf(x_2d):
    try:
        logpdf_val = log(
            bivariate_normal_pdf(x_2d,0,0,0)/3
            + bivariate_normal_pdf(x_2d, 0.9, -8, -6)/3 
            + bivariate_normal_pdf(x_2d, -0.9, 8, 6)/3)
    except ValueError:
        err_pdfval = (bivariate_normal_pdf(x_2d,0,0,0)/3 
            + bivariate_normal_pdf(x_2d, 0.9, -8, -6)/3 
            + bivariate_normal_pdf(x_2d, -0.9, 8, 6)/3)
        err_str = "Underflow: at "+ str(x_2d) + ", pdf value: " + str(err_pdfval) + " is too close to 0."
        raise ValueError(err_str)
    return logpdf_val


def setting_contourplot(start=-10, end=10):    
    grid = [x/10 + start for x in range(10*(end-start))]
    mixture_val = [[mixture_log_pdf([x,y]) for x in grid] for y in grid]
    plt.contour(grid, grid, mixture_val, levels=20)


def mixture_partition_indicator(data_point):
    try:
        logpdfval = mixture_log_pdf(data_point)
    except ValueError:
        return None
    
    if logpdfval > -3.5:
        return 0
    elif logpdfval > -4.5:
        return 1
    elif logpdfval > -5.6:
        return 2
    elif logpdfval > -6.7:
        return 3
    elif logpdfval > -7.9:
        return 4
    elif logpdfval > -9.3:
        return 5
    elif logpdfval > -10.8:
        return 6
    elif logpdfval > -12.7:
        return 7
    elif logpdfval > -15:
        return 8
    elif logpdfval > -17.5:
        return 9
    else:
        return None



#for multiprocessing
def multiproc_1unit_do(result_queue, initial, num_iter):
    func_pid = os.getpid()
    print("pid: ", func_pid, "start!")
    SAMCchain = SAMC_withUtil_2dim(log_target_pdf = mixture_log_pdf,
                                partition_indicator = mixture_partition_indicator,
                                set_visiting_freq = tuple([1/10 for _ in range(10)]),
                                start_const_on_exp_vec = tuple([1/10 for _ in range(10)]),
                                proposal_sigma = 3,
                                initial = initial
                                )
    SAMCchain.pid = func_pid
    SAMCchain.generate_samples(num_iter)
    SAMCchain.burnin(100000)
    SAMCchain.thinning(200)

    result_queue.put(SAMCchain)
    print("pid: ", func_pid, " end!")
    


if __name__=="__main__":
    seed(2019311252)
    core_num = 8
    #setting
    initial = [(x,x) for x in range(-4,5)]
    num_iter = 500000

    
    #generate SAMC chains using parallel multiprocessing
    print("start.mp")
    proc_vec = []
    proc_queue = mp.Queue()
    
    for i in range(core_num):
        unit_proc = mp.Process(target = multiproc_1unit_do, 
            args=(proc_queue, initial[i], num_iter,))
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


    #check traceplot
    grid_column= 8
    grid_row = 4
    plt.figure(figsize=(5*grid_column, 3*grid_row))
    for i, chain in enumerate(mp_result_vec):
        #plot 1
        plt.subplot(grid_row, grid_column, 4*i+1)
        plt.subplots_adjust(hspace=0.6)
        setting_contourplot(-12,12)
        chain.show_scatterplot(show=False)
        title_str = "initial: " + str(round(chain.initial[0],4)) + ", " + str(round(chain.initial[1],4)) \
            + "\ngenerated sample plot"
        plt.title(title_str)
        
        #plot 2
        plt.subplot(grid_row, grid_column, 4*i+2)
        title_str = "total iter num:" + str(chain.num_total_iters) \
            + "\nacceptance rate:" + str(round(chain.get_accept_rate(),5)) \
            + "\nvisiting frequency"
        plt.title(title_str)
        chain.show_visiting_idx_hist(show=False)
        
        #plot 3
        plt.subplot(grid_row, grid_column, 4*i+3)
        chain.show_acf(0,10,show=False)
        title_str = "acf of x"
        plt.title(title_str)

        #plot 4
        plt.subplot(grid_row, grid_column, 4*i+4)
        chain.show_acf(1,10,show=False)
        title_str = "acf of y"
        plt.title(title_str)
    plt.show()
