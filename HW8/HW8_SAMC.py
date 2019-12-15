#python 3 file created by Choi, Seokjun

#using SAMC,
#get standard normal samples.

from math import log, exp, pi
from random import normalvariate, uniform, seed
import time
from statistics import mean
# import multiprocessing as mp
# import os

import matplotlib.pyplot as plt


class MC_StochasticApproximation:
    def gain_factor_generator(self):
        t0 = 100000 #should be >1
        xi = 0.9 #shouwld be 0.5 < xi <= 1
        iter_num = 0
        while(True):
            iter_num += 1
            gain_factor = t0 / max(t0, iter_num**xi)
            yield gain_factor

    def __init__(self, log_target_pdf, partition_indicator, set_visiting_freq, start_const_on_exp_vec, proposal_sigma, initial):
        # code variable name : Lecturnote notation 
        
        self.dim = len(initial)
        self.log_target_pdf = log_target_pdf #function. if input data point, return log-pdf value
        

        self.partition_indicator = partition_indicator #function. if input data point, then return i (index of partition)
        self.set_visiting_freq = set_visiting_freq #vector. for each partition i. (trivially, 1/n, n=#of partitions)

        self.proposal_sigma = proposal_sigma
        self.normalizing_const_exponent = [tuple(start_const_on_exp_vec)] # vector : theta
        self.MC_sample  = [tuple(initial)]
        self.MC_visiting_idx = [self.partition_indicator(initial)]
        
        self.gain_factor_seq = self.gain_factor_generator()

        self.num_total_iters = 0
        self.num_accept = 0
        self.pid = None
        
    def proposal_sampler(self, last):
        return normalvariate(last, self.proposal_sigma)
    
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
        now_gain_factor = next(self.gain_factor_seq)
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
            self.MC_visiting_idx.append(proposal_partition_idx)
            self.num_accept += 1
        else :
            self.MC_sample.append(tuple(last_sample_point))
            self.MC_visiting_idx.append(last_partition_idx)
        
        #Weight updating step
        new_theta = self.Weight_updating_step(proposal_partition_idx)
        #알고리즘상엔 여기서 theta 바로 넣지말고 이게 유효한 theta 범위인지 검사해야함
        #알수없는부분: 뭐가 유효한 범위냐??
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


class SAMC_withUtil_2dim(MC_StochasticApproximation):
    def __init__(self, log_target_pdf, partition_indicator, set_visiting_freq, start_const_on_exp_vec, proposal_sigma, initial):
        super().__init__(log_target_pdf, partition_indicator, set_visiting_freq, start_const_on_exp_vec, proposal_sigma, initial)
    
    #여기있으면 안되는 함수임, 옮길 것! (인자로 받게하던가 해라...)
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
        plt.plot(x_vec, y_vec, '-o', marker=".")
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
        plt.hist(self.MC_visiting_idx)
        if show:
            plt.show()


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

def test_partition_indicator1(data_point):
    if data_point[0]<0: #이렇게 나눠놓으면 0.5, 0.5임ㅋㅋ 음 근데 이렇게 하면 안될듯 넘 허접스러워서 (아니근데 잘동작하잖아?!)
        return 0
    else:
        return 1

def test_partition_indicator2(data_point):
    if data_point[0]<-2: #ㅋㅋㅋ
        return 0
    elif data_point[0]<2:
        return 1
    else:
        return 2

if __name__=="__main__":
    seed(2019311252)
    #args 순서: log_target_pdf, partition_indicator, set_visiting_freq, start_const_on_exp_vec, proposal_sigma, initial
    SAMCchain1 = SAMC_withUtil_2dim(log_target_pdf = mixture_log_pdf,
                                partition_indicator = test_partition_indicator1,
                                set_visiting_freq = (0.5, 0.5),
                                start_const_on_exp_vec = (0.5, 0.5), #여기가 문제임
                                proposal_sigma = 3,
                                initial = (0,0)
                                )
    SAMCchain1.generate_samples(300000)
    print(SAMCchain1.get_accept_rate())
    
    setting_contourplot()
    SAMCchain1.show_scatterplot()
    SAMCchain1.show_visiting_idx_hist()
    

    SAMCchain2 = SAMC_withUtil_2dim(log_target_pdf = mixture_log_pdf,
                                partition_indicator = test_partition_indicator2,
                                set_visiting_freq = (0.3, 0.4, 0.3),
                                start_const_on_exp_vec = (0.3, 0.4, 0.3), #여기가 문제임
                                proposal_sigma = 3,
                                initial = (0,0)
                                )
    SAMCchain2.generate_samples(300000)
    print(SAMCchain2.get_accept_rate())
    
    setting_contourplot()
    SAMCchain2.show_scatterplot()
    SAMCchain2.show_visiting_idx_hist()