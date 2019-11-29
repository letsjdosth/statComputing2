#python 3 file created by Choi, Seokjun

#using Hamiltonian monte carlo,
#get standard normal samples.

#더 할일:
# 2. 아예 이상한 점에서 시작하기 (음 underflow 날 듯)
# 3. log(0)에 가까운경우 
# underflow 잡고 끊지말고 그냥그거 reject한담에 또 시도하도록? (<이거 conv.prop을 흔들수도 있을것같음)
# 4. method 정리 (2d전용은 아니지만 core에두기엔 애매한거 어디에둘까 생각좀해봐)

from math import log, exp, pi
from random import uniform, normalvariate, seed
from statistics import mean
import time
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MC_Hamiltonian:
    def __init__(self, log_target_pdf, log_target_grad, leapfrog_step_num, step_size, initial):
        self.dim = len(initial)
        self.log_target_pdf = log_target_pdf
        self.log_target_grad = log_target_grad
        self.leapfrog_step_num = leapfrog_step_num #L
        self.step_size = step_size #epsilon
        self.MC_sample  = [tuple(initial)]
        self.MC_momentum = [tuple([normalvariate(0, 1) for _ in range(self.dim)])]
        # self.MC_momentum = [(0,0)]
        self.momentum_stdNormPrior_cov = 1 #음 lecture note상 M인데 이건그냥 fix시키게겠음 inverse 구현이너무귀찮음
        self.momentum_stdNormPrior_cov_inv = 1

        self.HMCiter = 0
        self.HMCaccept = 0
    
    def leap_frog_step(self, start_sample_point, start_momentum):
        log_target_gradient = self.log_target_grad(start_sample_point)
        momentum = [start_momentum[i] + 0.5 * self.step_size * log_target_gradient[i] for i in range(self.dim)]
        sample_point = [start_sample_point[i] + self.step_size * self.momentum_stdNormPrior_cov_inv * momentum[i] for i in range(self.dim)]
        for _ in range(1, self.leapfrog_step_num):
            log_target_gradient = self.log_target_grad(start_sample_point)
            momentum = [momentum[i] + self.step_size * log_target_gradient[i] for i in range(self.dim)]
            sample_point = [sample_point[i] + self.step_size * self.momentum_stdNormPrior_cov_inv * momentum[i] for i in range(self.dim)]
        momentum = [start_momentum[i] + 0.5 * self.step_size * log_target_gradient[i] for i in range(self.dim)]
        return (sample_point, momentum)

    def log_normal_pdf(self, x_vec):
        #not lognormal pdf. but log(std.normal.pdf)
        #need only kernel part (not constant)
        log_kernel = -0.5 * sum([x**2 for x in x_vec])
        return log_kernel

    def log_r_calculator(self, last_sample_point, last_momentum, 
                            proposed_sample_point, proposed_momentum):
        log_r = self.log_target_pdf(proposed_sample_point) + self.log_normal_pdf(proposed_momentum) \
            - self.log_target_pdf(last_sample_point) - self.log_normal_pdf(last_momentum)
        return log_r

    def MH_rejection_step(self, last_sample_point, last_momentum, 
                            proposed_sample_point, proposed_momentum):
        unif_sample = uniform(0,1)
        # proposed_momentum = [-elem for elem in proposed_momentum]
        # last_momentum = [-elem for elem in last_momentum]
        log_HMC_ratio = self.log_r_calculator(last_sample_point, last_momentum, 
                            proposed_sample_point, proposed_momentum)
        if log(unif_sample) < log_HMC_ratio:
            return True
        else:
            return False
    
    def sampler(self):
        last_sample_point = self.MC_sample[-1]
        last_momentum = self.MC_momentum[-1]
        new_momentum = tuple([normalvariate(0, 1) for _ in range(self.dim)])
        proposal_sample_point, proposal_momentum = self.leap_frog_step(last_sample_point, new_momentum)
        accept_bool = self.MH_rejection_step(last_sample_point, last_momentum, proposal_sample_point, proposal_momentum)
        self.HMCiter += 1
        if accept_bool:
            self.MC_sample.append(tuple(proposal_sample_point))
            self.MC_momentum.append(tuple(proposal_momentum))
            self.HMCaccept += 1
        else :
            self.MC_sample.append(tuple(last_sample_point))
            self.MC_momentum.append(tuple(last_momentum))

    def generate_samples(self, num_samples, pid=None, verbose=True):
        start_time = time.time()
        for i in range(1, num_samples):
            self.sampler()
            if i%100000 == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%100000 == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        if pid is not None and verbose:
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif pid is None and verbose:
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")

class MC_Hamiltonian_withUtil_2dim(MC_Hamiltonian):
    def __init__(self, log_target_pdf, log_target_grad, leapfrog_step_num, step_size, initial):
        super().__init__(log_target_pdf, log_target_grad, leapfrog_step_num, step_size, initial)
    
    def get_specific_dim_samples(self, dim_idx):
        #0,1로 넣을 것
        if dim_idx >= 2:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.MC_sample]
    
    def show_scatterplot(self, show=True):
        # for sample_point in self.MC_sample:
        #     x, y = sample_point
        #     plt.scatter(x,y)
        x_vec = self.get_specific_dim_samples(0)
        y_vec = self.get_specific_dim_samples(1)
        # plt.scatter(x_vec,y_vec, marker=".")
        plt.plot(x_vec, y_vec, '-o', marker=".")
        if show:
            plt.show()
    
    def show_momentum(self, show=True):
        norm_momentum_vec = [(momentum[0]**2+momentum[1]**2)**0.5 for momentum in self.MC_momentum]
        plt.plot(range(len(norm_momentum_vec)),norm_momentum_vec)
        if show:
            plt.show()
    
    def get_accept_rate(self):
        try:
            acc_rate = self.HMCaccept / self.HMCiter
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
            acf.append(n_cov_term / n_var)
        return acf

    def show_acf(self, dim_idx, maxLag, show=True):
        grid = [i for i in range(maxLag+1)]
        acf = self.get_autocorr(dim_idx, maxLag)
        plt.ylim([-1,1])
        plt.bar(grid, acf, width=0.3)
        plt.axhline(0, color="black", linewidth=0.8)
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
    #어차피 unnormallized인데 1/3 안해도되지않나
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

def mixture_log_grad(x_2d):
    x,y = x_2d
    const = (bivariate_normal_pdf(x_2d, 0, 0, 0)/3
        + bivariate_normal_pdf(x_2d, 0.9, -8, -6)/3 
        + bivariate_normal_pdf(x_2d, -0.9, 8, 6)/3)
    # const = 1
    ker1 = exp(-(x**2+y**2)/2) / (6*pi)
    ker2 = exp(-(x**2 + y**2 - 1.8*x*y)/(2*(1-0.9**2))) / (6*pi*(1-0.9**2)**0.5)
    ker3 = exp(-(x**2 + y**2 + 1.8*x*y)/(2*(1-0.9**2))) / (6*pi*(1-0.9**2)**0.5)
    diff_by_x = ker1*(-x) + ker2 * (-(x - 0.9*y)/(1 - 0.9**2)) + ker3 * (-(x + 0.9*y)/(1 - 0.9**2))
    diff_by_y = ker1*(-y) + ker2 * (-(y - 0.9*x)/(1 - 0.9**2)) + ker3 * (-(y + 0.9*x)/(1 - 0.9**2))
    diff_by_x /= const
    diff_by_y /= const
    return (diff_by_x, diff_by_y)

def setting_contourplot(start=-10, end=10):    
    grid = [x/10 + start for x in range(10*(end-start))]
    mixture_val = [[mixture_log_pdf([x,y]) for x in grid] for y in grid]
    plt.contour(grid, grid, mixture_val, levels=20)


#for multiprocessing
def multiproc_1unit_do(result_queue, leapfrog_step_num, step_size, initial, each_num_iter):
    func_pid = os.getpid()
    print("pid: ", func_pid, "start!")
    HMCchain = MC_Hamiltonian_withUtil_2dim(mixture_log_pdf, mixture_log_grad, leapfrog_step_num, step_size, initial)
    try:
        HMCchain.generate_samples(each_num_iter, pid=func_pid)
    except ValueError as e:
        print("pid: ", func_pid, ": early ended at "+ str(HMCchain.HMCiter) +"-th iteration.")
        print("exception catched!: ", e)


    result_queue.put(HMCchain)
    print("pid: ", func_pid, " end!")



if __name__ == "__main__":
    seed(2019-311-252)
    
    core_num = 8 #띄울 process 수
    testmode = 5
    #testmode setting
    #0. picture output debug mode
    #1. epsilon varying
    #2. leapfrog step L varying
    #3. set initial point randomly around 0 (high var)
    #4. set initial point randomly around -8,-6 
    #5. set initial point randomly around 8,6
    #6. set initial point randomly around -1,-1, keep epsilon/L ratio same
    if testmode==0:
        each_num_iter = 20000
        each_leapfrog_step_num = [4 for _ in range(8)]
        each_step_size = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
        each_initial = [(0,0) for _ in range(8)]

    if testmode==1:
        each_num_iter = 200000
        each_leapfrog_step_num = [4 for _ in range(8)]
        each_step_size = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
        each_initial = [(-5,-2) for _ in range(8)]

    if testmode==2:
        each_num_iter = 200000
        each_leapfrog_step_num = [1,2,3,5,8,10,15,20]
        each_step_size = [0.2 for _ in range(8)]
        each_initial = [(-5,-2) for _ in range(8)]

    if testmode==3:
        each_num_iter = 200000
        each_leapfrog_step_num = [4 for _ in range(8)]
        each_step_size = [0.2 for _ in range(8)]
        each_initial = [(normalvariate(0,2), normalvariate(0,2)) for _ in range(8)]

    if testmode==4:
        each_num_iter = 200000
        each_leapfrog_step_num = [4 for _ in range(8)]
        each_step_size = [0.2 for _ in range(8)]
        each_initial = [(normalvariate(-8,1), normalvariate(-6,1)) for _ in range(8)]
    
    if testmode==5:
        each_num_iter = 200000
        each_leapfrog_step_num = [4 for _ in range(8)]
        each_step_size = [0.2 for _ in range(8)]
        each_initial = [(normalvariate(8,1), normalvariate(6,1)) for _ in range(8)]

    if testmode==6:
        each_num_iter = 200000
        each_leapfrog_step_num = [1,2,3,5,7,10,13,16]
        each_step_size = [0.8/L for L in each_leapfrog_step_num]
        each_initial = [(normalvariate(-1,1), normalvariate(-1,1)) for _ in range(8)]

    
    #generate HMC chains using multiprocessing parallelly
    proc_vec = []
    proc_queue = mp.Queue()

    for i in range(core_num):
        unit_proc = mp.Process(target = multiproc_1unit_do, 
            args=(proc_queue, each_leapfrog_step_num[i], each_step_size[i], each_initial[i], each_num_iter))
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
    grid_column= 8
    grid_row = 4
    plt.figure(figsize=(5*grid_column, 3*grid_row))
    for i, chain in enumerate(mp_result_vec):
        plt.subplot(grid_row, grid_column, 4*i+1)
        plt.subplots_adjust(hspace=0.6)
        setting_contourplot(-12,12)
        chain.show_scatterplot(show=False)
        title_str = "step size:" + str(round(chain.step_size,4)) \
            + "\nleapfrog iter num:" + str(chain.leapfrog_step_num) \
            + "\ninitial: " + str(round(chain.MC_sample[0][0],4)) + ", " + str(round(chain.MC_sample[0][1],4))
        plt.title(title_str)
        
        plt.subplot(grid_row, grid_column, 4*i+2)
        title_str = "total iter num:" + str(chain.HMCiter) \
            + "\nacceptance rate:" + str(round(chain.get_accept_rate(),5))
        plt.title(title_str)
        chain.show_momentum(show=False)
        plt.subplot(grid_row, grid_column, 4*i+3)
        chain.show_acf(0,10,show=False)
        plt.subplot(grid_row, grid_column, 4*i+4)
        chain.show_acf(1,10,show=False)
    plt.show()
