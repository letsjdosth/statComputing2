#python 3 file created by Choi, Seokjun

#using independent Markov chain monte carlo,
#draw posterior samples

#더구현할내용: 1.알아서 initial point 여러개 잡고 수렴점 비교하기
#<<음 만들었는데 좀 병맛인

# 2. acf plot (음 귀찮다)

from math import exp, log, factorial, pi, tan
from random import seed, uniform, normalvariate
from statistics import mean

import matplotlib.pyplot as plt

class IndepMcPost:
    def __init__(self, likelihood_pdf, proposal_pdf, proposal_sampler, data):
        self.likelihood_pdf = likelihood_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        self.data = data
        self.posterior_sample = []

        self.num_total_iters = 0
        self.num_accept = 0
    
    def log_likelihood(self, *params):
        log_likelihood_val = 0
        for elem in self.data:
            pdfval = self.likelihood_pdf(elem, *params)
            if pdfval==0: #엿같게 cauchy가 너무 꼬리가길어서 튄점에서 underflow나서 0나옴 log(0)=음 끔찍
                pdfval=0.00000000000000001 #python fixed-float minimum value
            log_likelihood_val += log(pdfval)
        return log_likelihood_val

    def log_r_calculator(self, candid, last):
        log_r = self.log_likelihood(candid) - self.log_likelihood(last)
        return log_r

    def sampler(self):
        last = self.posterior_sample[-1]
        candid = self.proposal_sampler()
        unif_sample = uniform(0, 1)
        log_r = self.log_r_calculator(candid, last)
        # print(log(unif_sample), log_r) #for debug
        if log(unif_sample) < log_r:
            self.posterior_sample.append(candid)
            self.num_total_iters += 1
            self.num_accept += 1
        else:
            self.posterior_sample.append(last)
            self.num_total_iters += 1

    def generate_samples(self, initial_val, num_samples, num_burn_in=0):
        n = num_samples + num_burn_in
        self.posterior_sample.append(initial_val)
        for _ in range(1, n):
            self.sampler()
        self.posterior_sample = \
            self.posterior_sample[(len(self.posterior_sample)-num_samples):len(self.posterior_sample)]

    def get_autocorr(self, maxLag):
        y = self.posterior_sample
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
        plt.plot(range(len(self.posterior_sample)),self.posterior_sample)
        plt.show()

    def show_hist(self):
        plt.hist(self.posterior_sample, 100)
        plt.show()

    def get_sample_mean(self, startidx=None):
        if startidx is not None:
            return mean(self.posterior_sample[startidx:])
        else:
            return mean(self.posterior_sample)

    def get_acceptance_rate(self):
        return self.num_accept/self.num_total_iters

#음 좀맘에안드는데 어떻게해야맘에들지모르겠음
def MCMC_conv_verifier(initial_val_list, likelihood_pdf, proposal_pdf, proposal_sampler, data):
    #for 'so called' standard setting
    moment1_list = []
    for initial in initial_val_list:
        testInst = IndepMcPost(likelihood_pdf, proposal_pdf, proposal_sampler, data)
        testInst.generate_samples(initial_val=initial, num_samples=30000)
        moment1_list.append(testInst.get_sample_mean(10000)) #앞에 10000개 버리고
        plt.plot(range(30000),testInst.posterior_sample, linewidth=0.1)
    print(moment1_list)
    print("maximum of diff : ", max(moment1_list)-min(moment1_list))
    plt.show()

if __name__ == "__main__":
    print("run as main")
    seed(2019-311-252)
    
    #ex1
    ex1_data = (8,3,4,3,1,7,2,6,2,7)
    def pois_pmf(x, param_lambda):
        if not isinstance(x, int):
            raise ValueError("x should be integer.")
        return ((param_lambda**x)*exp(-param_lambda)/factorial(x))
    def lognormal_pdf(x, param_mu=4, param_sigma=0.5):
        if not param_sigma>0:
            raise ValueError("sigma should be greater then 0")
        if not x>0:
            raise ValueError("x should be greater then 0")
        const = 1/(x*param_sigma*((2*pi)**0.5))
        ker = exp(-(log(x)-param_mu)**2/(2*(param_sigma**2)))
        return const*ker
    def lognormal_sampler(param_mu=4, param_sigma=0.5):
        return exp(normalvariate(log(param_mu), param_sigma))

    Pois_Lognorm_model = IndepMcPost(pois_pmf, lognormal_pdf, lognormal_sampler, ex1_data)
    Pois_Lognorm_model.generate_samples(initial_val=1, num_samples=10000, num_burn_in=1000)
    print(Pois_Lognorm_model.get_acceptance_rate())
    print(Pois_Lognorm_model.get_sample_mean())
    print(Pois_Lognorm_model.get_autocorr(10))
    Pois_Lognorm_model.show_traceplot()
    Pois_Lognorm_model.show_hist()

    #verify convergence with random initial points
    rand_initials = [uniform(0,10) for _ in range(10)]
    MCMC_conv_verifier(rand_initials, pois_pmf, lognormal_pdf, lognormal_sampler, ex1_data)


    #ex2.
    ex2_data = (2.983, 1.309, 0.957, 2.16, 0.801, 1.747, -0.274, 1.071, 2.094, 2.215,
        2.255, 3.366, 1.028, 3.572, 2.236, 4.009, 1.619, 1.354, 1.415, 1.937)
    print(mean(ex2_data)) #1.8927
    
    def normal_pdf(x, param_mu, param_sigma=1):
        if not param_sigma>0:
            raise ValueError("sigma should be greater then 0")
        const = 1/(param_sigma*((2*pi)**0.5))
        ker = exp(-(x-param_mu)**2/(2*(param_sigma**2)))
        return const*ker

    def cauchy_pdf(x,param_loc=0, param_scale=1):
        loc_part = ((x-param_loc)/param_scale)**2
        numer = pi * param_scale * (1 + loc_part)
        return 1/numer

    def onlyloc_cauchy_pdf(x, param_loc):
        return cauchy_pdf(x, param_loc, 1)

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

    cauchy_sampler = CauchySampler(0,1)
    Norm_Cauchy_model = IndepMcPost(normal_pdf, onlyloc_cauchy_pdf, cauchy_sampler, ex2_data)
    Norm_Cauchy_model.generate_samples(initial_val=1, num_samples=30000, num_burn_in=10000)
    print(Norm_Cauchy_model.get_acceptance_rate())
    print(Norm_Cauchy_model.get_sample_mean())
    print(Norm_Cauchy_model.get_autocorr(10))
    Norm_Cauchy_model.show_traceplot()
    Norm_Cauchy_model.show_hist()

    #verify convergence with random initial points
    rand_initials = [uniform(0,10) for _ in range(10)]
    MCMC_conv_verifier(rand_initials, normal_pdf, onlyloc_cauchy_pdf, cauchy_sampler, ex2_data)
