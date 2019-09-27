#HW3

set.seed(2019-311-252)

#hw4.1 : SIR
norm.txt.sample=c(2.983, 1.309, 0.957, 2.16, 0.801, 1.747, -0.274, 1.071, 2.094, 2.215, 2.255, 3.366, 1.028, 3.572, 2.236, 4.009, 1.619, 1.354, 1.415, 1.937)
m=100000
theta.sample = rcauchy(m,0,1)
weight = rep(NA, m)
#weight
for(i in 1:m){
    weight[i] = sum(dnorm(norm.txt.sample, theta.sample[i], 1, log=TRUE))
}
weight = weight - max(weight)
weight.st = exp(weight) / sum(exp(weight))

#resampling
sir.result.sample = sample(theta.sample, 5000, replace=TRUE, prob=weight.st)
# hist(sir.result.sample, nclass=100)


#hw4.2 : Rejection Sampling
norm.txt.sample=c(2.983, 1.309, 0.957, 2.16, 0.801, 1.747, -0.274, 1.071, 2.094, 2.215, 2.255, 3.366, 1.028, 3.572, 2.236, 4.009, 1.619, 1.354, 1.415, 1.937)
theta.mle= mean(norm.txt.sample)
log_p_cal = function(candid){
        p=0
        for(i in 1:length(norm.txt.sample)){
            p = p + dnorm(norm.txt.sample[i], candid, 1, log=TRUE)-dnorm(norm.txt.sample[i], theta.mle, 1, log=TRUE)
        }
        return(p)
}
rj.result.sample = rep(NA, 5000)
gen.sample.num=0
while(gen.sample.num<5000){
    cauchy.sample = rcauchy(1,0,1)
    unif.sample = runif(1,0,1) 
    log_p = log_p_cal(cauchy.sample)
    if(log(unif.sample)<log_p){
        # print('accept')
        gen.sample.num = gen.sample.num+1
        rj.result.sample[gen.sample.num] = cauchy.sample
    } else {
        # print('reject')
    }
}
# hist(rj.result.sample, nclass=100)


#4.3. compare
par(mfrow=c(1,2))
hist(sir.result.sample, nclass=100)
hist(rj.result.sample, nclass=100)
c(mean(sir.result.sample), mean(rj.result.sample))
c(var(sir.result.sample), var(rj.result.sample))