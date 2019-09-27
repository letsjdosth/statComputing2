#SIR

##
#example 1
#과거 bayes model posterior 예제
x = c(8,3,4,3,1,7,2,6,2,7)
m = 100000
lambda.sample = exp(rnorm(m, log(4), 0.5))
w = rep(NA, m)

#calculate weight
for(i in 1:m){
    w[i] = sum(dpois(x, lambda.sample[i], log=TRUE))
}
w = w - max(w) #protect overflow
w.st = exp(w) / sum(exp(w))

#resampling
lambda = sample(lambda.sample, 10000, replace=TRUE, prob=w.st)
hist(lambda, nclass=100)


#HW3
#example 2
#hw4.1
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
result.sample = sample(theta.sample, 10000, replace=TRUE, prob=weight.st)
hist(result.sample, nclass=100)


#hw4.2
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
hist(rj.result.sample, nclass=100)


#compare
par(mfrow=c(1,2))
hist(result.sample, nclass=100)
hist(rj.result.sample, nclass=100)

