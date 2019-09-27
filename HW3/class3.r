#importance sampling : small tail probabilities
#그냥 할 때 문제점
sample.num = 1e+4
x= rnorm(sample.num)
length(x[x>4.5])/sample.num #x>4.5가 너무 rare해서 왠만한 sample수로는 case가 뽑히질 않음
pnorm(-4.5) #참고
#importance sampling
y = rexp(sample.num, 1) + 4.5
weight = exp(dnorm(y, log=TRUE) - dexp(y-4.5, log=TRUE)) #over/underflow protect
mean(weight) #y가 다 4.5보다 큰 샘플이므로 안 버려도 됨


#HW3: prob 1 : f(x)~normal(0,1), g(x)~normal(20,1), Ef(I(X>20))?
y=rnorm(sample.num, 20, 1)
weight = exp(dnorm(y, log=TRUE) - dnorm(y, 20, 1, log=TRUE)) * (y>20)
mean(weight)



#antithetic sampling example1
#example ; \int_{0}^{1}\frac{1}{1+x}dx ???
iter.num=1000
sample.num=10000
#그냥
est.mc = rep(NA, iter.num)
for(i in 1:iter.num){
    u=runif(sample.num)
    y=1/(1+u)
    est.mc[i] = mean(y)
}
#antithetic sampling
est.as = rep(NA, iter.num)
for(i in 1:iter.num){
    u=runif(sample.num/2)
    u=c(u, 1-u)
    y=1/(1+u)
    est.as[i] = mean(y)
}
#comparing 2 results
c(mean(est.mc), mean(est.as))
c(sd(est.mc), sd(est.as)) #<-매우 차이남
(var(est.mc)-var(est.as))/var(est.mc) #약 94% 줄었음

#HW3
#antithetic sampling example 2
#cdf value of normal distribution(0,1), when x>0
#그냥 해보기
x = seq(0.1, 2.5, by=0.1)
n = 100000
u = runif(n)
cdf = rep(NA, length(x))
for(i in 1:length(x)){
    g = x[i] * exp(-0.5*(u*x[i])^2)
    cdf[i] = 0.5 + mean(g)/sqrt(2*pi)
}
#verify
phi = pnorm(x)
rbind(x, cdf, phi)


#HW3
#(직접할것) x=1.96에서, 그냥 montecarlo integral하여 분산 뽑고,
# antithetic sampling으로 마찬가지로 반복해 분산 뽑은 후
pnorm(1.96)
#분산 비교할 것
#그냥
x = 1.96
iter = 100
n = 100
mc.cdf = rep(NA, iter)
antithetic.cdf = rep(NA, iter)
for(i in 1:iter){
    u = runif(n)
    g = x * exp(-0.5*(u*x)^2)
    mc.cdf[i] = 0.5 + mean(g)/sqrt(2*pi)
}
mean(mc.cdf)
var(mc.cdf)

#antithetic
as.cdf = rep(NA, iter)
antithetic.cdf = rep(NA, iter)
for(i in 1:iter){
    u = runif(n/2)
    u = c(u, 1-u)
    g = x * exp(-0.5*(u*x)^2)
    as.cdf[i] = 0.5 + mean(g)/sqrt(2*pi)
}
mean(as.cdf)
var(as.cdf)

#비교
(var(mc.cdf)-var(as.cdf))/var(mc.cdf)



#control variate
#example 1
m= 10000
lambda = -12 + 6*(exp(1)-1)
u = runif(m)
T1 = exp(u) #general MC estimator 
T2 = exp(u) + lambda*(u - 0.5) #use control variable 
c(var(T1), var(T2))
c(mean(T1), mean(T2))
(var(T1)-var(T2))/var(T1)


#example 2 (HW3)
m = 10000
#그냥 mc
u = runif(m)
mc.sample = (exp(-u)/(1+u^2))
mean(mc.sample)
var(mc.sample)

#control variate mc
con.var.sample = (exp(-0.5)/(1+u^2)) #같은 u 써야함
lambda = -cov(mc.sample, con.var.sample)/var(con.var.sample)
print(lambda) # -2.45가량
con.sample= mc.sample + lambda*(con.var.sample - exp(-0.5)*pi/4)
mean(con.sample)
var(con.sample)

#개선?
(var(mc.sample)-var(con.sample))/var(mc.sample)