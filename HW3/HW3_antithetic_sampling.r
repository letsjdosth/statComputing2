#HW3
#(직접할것) x=1.96에서, 그냥 montecarlo integral하여 분산 뽑고,
# antithetic sampling으로 마찬가지로 반복해 분산 뽑은 후
#분산 비교할 것

pnorm(1.96) #참고용

#MC
x = 1.96
iter = 1000
n = 10000
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
c(mean(mc.cdf), mean(as.cdf))
c(var(mc.cdf), var(as.cdf))
(var(mc.cdf)-var(as.cdf))/var(mc.cdf)


