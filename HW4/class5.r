
#for data loading. of mixture distritbuion: find delta
mixture.dat = read.table("mixture.dat", header=T)
y = mixture.dat$y
hist(y, freq=FALSE, nclass=100)
?hist
x = seq(5,14,by=0.01)
d = 0.7 * dnorm(x,7,0.5) + 0.3*dnorm(x,10,0.5)
points(x,d,type="l")

#example 1. independent MCMC
n = 10500
delta = rep(NA, n+1) #delta[1]은 initial value
delta[1] = 0.5
delta[1]= rbeta(1,1,1) #case 1 : unif prior(or proposal)
# delta[1]= rbeta(1,2,10) #case 2: skewed
f= function(delta, y){
    return (prod(delta*dnorm(y,7,0.5)+(1-delta)*dnorm(y,10,0.5)))
}

accept.ratio = 0
for(i in 1:n){
    delta.new = rbeta(1,1,1) #case 1
    # delta.new = rbeta(1,2,10) #case 2
    r = (f(delta.new,y) / f(delta[i],y)) * (dbeta(delta[i],1,1) / dbeta(delta.new,1,1))
    u = runif(1)
    if(u < min(1,r)){
        delta[i+1] = delta.new
        accept.ratio = accept.ratio + (1/n)
    }
    else delta[i+1] = delta[i]
}
print(accept.ratio)
ts.plot(delta) #trace plot
acf(delta) #correlation plot
hist(delta, nclass=100)
#case 1은 괜찮은 것을, case 2에서는 망한걸... 확인할 수 있음



#example2. normal proposal to add previous. with logit transformation
niter = 10500
u = rep(NA, niter+1)
p = rep(NA, niter+1)
u[1] = runif(1, -1, 1)
p[1] = exp(u[1])/(1+exp(u[1]))

log.f= function(delta, y){
    return (sum(log(delta*dnorm(y,7,0.5)+(1-delta)*dnorm(y,10,0.5)))) #overflow/underflow protect
}
accept.ratio = 0
for(i in 1:niter){
    u.new = u[i] + runif(1,-1,1) #case 1
    # u.new = u[i] + runif(1,-0.01,0.01) #case 2 : too narrow jumping rule (조지는 케이스)
    p.new = exp(u.new) / (1+exp(u.new))
    #log density 쓰는김에 다 log버전으로 진행
    log.r = log.f(p.new, y) - log.f(p[i], y) + u[i] - u.new #마지막 2 term은 jacobian에서
    u2 = runif(1, 0, 1)
    # print(log(u2));print(log.r)
    if(log(u2) < log.r){
        u[i+1] = u.new
        p[i+1] = p.new
        accept.ratio = accept.ratio + (1/niter)
    } else {
        u[i+1] = u[i]
        p[i+1] = p[i]
    }
}
print(accept.ratio)
u = u[502:(niter+1)]
p = p[502:(niter+1)]
hist(p, nclass=100, xlim=c(0.3,1))
ts.plot(p)
acf(p)