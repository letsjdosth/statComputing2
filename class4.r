#Sequential monte carlo
#ex1. AR1
n = 1000
sigma = 0.16
alpha = 0.975
x = rep(NA, n+1)
x[1] = 0

for(i in 1:n) {
    x[i+1] = alpha * x[i] + sigma * rnorm(1)
}
plot(1:(n+1), x, type='l')

#ex2. Stochastic Vol model
n = 1000
alpha = 0.75
beta = 0.63
sigma = 0.16
x = rep(NA, n+1) #vol sequence
y = rep(NA, n) #observed log-gain sequence
x[1] = 0

for(i in 1:n){
    x[i+1] = alpha * x[i] + sigma * rnorm(1)
    y[i] = beta * exp(x[i+1]/2) * rnorm(1)
}
plot(1:(n+1), x, type='l')
plot(1:n, y, type='l')


#ex3. Kalman Filter
library(dlm)
data(Nile)

#Kalman filter: exact part
nileBuild = function(par) {
    dlmModPoly(1, dV=exp(par[1]), dW=exp(par[2]), m0=rep(1180,1), C0=0.001*diag(nrow=1))
}
nileMLE = dlmMLE(Nile, rep(0,2), nileBuild)
#the function returns the MLE of unknown parameters in the specification of a statce space model

nileMLE$conv
nileMod = nileBuild(nileMLE$par)
nileFilt = dlmFilter(Nile, nileMod)

#SIS part
V = as.numeric(nileMod$V) #sigma^2_y. variance of observation noise
W = as.numeric(nileMod$W) #sigma^2_x + sigma^2_n. diagonal element of the system noise (assume cov=0)
n_samples = 2000
n_time = length(Nile)
p = function(x, y){ #y_k = x_k + sigma_y * epsilon_k (epsilon_k~N(0,1))
    dnorm(x = y, mean=x, sd=sqrt(V))
}
q = function(x, x_p){ #x_k+1 = x_k + sigma_x * e_(k+1) (e_(k+1)~N(0,1))
    dnorm(x=x, mean=x_p, sd=sqrt(W))
}
sigma2 = 1/(1/V + 1/W) #sigma_(n+1)
g = function(x, x_p, y){
    dnorm(x, mean=sigma2 * (x_p/W + y/V), sd=sqrt(sigma2))
}
rg = function(x_p, y){ #g에서 sample generate
    rnorm(n=n_samples, mean=sigma2*(x_p/W + y/V), sd=sqrt(sigma2))
}

X_ = rnorm(n_samples, nileFilt$m[1], sd=0.0001)
w = rep(1, n_samples)
X_vec = matrix(nrow=n_time, ncol=n_samples)
w_vec = X_vec
X_mean = vector(mode='numeric', length=n_time)
X_mean[1] = sum((w/sum(w))*X_)
for(i in 1:n_time){
    X_p = X_
    X_ = rg(X_p, Nile[i])
    w = w*p(X_,Nile[i]) * q(X_, X_p) / g(X_, X_p, Nile[i]) #unstandardized weight
    X_mean[i] = sum((w/sum(w))*X_)
    w_vec[i,] = w
    X_vec[i,] = X_
}


#중간에 코드 좀 빠졌을것임

X_mean = ts(X_mean, frequency=1, start=c(1871,1))
plot(cbind(Nile, nileFilt$m[-1]), plot.type='s', col=c("black","red"), ylab="Level", main="Nile river", lwd=c(1,2))
lines(X_mean, col='blue', lwd=2)
#red: exact
#blue: Sample Importance Resampling