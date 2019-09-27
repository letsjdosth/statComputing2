sample.num = 1e+4
y=rnorm(sample.num, 20, 1)
weight = exp(dnorm(y, log=TRUE) - dnorm(y, 20, 1, log=TRUE)) * (y>20)
mean(weight)