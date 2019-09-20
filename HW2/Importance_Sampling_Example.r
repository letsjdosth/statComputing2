# R file

# example 1 : want E(h(x)) where f~unif(0,10)
x = rnorm(10000,5,1) #draw from g: normal(5,1)
weight.unstandardized = sqrt(2*pi) * exp(0.5*(x-5)^2) / 10 #f/g
weight.standardized = weight.unstandardized / sum(weight.unstandardized)
h <- function(x){
    return (exp(-2*abs(x-5)) * 10)
}
mean(h(x)*weight.unstandardized)
sum(h(x)*weight.standardized)