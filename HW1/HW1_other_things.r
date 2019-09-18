#implement
x = rnorm(10000)
mean(x)
mean(x**2)
mean(x**3)
mean(x**4)

#######################################
#implement
#exponential dist of sampling

u = runif(1000)
beta = 3
sample = -beta * log(u)
hist(sample,nclass=100)
# using nclass with more than # 100

##################################


####################################
#implement
#rejection sampling with f = normal and g = exponential
# generating normal dist

n = 10000
x = rep(NA, n)
index = 1

while(index <=n){
    y = rexp(1)
    r = exp(-(y-1)**2 / 2)
    u = runif(1)
    
    if(u < r ){
      u2 = runif(1)
      if(u2 < 0.5)
        x[index] = -y
      else x[index] = y
      
      index = index + 1
    }
}

hist(x , nclass = 100)