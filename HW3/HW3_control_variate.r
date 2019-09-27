#example 2 (HW3)
m = 10000
# MC
u = runif(m)
mc.sample = (exp(-u)/(1+u^2))
mean(mc.sample)
var(mc.sample)

#control variate MC
con.var.sample = (exp(-0.5)/(1+u^2)) #같은 u 써야함
lambda = -cov(mc.sample, con.var.sample)/var(con.var.sample)
print(lambda) # -2.45가량
con.sample= mc.sample + lambda*(con.var.sample - exp(-0.5)*pi/4)
mean(con.sample)
var(con.sample)

#개선?
c(mean(mc.sample), mean(con.sample))
c(var(mc.sample), var(con.sample))
(var(mc.sample)-var(con.sample))/var(mc.sample)