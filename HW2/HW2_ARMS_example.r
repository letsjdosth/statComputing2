# R file

#(~HW2~)
#ARMS(adaptive rejection metropolis sampling) example
# Mixture of normals: 0.4 N(-1, 1) + 0.6 N(4, 1). Not log concave.

library(armspp)


dnormmixture <- function(x) {
    parts <- log(c(0.4, 0.6)) + dnorm(x, mean = c(-1, 4), log = TRUE)
    log(sum(exp(parts - max(parts)))) + max(parts) 
    #overflow protect
}

#pdf 그려보자 (진짜 bimodal인지)
curve(exp(Vectorize(dnormmixture)(x)), xlim=c(-4,7))

#sample 뽑자
#이건 log-concave case가 아니므로, metropolis option을 true로 놓고 ARMS를 돌려야함
samples <- arms(5000, dnormmixture, -1000, 1000) 
hist(samples, freq = FALSE, nclass=100)
curve(exp(Vectorize(dnormmixture)(x)), add=TRUE)
