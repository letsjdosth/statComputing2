#R files
#example of ARS

library(armspp)

#ARS(그냥 adaptive rejection sampling) example
?arms
output = arms(5000, function(x)-x^2/2, -1000, 1000, metropolis=FALSE, include_n_evaluation=TRUE)
hist(output$samples,nclass=100)
shapiro.test(output$samples)


#(~HW2~)
#ARMS(adaptive rejection metropolis sampling) example
# Mixture of normals: 0.4 N(-1, 1) + 0.6 N(4, 1). Not log concave.
#(그려보면 bimodal임.)
dnormmixture <- function(x) {
    parts <- log(c(0.4, 0.6)) + dnorm(x, mean = c(-1, 4), log = TRUE)
    log(sum(exp(parts - max(parts)))) + max(parts) 
    # 이부분도 overflow 방지용임. 즉, 그냥 exp(x)를 왕창 더하면 너무 커지므로
    # max(parts)를 나눠줬다가 다시 곱해주는 것임 
    # (그럼 더하는 파트는 음수가 되어서 ,exp(parts-max(parts)가 작아짐))
    
    # 이 예제에서는 log pdf를 만들어야하기 떄문에
    # 어차피 앞부분에 log를 씌우게되므로
    # 간단히 뒤에 더해주면 됨 (+max(parts))
}
#pdf 그려보자 (진짜 bimodal인지)
curve(exp(Vectorize(dnormmixture)(x)), xlim=c(-4,7))

#sample 뽑자
#이건 log-concave case가 아니므로, metropolis option을 true로 놓고 ARMS를 돌려야함
samples <- arms(5000, dnormmixture, -1000, 1000) 
hist(samples, freq = FALSE, nclass=100)
curve(exp(Vectorize(dnormmixture)(x)), add=TRUE)


#뒤쪽 공식 arms함수 예제
# List of log pdfs, demonstrating recycling of log_pdf argument
samples <- arms(
    10,
    list(
        function(x) -x ^ 2 / 2,
        function(x) -(x - 10) ^ 2 / 2
    ),
    -1000,
    1000
)
print(samples)

# Another way to achieve the above, this time with recycling in arguments
samples <- arms(
10, dnorm, -1000, 1000,
arguments = list(
    mean = c(0, 10), sd = 1, log = TRUE
)
)
print(samples)