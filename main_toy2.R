#generate a linear model with slowly changing slope
set.seed(1) 
N = 1000
a0 = 2
b0 = -1
eps = rnorm(N,mean=0,sd=0.001)
x = rnorm(N, mean = 0.75, sd = 0.05)
y=b0+(a0+0.00005*seq(1,N))*x + eps
df = data.frame(cbind(x, y))
names(df) = c("x", "y")
train = df[1:floor(0.5*N), ]
test = df[(floor(0.5*N) + 1):N, ]

#find R^2 on the test set
lmfit = lm(y~x, train)
gamma_lm = predict(lmfit, test)
r2_test = 1 - sum((test[, "y"]  - gamma_lm)^2) / sum((test[, "y"] - mean(test[, "y"]))^2)

#prediction interval for outcomes
A = min(train$y)
B = max(train$y)

#step of piecewise-constant function
step = 10000
int = seq(A, B, (B-A)/step)

#apply our algorithm on the test data set
outcomes = test[, "y"]
gamma = AA_CRPS(outcomes, as.matrix(test[, "x"]), A, B, M = 1500, M0 = 300, a = 0.5, sigma = 0.1, discount = 0.999, step = 10000)$gamma

T = nrow(test)
#represent as piecewise-constant functions
outcomes_vec = matrix(0, T, length(int))
for (i in 1:T) {
  outcomes_vec[i, ] = as.numeric(rep(outcomes[i], length(int)) < int)
}

#linear regression
lm_vec = matrix(0, T, length(int))
for (i in 1:T) {
  lm_vec[i, ] = as.numeric(rep(gamma_lm[i], length(int)) < int)
}

#quantile regression
library(quantreg)
tau = seq(0, 1, 0.01)
qr1 <- rq(y~x, data=train, tau = seq(0, 1, 0.01))
gamma_qr = predict(qr1, test)
qr_vec = matrix(0, nrow = dim(gamma_qr)[1], ncol = length(int))
for (i in 1:length(int))  {
  qr_vec[, i] = tau[apply(abs(gamma_qr - int[i]), 1, which.min)]
}

#quantile regression online
gamma_qr2 = matrix(0, nrow = T, ncol = 101)
for (k in 1:T)  {
  temp = rbind(train, test[1:k,])
  qr2 <- rq("y~x", data=temp, tau = seq(0, 1, 0.01))
  gamma_qr2[k, ] = predict(qr2, test[k,])
}
qr_vec2 = matrix(0, nrow = dim(gamma_qr2)[1], ncol = length(int))
for (i in 1:length(int))  {
  qr_vec2[, i] = tau[apply(abs(gamma_qr2 - int[i]), 1, which.min)]
}

#Calculate losses
loss_AA = apply((gamma - outcomes_vec)^2, 1, sum) * (B-A) / step
Loss_AA = cumsum(loss_AA)
loss_lm = apply((lm_vec - outcomes_vec)^2, 1, sum) * (B-A) / step
Loss_lm = cumsum(loss_lm) 
loss_qr = apply((qr_vec - outcomes_vec)^2, 1, sum) * (B-A) / step
Loss_qr = cumsum(loss_qr) 
loss_qr2 = apply((qr_vec2 - outcomes_vec)^2, 1, sum) * (B-A) / step
Loss_qr2 = cumsum(loss_qr2) 

pdf("Loss_diff_toy2.pdf", height = 8.5, width = 8.5, paper = "special")
plot(Loss_lm - Loss_AA, type = "l", lty = 1, lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, xlab = "Time", ylab = "", main= "Loss difference")
lines(Loss_qr - Loss_AA, lty = 2, lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
lines(Loss_qr2 - Loss_AA, lty = 3, lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
legend("topleft", legend = c("LM", "QR", "QR online"), lty = 1:3, lwd=3,cex=2)
lines(rep(0, length(Loss_AA)),  lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
dev.off()
