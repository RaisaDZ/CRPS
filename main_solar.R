#reproducing experiments for solar power forecasting
df = read.csv("data.csv")
features = c("radiation_direct_horizontal", "radiation_diffuse_horizontal")
target_name = c("solar_generation_actual")

df[[target_name]] = df[[target_name]] / 1000

train = subset(df, df$year == 2015)
test = subset(df, df$year == 2016)

features_min = apply(train[, features], 2, min)
features_max = apply(train[, features], 2, max)
features_min_train = matrix(rep(features_min, dim(train)[1]), ncol = length(features), byrow = T)
features_max_train = matrix(rep(features_max, dim(train)[1]), ncol = length(features), byrow = T)
train_scaled = (train[, features] - features_min_train) / (features_max_train - features_min_train)
features_min_test = matrix(rep(features_min, dim(test)[1]), ncol = length(features), byrow = T)
features_max_test = matrix(rep(features_max, dim(test)[1]), ncol = length(features), byrow = T)
test_scaled = (test[, features] - features_min_test) / (features_max_test - features_min_test)

train_target_scaled = cbind(train[, target_name], train_scaled)
names(train_target_scaled)[1] = target_name

#linear model for solar power
formula = as.formula(paste(target_name, paste(features, collapse=" + "), sep=" ~ "))
lmfit = lm(formula, train_target_scaled)
summary(lmfit)
gamma_lm = predict(lmfit, test_scaled)
r2_test = 1 - sum((test[, target_name]  - gamma_lm)^2) / sum((test[, target_name] - mean(test[, target_name]))^2)

#prediction interval for outcomes
A = min(train_target_scaled[, target_name])
B = max(train_target_scaled[, target_name])

#step of piecewise-constant function
step = 10000
int = seq(A, B, (B-A)/step)

#apply our algorithm on the test data set
outcomes = test[, target_name]
gamma = AA_CRPS(outcomes, as.matrix(test_scaled[, features]), A, B, M = 1500, M0 = 300, a = 0.1, sigma = 0.03, discount = 1, step = 10000)$gamma

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
qr1 <- rq(formula, data=train_target_scaled, tau = seq(0, 1, 0.01))
gamma_qr = predict(qr1, test_scaled)
qr_vec = matrix(0, nrow = dim(gamma_qr)[1], ncol = length(int))
for (i in 1:length(int))  {
  qr_vec[, i] = tau[apply(abs(gamma_qr - int[i]), 1, which.min)]
}

#Calculate losses
loss_AA = apply((gamma - outcomes_vec)^2, 1, sum) * (B-A) / step
Loss_AA = cumsum(loss_AA)
loss_lm = apply((lm_vec - outcomes_vec)^2, 1, sum) * (B-A) / step
Loss_lm = cumsum(loss_lm) 
loss_qr = apply((qr_vec - outcomes_vec)^2, 1, sum) * (B-A) / step
Loss_qr = cumsum(loss_qr) 
max(Loss_AA)
max(Loss_lm)
max(Loss_qr)
