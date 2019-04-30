AA_CRPS = function(outcomes, X, A, B, M, M0, a, sigma, discount = 1, step = 10000)  {
  #Inputs: outcomes - target vector,
  #        X - matrix of features where rows are observations and columns are features
  #        A - minimum of outcomes
  #        B - maximum of outcomes
  #        M - maximum number of iteration
  #        M0 - the length of "burn-in" period
  #        a - regualrization parameter
  #        sigma - standard deviation
  #        discount - discount factor
  #        step - step of piecewise-constant function
  
  #Outputs: gamma - calculated predictions for WAAQR,
  #         theta - sampling parameters at each iteration,
  #         lik - log-likelihood of parameters at each iteration
  
  X <- cbind(rep(1, dim(X)[1]), X)  #add bias
  T <- dim(X)[1]   #time
  n <- dim(X)[2]   #dimension
  eta = 2/(B-A)  #learning rate
  
  omega <- function(theta, ksi, outcomes, discount) {
    #likelihood of parameters theta
    discount_vector = discount^seq(1, length(ksi))
    w <- - a * eta * sum(abs(theta)) - eta * sum(abs(ksi - outcomes) * discount_vector)
    return(w)
  }
  
  omega0 <- function(theta) {
    #likelihood of parameters theta at time t = 0
    w <- - a * eta * sum(abs(theta))
    return(w)
  }
  
  int = seq(A, B, (B-A)/step)
  
  gamma <- matrix(0, nrow = T, ncol = step+1)
  theta <- array(0, dim = c(T, M, n))
  lik = matrix(0, nrow = T, ncol = M)
  accept_mat =  matrix(0, T, 1)
  for (t in 1:T) {
    #initial estimates of theta
    if (t > 1) {
      theta[t, 1, ] <- theta[t-1, M, ] 
    }
    ksi1 = 0
    ksi2 = 0
    accept = 0
    for (m in 2:M)  {
      theta_old <- as.matrix(theta[t, m-1, ])  #theta from previous step m-1
      theta_new <- theta_old + matrix(rnorm(n, 0, sigma^2), nrow = n, ncol = 1) #sample new params
      if (t > 1) {
        ksi_old <- X[1:(t-1), ] %*% theta_old  #old probs
        ksi_new <- X[1:(t-1), ] %*% theta_new  #new probs
        omega_old = omega(theta_old, ksi_old, outcomes[1:(t-1)], discount)
        omega_new = omega(theta_new, ksi_new, outcomes[1:(t-1)], discount)
      } else {
        omega_old = omega0(theta_old)
        omega_new = omega0(theta_new)
      }
      alpha0 <- exp(omega_new - omega_old)
      alpha <- min(1, alpha0)
      rand <- runif(1, 0, 1)  #flip a coin
      if (alpha >= rand)  {  #accept new params
        theta[t, m, ] <- theta_new
        lik[t, m] = omega_new 
        accept = accept + 1
      } else {              #keep old params
        theta[t, m, ] <- theta[t, m-1, ]
        lik[t, m] = omega_old
      }
      ksi <- X[t, ] %*% theta[t, m, ]
      ksi_distr = as.numeric(rep(ksi, length(int)) < int)
      
      #burn-in
      if (m > M0) {
        ksi1 <- ksi1 + exp(-2*ksi_distr^2)
        ksi2 <- ksi2 + exp(-2*(1 - ksi_distr)^2)
      }
    }
    gamma[t, ] <- 1/2 - 1/4*log(ksi1 / ksi2)  #predictions
    accept_mat[t, ] = accept / (M-1)          #accept rate
  }
  return(list(gamma = gamma, theta = theta, lik = lik[, 2:M]))
}

