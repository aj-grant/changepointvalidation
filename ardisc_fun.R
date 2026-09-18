#Compute the autocovariance function of time series x up to lag p
#If mean_correct = FALSE, the time series will not be mean corrected, and the time series x should have zero mean
acf <- function(x, p, mean_correct = TRUE){
  Tx = length(x)
  if (mean_correct == TRUE){
    x1 = x - mean(x)
  } else{
    x1 = x
  }
  ac = numeric(p+1)
  for (j in 0:p){
    ac[j+1] = crossprod(x1[(j+1):Tx], x1[1:(Tx-j)]) / Tx
  }
  return(ac)
}

#Implement the Levinson-Durbin algorithm to compute estimates of autoregressive parameters b and innovation variance s
#given an autocovariance function ac.
#The autoregressive order is length(ac)-1
ldar <- function(ac){
  K <- length(ac) - 1
  s <- ac[1]
  if (K == 0){
    b <- 0
  } else {
    b <- numeric(K)
    b[1] <- -ac[2] / ac[1]
    s <- s*(1 - b[1]^2)
    if (K > 1){
      for (k in 1:(K-1)){
        d <- ac[k+2] + crossprod(ac[(k+1):2], b[1:k])
        v <- -d / s
        b[1:k] <- b[1:k] + b[k:1] * c(v) 
        b[k+1] <- v
        s <- s*(1 - v^2)
      }
    }
  }
  return(list("b" = b, "s" = s))
}

#Estimates common autoregressive parameters under the null hypothesis that x and y are realisations from autoregressive
#process with the same autoregressive parameters.
#lam is the ratio of the innovation variances of the two time series.
#tol is the tolerance for convergence.
bivar <- function(x, y, p, lam, tol = 1e-8){
  Tx = length(x)
  Ty = length(y)
  cx <- acf(x, p)
  cy <- acf(y, p)
  toe <- toeplitz(cy[1:p])
  lamdif <- tol + 1
  while (abs(lamdif) > tol) {
    c0 <- (Tx * cx + c(lam) * Ty * cy) / (Tx + Ty)
    parest <- ldar(c0)
    lamdif <- lam - parest$s / (cy[1] + 2*crossprod(parest$b, cy[2:(p+1)]) + crossprod(parest$b, toe %*% parest$b))
    lam <- lam - lamdif
  }
  return(list("b" = parest$b, "sx" = parest$s, "sy" = parest$s/lam))
}

#Implements the likelihood ratio test for comparing two univariate time series.
#If mu_cp = sigma_cp = FALSE, the null hypothesis tested is that x and y are from autoregressions with the same
#autoregressive parameters.
#If mu_cp = FALSE, sigma_cp = TRUE, the null hypothesis tested is that x and y are from autoregressions with the same
#autoregressive parameters and the same innovation variances.
#If mu_cp = TRUE, sigma_cp = TRUE, the null hypothesis tested is that x and y are from autoregressions with the same
#autoregressive parameters, the same innovation variances, and the same means.
ardisc <- function(x, y, mu_cp = FALSE, sigma_cp = FALSE, tol = 1e-8){
  Tx <- length(x)
  Ty <- length(y)
  p <- floor(log(min(Tx, Ty))^1.01)
  df <- p
  mx <- mean(x)
  my <- mean(y)
  if (mu_cp == TRUE){
    mx0 <- my0 <- (Tx*mx + Ty*my) / (Tx + Ty)
    df <- df + 1
  } else{
    mx0 <- mx
    my0 <- my
  }
  cx <- acf(x, p)
  cy <- acf(y, p)
  c <- (Tx*acf(x - mx0, p, mean_correct = FALSE) + Ty*acf(y - my0, p, mean_correct = FALSE))/(Tx + Ty)
  
  parestx <- ldar(cx)
  paresty <- ldar(cy)
  
  if (sigma_cp == TRUE){
    parest0 <- ldar(c)
    df <- df + 1
    sx0 <- parest0$s
    sy0 <- parest0$s
  } else{
    lam_init <- parestx$s / paresty$s
    parest0 <- bivar(x, y, p, lam_init, tol = tol)
    sx0 <- parest0$sx
    sy0 <- parest0$sy
  }
  l <- Tx*log(sx0 / parestx$s) + Ty*log(sy0 / paresty$s)
  
  pvalue <- 1 - pchisq(l, df)
  h <- pvalue > 0.05
  
  return(list("pvalue" = pvalue, "l" = l, "df" = df, "cx" = cx, "cy" = cy, "c" = c,
              b0 = parest0$b, sx0 = sx0, sy0 = sy0))
}