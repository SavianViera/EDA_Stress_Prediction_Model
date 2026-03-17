df <- read.csv("Data/Processed/Window_10_No_Overlap.csv")
install.packages("moments")
library(moments)
df$Max <- apply(df[2:40], MARGIN = 1, FUN = max, na.rm=TRUE)
df$Min <- apply(df[2:40], MARGIN = 1, FUN = min, na.rm=TRUE)
df$std <- apply(df[2:40], MARGIN = 1, FUN = sd, na.rm=TRUE)
df$kurtosis <- apply(df[2:40], MARGIN = 1, FUN = kurtosis, na.rm=TRUE)
df$skewness <- apply(df[2:40], MARGIN = 1, FUN = skewness, na.rm=TRUE)
df$mean <- apply(df[2:40], MARGIN = 1, FUN = mean, na.rm=TRUE)

library(ggplot2)
ggplot(df, aes(x = condition, y = Max, fill = condition)) +
  geom_boxplot() +
  labs(Title = "Spread of Max values for Each Condition")

ggplot(df, aes(x = condition, y = Min, fill = condition)) +
  geom_boxplot() +
  labs(Title = "Spread of Min values for Each Condition")

ggplot(df, aes(x = condition, y = std, fill = condition)) +
  geom_boxplot() +
  labs(Title = "Spread of Standard Deviation values for Each Condition")

ggplot(df, aes(x = condition, y = kurtosis, fill = condition)) +
  geom_boxplot() +
  labs(Title = "Spread of Kurtosis values for Each Condition")

ggplot(df, aes(x = condition, y = skewness, fill = condition)) +
  geom_boxplot() +
  labs(Title = "Spread of Skewness values for Each Condition")

ggplot(df, aes(x = condition, y = mean, fill = condition)) +
  geom_boxplot() +
  labs(Title = "Spread of mean values for Each Condition")
library(nnet)
#Multinomial logistic regression
model <- multinom(condition ~ Max + Min + skewness + kurtosis + mean + std, data=df)
#log-odds
z <- summary(model)$coefficients/summary(model)$standard.errors
z
#2-tailed z-test
(1 - pnorm(abs(z), 0, 1)) * 2
#Relative risk
exp(coef(model))

