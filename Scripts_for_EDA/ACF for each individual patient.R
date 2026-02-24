library(tidyverse)
library(dplyr)
Pos <- filter(data2, data2$condition=="positive")
acf_pos <- acf(Pos, na.action = na.pass, plot = TRUE, 
               main = "ACF for Participant 01 (Positive Condition)")

  #data_long[data_long$condition == "positive", ]
neg <- data_long[data_long$condition == "negative", ]
black <- data_long[data_long$condition == "black", ]
neut <- data_long[data_long$condition == "neutral", ]

ggplot(Pos,
       aes(x = Time, y = EDA_Value)) +
  geom_line(color = "steelblue") +
  facet_wrap(~ PatientID, scales = "free_y") +
  labs(title = "EDA Trajectories per Patient",
       x = "Time",
       y = "EDA") +
  theme_minimal()


# Option A: Remove rows with any NA values
data_positive_clean <- na.omit(Pos)
data_negative_clean <- na.omit(neg)
data_black_clean <- na.omit(black)
data_neutral_clean <- na.omit(neut)

#Minimum time is 239 second why is lag 30
#All ACF graphs are very similar
acf(data_positive_clean)
acf(data_negative_clean)
acf(data_black_clean)
acf(data_neutral_clean)


# Check for NAs in the entire dataset
#sum(is.na(Pos))

# Check for NAs in specific columns
#sum(is.na(Pos$EDA_Value))
#sum(is.na(Pos$Time))


#Staple Positve PLUS all the rows where black precedes the time (780-1075)
black_staple <- data_black_clean[data_black_clean$Time >= 780 & data_black_clean$Time <= 1075, ]
recovery_pos <- rbind(data_positive_clean,black_staple)
acf(recovery_pos)
