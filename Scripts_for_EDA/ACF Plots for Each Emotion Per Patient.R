# Example for one participant
Pos_01 <- data2$SUD_003_01_EDA[data2$condition == "positive"]
acf_pos_01 <- acf(Pos_01, na.action = na.pass, plot = TRUE, 
                  main = "ACF for Participant 01 (Positive Condition)")

# Or for multiple participants
par(mfrow = c(3, 3))  # Adjust grid based on number of participants
for(i in 1:31) {
  col_name <- paste0("SUD_003_", sprintf("%02d", i), "_EDA")
  if(col_name %in% names(data2)) {
    pos_data <- data2[[col_name]][data2$condition == "positive"]
    acf(pos_data, na.action = na.pass, 
        main = paste("Participant", i, "(Positive)"))
  }
}
#Fdata2$condition
#Negative Cycle
# Example for one participant
Neg_01 <- data2$SUD_003_01_EDA[data2$condition == "negative"]
acf_Neg_01 <- acf(Neg_01, na.action = na.pass, plot = TRUE, 
                  main = "ACF for Participant 01 (Negative Condition)")

# Or for multiple participants
par(mfrow = c(3, 3))  # Adjust grid based on number of participants
for(i in 1:31) {
  col_name <- paste0("SUD_003_", sprintf("%02d", i), "_EDA")
  if(col_name %in% names(data2)) {
    neg_data <- data2[[col_name]][data2$condition == "negative"]
    acf(neg_data, na.action = na.pass, 
        main = paste("Participant", i, "(Negative)"))
  }
}

#Neutral Cycle

# Or for multiple participants
par(mfrow = c(3, 3))  # Adjust grid based on number of participants
for(i in 1:31) {
  col_name <- paste0("SUD_003_", sprintf("%02d", i), "_EDA")
  if(col_name %in% names(data2)) {
    neu_data <- data2[[col_name]][data2$condition == "neutral"]
    acf(neu_data, na.action = na.pass, 
        main = paste("Participant", i, "(Neutral)"))
  }
}

#Black Cycle
par(mfrow = c(3, 3))  # Adjust grid based on number of participants
for(i in 1:31) {
  col_name <- paste0("SUD_003_", sprintf("%02d", i), "_EDA")
  if(col_name %in% names(data2)) {
    black_data <- data2[[col_name]][data2$condition == "black"]
    acf(black_data, na.action = na.pass, 
        main = paste("Participant", i, "(Black)"))
  }
}
