# Function to assign condition based on row index
assign_condition_by_row <- function(row_index) {
  if (row_index >= 1 && row_index <= 239) return("neutral")
  if (row_index >= 240 && row_index <= 539) return("black")
  if (row_index >= 540 && row_index <= 779) return("positive")
  if (row_index >= 780 && row_index <= 1075) return("black")
  if (row_index >= 1076 && row_index <= 1300) return("negative")
  return(NA)  # Or "rest" if you prefer
}

# Apply the function
df$condition <- sapply(1:nrow(df), assign_condition_by_row)

# Convert to factor
df$condition <- factor(df$condition, 
                             levels = c("neutral", "positive", "negative", "black"))
summary(data$condition)

data2 <- data[!is.na(data$condition), ]

data2 <- data[, !names(data) %in% c("SUD_003_25_EDA", "SUD_003_24_EDA", "SUD_003_05_EDA")]

data_long <- data2 %>%
  mutate(Time = row_number()) %>%   # create time index
  pivot_longer(
    cols = -Time,
    names_to = "PatientID",
    values_to = "EDA_Value"
  ) %>%
  mutate(EDA_Value = as.numeric(EDA_Value))
