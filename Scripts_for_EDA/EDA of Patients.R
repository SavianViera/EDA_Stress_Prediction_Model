data <- read.csv("participants_combined.csv")
# Remove the first 3 rows
data <- data[-c(1:3), ]
library(ggplot2)
# Transpose the data so patients are rows
data_t <- as.data.frame(t(data))
colnames(data_t) <- "Measurement"  # Or "EDA_Value" or "TimePoint1" if you have column names

# Add patient ID column
data_t$PatientID <- rownames(data_t)

# Reset index
data_t <- data_t[, c("PatientID", "Measurement")]
rownames(data_t) <- NULL

# Convert measurement to numeric
data_t$Measurement <- as.numeric(data_t$Measurement)
# Assuming your data has multiple rows (time points)
# Here's how to structure it properly

# First, transpose so patients are rows
data_t <- as.data.frame(t(data))

# Name columns as TimePoint1, TimePoint2, etc.
colnames(data_t) <- paste0("TimePoint", 1:ncol(data_t))

# Add patient ID column
data_t$PatientID <- rownames(data_t)

# Move PatientID to first column
data_t <- data_t[, c("PatientID", paste0("TimePoint", 1:(ncol(data_t)-1)))]

# Convert to long format for plotting
library(tidyr)
data_long <- pivot_longer(data_t, 
                          cols = starts_with("TimePoint"),
                          names_to = "Time",
                          values_to = "EDA_Value")

# Convert to proper types
data_long$EDA_Value <- as.numeric(data_long$EDA_Value)
data_long$Time <- factor(data_long$Time, 
                         levels = paste0("TimePoint", 1:(ncol(data_t)-1)))

ggplot(data = data_long, aes(x = Time, y = EDA_Value)) +
  geom_point() +  # or geom_line() or geom_bar(), etc.
  geom_line(aes(group = PatientID, color = PatientID))

