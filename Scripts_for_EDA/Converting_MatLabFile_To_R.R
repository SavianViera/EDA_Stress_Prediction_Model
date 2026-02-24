#install.packages("R.matlab")
library(R.matlab)

data <- readMat("Data/eda_5m32s_23ptcp_shift.mat")
str(data)

inner <- data$eda.5m32s.23ptcp.shift

# Extract the numeric matrix (1328 timepoints x 23 participants)
eda_matrix <- inner[[1]]  # num [1:1328, 1:23]

# Extract participant IDs as a clean character vector
ids <- sapply(inner[[2]], function(x) x[[1]][1, 1])
# Result: c("SUD_003_01_EDA", "SUD_003_02_EDA", ...)

# Assign participant IDs as column names
colnames(eda_matrix) <- ids

# Optional: convert to a data frame
eda_df <- as.data.frame(eda_matrix)

dim(eda_df)       # should be 1328 x 23
head(eda_df[, 1:5])
colnames(eda_df)

df <- eda_df
