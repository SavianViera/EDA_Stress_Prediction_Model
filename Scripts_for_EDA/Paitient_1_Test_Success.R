# Assuming your dataframe is called df_t
library(dplyr)

# Calculate how many columns you'll have (number of 40-row chunks)
n_cols <- ceiling(nrow(df_t) / 40)

# Reshape: split data into chunks of 40 and make them columns
# Pad with NA if the last chunk is incomplete
df_reshaped <- data.frame(
  matrix(c(df_t$df.SUD_003_01_EDA, rep(NA, n_cols * 40 - nrow(df_t))), 
         nrow = 40, 
         ncol = n_cols, 
         byrow = FALSE)
)

# Create column names: Patient_1_10s, Patient_1_20s, Patient_1_30s, etc.
colnames(df_reshaped) <- paste0("Patient_1_", seq(10, n_cols * 10, by = 10), "s")

# Now the dataframe has 40 rows and n_cols columns
head(df_reshaped)

df_transposed <- as.data.frame(t(df_reshaped))

head(df_transposed)
