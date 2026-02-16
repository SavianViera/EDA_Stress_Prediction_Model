df <- read.csv("data/participants_combined.csv")

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

# Get all column names (excluding 'condition' if it exists)
may <- colnames(df)[colnames(df) != "condition"]

# Window size
m <- 10

# Calculate number of chunks
n_cols2 <- ceiling(nrow(df) / (m*4))

# Store the condition for each row before reshaping
condition_vector <- df$condition

# Create an empty list to store reshaped columns
df_reshaped_list2 <- lapply(may, function(col_name) {
  data.frame(
    matrix(c(df[[col_name]], rep(NA, n_cols2 * (m*4) - nrow(df))), 
           nrow = m*4, 
           ncol = n_cols2, 
           byrow = FALSE)
  )
})

# Combine all reshaped columns
df_reshaped2 <- do.call(cbind, df_reshaped_list2)

# Reshape the condition column the same way
condition_reshaped <- matrix(c(as.character(condition_vector), 
                               rep(NA, n_cols2 * (m*4) - length(condition_vector))), 
                             nrow = m*4, 
                             ncol = n_cols2, 
                             byrow = FALSE)

# For each interval (column), find the most common condition
get_mode <- function(x) {
  x <- x[!is.na(x)]
  if(length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

interval_conditions <- apply(condition_reshaped, 2, get_mode)

# Create column names counting by window size
new_colnames2 <- unlist(lapply(may, function(col_name) {
  paste0(col_name, "_", seq(m, n_cols2 * m, by = m), "s")
}))

colnames(df_reshaped2) <- new_colnames2

# Transpose
df_transposed2 <- as.data.frame(t(df_reshaped2))

# Add the condition column
df_transposed2$condition <- factor(interval_conditions, 
                                   levels = c("neutral", "positive", "negative", "black"))

head(df_transposed2)
table(df_transposed2$condition)

