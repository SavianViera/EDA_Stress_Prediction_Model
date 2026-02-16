# Get all column names
may <- colnames(df)

#Window size
m <- 10

# Calculate number of chunks
n_cols2 <- ceiling(nrow(df) / (m*4))

# Create an empty list to store reshaped columns
df_reshaped_list2 <- lapply(may, function(col_name) {
  data.frame(
    matrix(c(df[[col_name]], rep(NA, n_cols * (m*4) - nrow(df))), 
           nrow = m*4, 
           ncol = n_cols, 
           byrow = FALSE)
  )
})

# Combine all reshaped columns
df_reshaped2 <- do.call(cbind, df_reshaped_list2)

# Create column names: original_col_10s, original_col_20s, etc.
new_colnames2 <- unlist(lapply(may, function(col_name) {
  paste0(col_name, "_", seq(m, n_cols * m, by = m), "s")
}))

colnames(df_reshaped2) <- new_colnames

head(df_reshaped2)

df_transposed2 <- as.data.frame(t(df_reshaped2))

head(df_transposed2)
