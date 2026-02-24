# Get all column names
may <- colnames(df)


# Calculate number of chunks
n_cols <- ceiling(nrow(df) / 40)

# Create an empty list to store reshaped columns
df_reshaped_list <- lapply(may, function(col_name) {
  data.frame(
    matrix(c(df[[col_name]], rep(NA, n_cols * 40 - nrow(df))), 
           nrow = 40, 
           ncol = n_cols, 
           byrow = FALSE)
  )
})

# Combine all reshaped columns
df_reshaped <- do.call(cbind, df_reshaped_list)

# Create column names: original_col_10s, original_col_20s, etc.
new_colnames <- unlist(lapply(may, function(col_name) {
  paste0(col_name, "_", seq(10, n_cols * 10, by = 10), "s")
}))

colnames(df_reshaped) <- new_colnames

head(df_reshaped)

df_transposed <- as.data.frame(t(df_reshaped))

head(df_transposed)
