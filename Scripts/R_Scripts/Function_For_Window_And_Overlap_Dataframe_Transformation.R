# 2nd r script
# some room for window_size and overlap
current_path <- rstudioapi::getActiveDocumentContext()$path
current_path = dirname(current_path)
setwd(current_path)
df <- read.csv("../../Data/Processed/MatLab_EDA_For_R.csv")

# Function to assign condition based on row index
markers = c(60, # end of n
            2*60+15, # end of b
            3*60+15, # end of p
            4*60+29, # end of b
            5*60+26)*4
assign_condition_by_row <- function(row_index) {
  if (row_index >= 1+240 && row_index <= markers[1]-1+240) return("neutral")
  if (row_index >= markers[1]+240 && row_index <= markers[2]-1+240) return("black")
  if (row_index >= markers[2]+240 && row_index <= markers[3]-1+240) return("positive")
  if (row_index >= markers[3]+240 && row_index <= markers[4]-1+240) return("black")
  if (row_index >= markers[4]+240 ) return("negative")
  return(NA)  # Or "rest" if you prefer
}

df$condition <- sapply(1:nrow(df), assign_condition_by_row)

df <- df[!is.na(df$condition), ]




# Convert to factor
df$condition <- factor(df$condition, 
                       levels = c("neutral", "positive", "negative", "black"))

# Get all column names (excluding 'condition' if it exists)
colnames <- colnames(df)[colnames(df) != "condition"]


reshape_by_condition <- function(df, window_size = 10, overlap = 0) {
  
  chunk_size <- window_size * 4
  
  colnames_data <- colnames(df)[colnames(df) != "condition"]
  
  rle_cond <- rle(as.character(df$condition))
  block_ends <- cumsum(rle_cond$lengths)
  block_starts <- c(1, head(block_ends, -1) + 1)
  
  chunk_list <- mapply(function(start, end, cond) {
    row_indices <- start:end
    n_rows <- length(row_indices)
    n_chunks <- ceiling(n_rows / chunk_size)
    
    padded <- c(row_indices, rep(NA, n_chunks * chunk_size - n_rows))
    chunks <- split(padded, ceiling(seq_along(padded) / chunk_size))
    
    list(chunks = chunks, condition = cond, n_chunks = n_chunks)
  }, block_starts, block_ends, rle_cond$values, SIMPLIFY = FALSE)
  
  if (overlap > 0) {
    for (i in seq_len(length(chunk_list) - 1)) {
      current_block <- chunk_list[[i]]
      next_block_start <- block_starts[i + 1]
      
      # Get overlap indices, capped to actual next block size
      overlap_indices <- next_block_start:min(next_block_start + overlap - 1, block_ends[i + 1])
      
      last_chunk <- current_block$chunks[[current_block$n_chunks]]
      real_indices <- last_chunk[!is.na(last_chunk)]
      
      # Combine real + overlap, then pad to chunk_size if needed, or allow extension
      combined <- c(real_indices, overlap_indices)
      
      if (length(combined) <= chunk_size) {
        # Fits within chunk — pad remainder with NA
        last_chunk <- c(combined, rep(NA, chunk_size - length(combined)))
      } else {
        # Exceeds chunk size — keep as-is (no padding needed)
        last_chunk <- combined
      }
      
      chunk_list[[i]]$chunks[[current_block$n_chunks]] <- last_chunk
    }
  }
  
  all_chunks <- do.call(c, lapply(chunk_list, `[[`, "chunks"))
  all_conditions <- unlist(lapply(chunk_list, function(b) rep(b$condition, b$n_chunks)))
  
  n_cols2 <- length(all_chunks)
  
  df_reshaped_list2 <- lapply(colnames_data, function(col_name) {
    vals <- df[[col_name]]
    mat <- sapply(all_chunks, function(idx) {
      out <- rep(NA, chunk_size)
      real <- !is.na(idx)
      # Handle case where chunk is longer than chunk_size due to overlap
      n_real <- sum(real)
      out <- rep(NA, max(chunk_size, length(idx)))
      out[seq_along(idx)[real]] <- vals[idx[real]]
      out[seq_len(chunk_size)]  # Trim back to chunk_size for consistency
    })
    as.data.frame(mat)
  })
  
  df_reshaped2 <- do.call(cbind, df_reshaped_list2)
  
  new_colnames2 <- unlist(lapply(colnames_data, function(col_name) {
    paste0(col_name, "_", seq(window_size, n_cols2 * window_size, by = window_size), "s")
  }))
  colnames(df_reshaped2) <- new_colnames2
  
  df_transposed2 <- as.data.frame(t(df_reshaped2))
  
  df_transposed2$condition <- factor(all_conditions, levels = c("neutral", "positive", "negative", "black"))
  
  chunk_row_indices <- sapply(all_chunks, function(idx) {
    paste(idx[!is.na(idx)], collapse = ", ")
  })
  df_transposed2$source_rows <- chunk_row_indices
  
  return(df_transposed2)
}

# Default: window = 10, no overlap
result <- reshape_by_condition(df)

write.csv(result, file = "../../Data/Processed/Window_10_No_Overlap.csv", row.names = TRUE)

# Window of 20, no overlap
result <- reshape_by_condition(df, window_size = 20)

# Window of 10, allow 2 rows of overlap at condition boundaries
result <- reshape_by_condition(df, window_size = 10, overlap = 2)

# Both changed
result <- reshape_by_condition(df, window_size = 20, overlap = 5)




