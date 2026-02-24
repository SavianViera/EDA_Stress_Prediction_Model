#Create column ID to relate each 10s (Window) to a patient,
#i.e. for every 40 rows df$column = patient_01_10s

#First pivot longer so all the rows are mutated to become columns
#then we subselect each 40 columns to become a new row
#then that row ID will be patient_01_Window+Window
#Start with patient 1 as base case
df_t <- data.frame(df$SUD_003_01_EDA)
p_long <- pivot_wider(df_t$df.SUD_003_01_EDA)



df_n <- data.frame()
window = 1:10
for i in window{
  df_long %>%
    select(ID, Patient_Column, within_segment, Value) %>%
    pivot_wider(
      names_from = within_segment,
      names_prefix = "Signal_",
      values_from = Value
    )
}