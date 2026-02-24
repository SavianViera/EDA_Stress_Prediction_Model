# If you want all features in one plot with dodged bars
features_long <- features %>%
  pivot_longer(
    cols = c(mean_eda, sd_eda, max_eda, slope),
    names_to = "feature",
    values_to = "value"
  ) %>%
  mutate(
    feature = case_when(
      feature == "mean_eda" ~ "Mean EDA",
      feature == "sd_eda" ~ "SD EDA",
      feature == "max_eda" ~ "Max EDA",
      feature == "slope" ~ "Slope"
    )
  )

ggplot(features_long, aes(x = condition, y = value, fill = feature)) +
  geom_bar(stat = "summary", fun = "mean", position = position_dodge(width = 0.8)) +
  stat_summary(fun.data = mean_se, geom = "errorbar", 
               width = 0.2, position = position_dodge(width = 0.8)) +
  labs(
    title = "Scaled Feature Means by Condition",
    x = "Condition",
    y = "Mean Z-score",
    fill = "Feature"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set1")
