#!/usr/bin/env Rscript
# =============================================================================
# analyze_decoder_performance.R
#
# Statistical analysis of neural feature decoding performance.
# Tests whether feature set (NFR, SBP-RAW, DWT-DB4, MAV) has a significant
# effect on decoding RMSE and correlation after controlling for participant,
# session (repeated measures), DOF config, and days since implant.
#
# Usage:
#   Rscript scripts/r/analyze_decoder_performance.R \
#     --metrics reports/aggregate_metrics.csv \
#     --output-dir reports/figures/stats/
#
# Dependencies:
#   install.packages(c("tidyverse", "lme4", "lmerTest", "broom.mixed",
#                       "emmeans", "flexplot", "patchwork", "argparse"))
# =============================================================================

library(tidyverse)
library(lme4)
library(lmerTest)    # p-values for lmer via Satterthwaite
library(broom.mixed) # tidy() for mixed models
library(emmeans)     # estimated marginal means for pairwise comparisons
library(patchwork)   # combine ggplots
library(argparse)

# Optional: flexplot for visual model summaries
has_flexplot <- require(flexplot, quietly = TRUE)

# --- CLI Arguments ---
parser <- ArgumentParser(description = "Analyze decoder performance")
parser$add_argument("--metrics", required = TRUE, help = "Path to aggregate_metrics.csv")
parser$add_argument("--output-dir", default = "reports/figures/stats/", help = "Output directory")
parser$add_argument("--filter", default = "intended",
                    choices = c("intended", "unintended", "all"),
                    help = "Filter by intended/unintended DOFs")
args <- parser$parse_args()

dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# 1. Load and Prepare Data
# =============================================================================
cat("Loading data from", args$metrics, "\n")

df_raw <- read_csv(args$metrics, show_col_types = FALSE)

# Apply intent filter
df <- df_raw %>%
  filter(!null_result) %>%
  filter(!is.na(rmse), !is.na(correlation))

if (args$filter == "intended") {
  df <- df %>% filter(intended == TRUE)
} else if (args$filter == "unintended") {
  df <- df %>% filter(intended == FALSE)
}

# Convert to factors with informative levels
df <- df %>%
  mutate(
    feature_set = factor(feature_set, levels = c("NFR", "SBP-RAW", "DWT-DB4", "MAV")),
    dof_config = factor(dof_config, levels = c("3DOF", "6DOF", "8DOF")),
    participant_id = factor(participant_id),
    job_id = factor(job_id),
    dof = factor(dof),
  )

cat(sprintf(
  "Data: %d rows, %d sessions, %d participants, %d feature sets, %d DOF configs\n",
  nrow(df),
  n_distinct(df$job_id),
  n_distinct(df$participant_id),
  n_distinct(df$feature_set),
  n_distinct(df$dof_config)
))

# =============================================================================
# 2. Exploratory Visualization
# =============================================================================
cat("Generating plots...\n")

# --- 2a. Faceted scatter: RMSE over time ---
p_rmse_time <- df %>%
  group_by(job_id, participant_id, days_since_implant, feature_set, dof_config) %>%
  summarise(
    rmse_mean = mean(rmse, na.rm = TRUE),
    rmse_se = sd(rmse, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = days_since_implant, y = rmse_mean, color = feature_set)) +
  geom_point(alpha = 0.4, size = 1.2) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 0.8, span = 0.5) +
  facet_grid(participant_id ~ dof_config, scales = "free_y") +
  scale_color_manual(values = c(
    "NFR" = "#E53935", "SBP-RAW" = "#1E88E5",
    "DWT-DB4" = "#43A047", "MAV" = "#FB8C00"
  )) +
  labs(
    x = "Days Since Implant",
    y = "RMSE (session mean)",
    color = "Feature Set",
    title = sprintf("Decoding RMSE Over Time (%s DOFs)", args$filter)
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(args$output_dir, sprintf("rmse_over_time_%s.png", args$filter)),
  p_rmse_time, width = 12, height = 8, dpi = 200
)

# --- 2b. Faceted scatter: Correlation over time ---
p_corr_time <- df %>%
  group_by(job_id, participant_id, days_since_implant, feature_set, dof_config) %>%
  summarise(
    corr_mean = mean(correlation, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = days_since_implant, y = corr_mean, color = feature_set)) +
  geom_point(alpha = 0.4, size = 1.2) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 0.8, span = 0.5) +
  facet_grid(participant_id ~ dof_config, scales = "free_y") +
  scale_color_manual(values = c(
    "NFR" = "#E53935", "SBP-RAW" = "#1E88E5",
    "DWT-DB4" = "#43A047", "MAV" = "#FB8C00"
  )) +
  labs(
    x = "Days Since Implant",
    y = "Pearson Correlation (session mean)",
    color = "Feature Set",
    title = sprintf("Decoding Correlation Over Time (%s DOFs)", args$filter)
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(args$output_dir, sprintf("correlation_over_time_%s.png", args$filter)),
  p_corr_time, width = 12, height = 8, dpi = 200
)

# --- 2c. Boxplots: Feature set comparison ---
p_box <- df %>%
  ggplot(aes(x = feature_set, y = rmse, fill = feature_set)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
  facet_grid(participant_id ~ dof_config, scales = "free_y") +
  scale_fill_manual(values = c(
    "NFR" = "#E53935", "SBP-RAW" = "#1E88E5",
    "DWT-DB4" = "#43A047", "MAV" = "#FB8C00"
  )) +
  labs(
    x = "Feature Set",
    y = "RMSE (per trial × DOF)",
    title = sprintf("RMSE Distribution by Feature Set (%s DOFs)", args$filter)
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold"))

ggsave(
  file.path(args$output_dir, sprintf("rmse_boxplot_%s.png", args$filter)),
  p_box, width = 10, height = 8, dpi = 200
)

# =============================================================================
# 3. Linear Mixed Effects Models
# =============================================================================
cat("\n--- Fitting Linear Mixed Effects Models ---\n\n")

# Model 1: Feature set effect on RMSE
# Fixed effects: feature_set, dof_config, days_since_implant, feature_set × days_since_implant
# Random effects: (1 | participant_id / job_id) — sessions nested within participants
#
# The interaction feature_set × days_since_implant tests whether features
# degrade at different rates over time (the core research question).

cat("Model 1: RMSE ~ feature_set * days_since_implant + dof_config + (1 | participant_id/job_id)\n")
m_rmse <- lmer(
  rmse ~ feature_set * scale(days_since_implant) + dof_config +
    (1 | participant_id / job_id),
  data = df,
  REML = TRUE
)

cat("\n--- ANOVA (Type III, Satterthwaite) ---\n")
print(anova(m_rmse, type = 3))

cat("\n--- Fixed Effects ---\n")
print(summary(m_rmse)$coefficients)

cat("\n--- Random Effects Variance ---\n")
print(VarCorr(m_rmse))

# Model 2: Feature set effect on correlation
cat("\n\nModel 2: correlation ~ feature_set * days_since_implant + dof_config + (1 | participant_id/job_id)\n")
m_corr <- lmer(
  correlation ~ feature_set * scale(days_since_implant) + dof_config +
    (1 | participant_id / job_id),
  data = df,
  REML = TRUE
)

cat("\n--- ANOVA (Type III, Satterthwaite) ---\n")
print(anova(m_corr, type = 3))

cat("\n--- Fixed Effects ---\n")
print(summary(m_corr)$coefficients)

# =============================================================================
# 4. Pairwise Comparisons (Estimated Marginal Means)
# =============================================================================
cat("\n--- Pairwise Comparisons: Feature Sets (RMSE) ---\n")
emm_rmse <- emmeans(m_rmse, pairwise ~ feature_set, adjust = "tukey")
print(emm_rmse$contrasts)

cat("\n--- Pairwise Comparisons: Feature Sets (Correlation) ---\n")
emm_corr <- emmeans(m_corr, pairwise ~ feature_set, adjust = "tukey")
print(emm_corr$contrasts)

# Save emmeans plot
p_emm <- plot(emm_rmse$emmeans, comparisons = TRUE) +
  labs(
    x = "Estimated Marginal Mean RMSE",
    y = "Feature Set",
    title = sprintf("Feature Set Comparison — EMM RMSE (%s DOFs)", args$filter)
  ) +
  theme_minimal(base_size = 11)

ggsave(
  file.path(args$output_dir, sprintf("emmeans_rmse_%s.png", args$filter)),
  p_emm, width = 7, height = 4, dpi = 200
)

# =============================================================================
# 5. Interaction Plot: Feature × Time
# =============================================================================
# This answers: "Do features degrade differently over time?"
cat("\n--- Interaction: Feature × Days Since Implant ---\n")
emm_interaction <- emtrends(m_rmse, pairwise ~ feature_set, var = "scale(days_since_implant)")
cat("Slopes per feature set:\n")
print(emm_interaction$emtrends)
cat("\nSlope contrasts:\n")
print(emm_interaction$contrasts)

# =============================================================================
# 6. Flexplot (optional visual model summary)
# =============================================================================
if (has_flexplot) {
  cat("\nGenerating flexplot...\n")

  # Flexplot shows the model-implied relationships with raw data overlaid
  p_flex <- flexplot(
    rmse ~ days_since_implant + feature_set | participant_id,
    data = df,
    method = "lm",
    se = TRUE
  )

  ggsave(
    file.path(args$output_dir, sprintf("flexplot_rmse_%s.png", args$filter)),
    p_flex, width = 12, height = 8, dpi = 200
  )
} else {
  cat("flexplot not installed. Skipping. Install with: install.packages('flexplot')\n")
}

# =============================================================================
# 7. Model Diagnostics
# =============================================================================
cat("\n--- Model Diagnostics ---\n")

# Residual plots
p_diag <- tibble(
  fitted = fitted(m_rmse),
  residuals = residuals(m_rmse)
) %>%
  ggplot(aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.1, size = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "blue", linewidth = 0.5) +
  labs(x = "Fitted Values", y = "Residuals", title = "RMSE Model: Residuals vs Fitted") +
  theme_minimal(base_size = 11)

p_qq <- tibble(residuals = residuals(m_rmse)) %>%
  ggplot(aes(sample = residuals)) +
  stat_qq(alpha = 0.1, size = 0.5) +
  stat_qq_line(color = "red") +
  labs(x = "Theoretical Quantiles", y = "Sample Quantiles", title = "Q-Q Plot") +
  theme_minimal(base_size = 11)

p_diagnostics <- p_diag + p_qq
ggsave(
  file.path(args$output_dir, sprintf("model_diagnostics_%s.png", args$filter)),
  p_diagnostics, width = 12, height = 5, dpi = 200
)

# =============================================================================
# 8. Export Tidy Model Summary
# =============================================================================
tidy_rmse <- tidy(m_rmse, effects = "fixed", conf.int = TRUE)
tidy_corr <- tidy(m_corr, effects = "fixed", conf.int = TRUE)

write_csv(tidy_rmse, file.path(args$output_dir, sprintf("model_rmse_fixed_effects_%s.csv", args$filter)))
write_csv(tidy_corr, file.path(args$output_dir, sprintf("model_corr_fixed_effects_%s.csv", args$filter)))

cat("\n--- Analysis complete ---\n")
cat("Outputs in:", args$output_dir, "\n")