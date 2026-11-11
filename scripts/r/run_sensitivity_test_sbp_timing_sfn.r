# SfN 2026 sensitivity check: does SBP-RAW's frame-timing drift change the confirmatory results?
#
# SBP-RAW frames sit on a fixed 990-sample grid, so in sessions with gaps or a mean frame
# interval above 990 samples SBP drifts relative to the kinematics. The per-session drift at
# the end of the session is in feature_timing_frames.csv (verify_feature_timing_sfn.py).
#
# Two independent checks:
#   1. Dose-response: per session, (SBP RMSE - NFR RMSE) vs drift. If the lag search absorbs
#      the drift, there is no relationship. Control: (DWT RMSE - NFR RMSE) vs the same drift,
#      since neither DWT nor NFR is affected by it.
#   2. Refit confirmatory model A on subsets and compare the SBP contrasts with the poster's:
#        S0 all data (poster)                       S1 SBP rows dropped where drift > 12 frames
#        S2 SBP rows dropped where drift > 5        S3 whole sessions dropped where drift > 12
#        S4 only sessions with drift <= 1 frame (no drift)
#      Same model, same families and corrections as fit_and_plot_sfn.R. The poster analysis
#      stays the primary analysis; these are sensitivity analyses.
#
# Run from the repo root on a compute node (4 model refits; same cost as model A each):
#   Rscript scripts/r/sensitivity_sbp_timing_sfn.R
# Outputs -> reports/stats_sfn2026/final/sensitivity_sbp_*.csv, sensitivity_sbp_dose_response.png

suppressPackageStartupMessages({
  library(tidyverse); library(lme4); library(emmeans)
})
emm_options(lmer.df = "asymptotic", msg.interaction = FALSE)
out_dir <- "reports/stats_sfn2026/final"
feat_levels <- c("NFR", "SBP-RAW", "DWT-DB4", "MAV")

# ---- data: identical to fit_and_plot_sfn.R ----------------------------------------------
raw <- read_csv("reports/aggregated_decoder_metrics.csv", show_col_types = FALSE)
d <- raw %>%
  filter(!null_result, !is.na(rmse), !is.na(correlation), intended) %>%
  group_by(participant_id) %>% filter(n_distinct(job_id) > 1) %>% ungroup() %>%
  mutate(
    participant = factor(participant_id),
    feature_set = factor(feature_set, levels = feat_levels),
    dof_config  = factor(dof_config, levels = c("3DOF", "6DOF", "8DOF")),
    session     = factor(job_id),
    trial       = factor(paste(job_id, dof_config, original_trial_index)),
  ) %>%
  group_by(participant) %>%
  mutate(days_mid = mean(days_since_implant),
         days100  = (days_since_implant - days_mid) / 100) %>%
  ungroup()

# ---- drift per session ------------------------------------------------------------------
frames <- read_csv(file.path(out_dir, "feature_timing_frames.csv"),
                   col_types = cols(session = col_character(), .default = col_guess())) %>%
  transmute(session = session, drift = abs(sbp_offset_end_frames), n_gaps)
d <- d %>% mutate(session_chr = as.character(job_id)) %>%
  left_join(frames, by = c(session_chr = "session"))
missing <- d %>% filter(is.na(drift)) %>% distinct(session_chr) %>% pull()
if (length(missing)) stop("No frame timing for sessions: ", paste(missing, collapse = ", "))

cat("== Sessions by drift at end of session (frames), per participant ==\n")
d %>% distinct(participant, session, drift) %>%
  mutate(band = cut(drift, c(-Inf, 1, 5, 12, Inf),
                    c("<=1 (none)", "1-5", "5-12", ">12 (beyond lag search)"))) %>%
  count(participant, band) %>% pivot_wider(names_from = band, values_from = n, values_fill = 0) %>%
  print()

# ---- 1. dose-response -------------------------------------------------------------------
sess <- d %>% group_by(participant, session, drift, feature_set) %>%
  summarise(rmse = mean(rmse), .groups = "drop") %>%
  pivot_wider(names_from = feature_set, values_from = rmse) %>%
  mutate(`SBP - NFR` = `SBP-RAW` - NFR,
         `SBP - DWT` = `SBP-RAW` - `DWT-DB4`,
         `DWT - NFR (control)` = `DWT-DB4` - NFR)
dose <- map_dfr(c("SBP - NFR", "SBP - DWT", "DWT - NFR (control)"), function(v) {
  dd <- sess %>% filter(!is.na(.data[[v]]))
  fit <- lm(dd[[v]] ~ log1p(dd$drift) + dd$participant)
  co <- summary(fit)$coefficients[2, ]
  sp <- cor.test(dd$drift, dd[[v]], method = "spearman", exact = FALSE)
  tibble(difference = v, n_sessions = nrow(dd),
         slope_per_log1p_frame = signif(co[1], 3), se = signif(co[2], 3), p_slope = signif(co[4], 2),
         spearman_rho = round(unname(sp$estimate), 3), p_spearman = signif(sp$p.value, 2))
})
cat("\n== Dose-response: per-session RMSE difference vs SBP drift (participant-adjusted) ==\n")
cat("Positive slope = SBP gets worse relative to the other feature as drift grows.\n")
print(dose, width = Inf)
write_csv(dose, file.path(out_dir, "sensitivity_sbp_dose_response.csv"))

p <- sess %>%
  pivot_longer(c(`SBP - NFR`, `DWT - NFR (control)`), names_to = "difference", values_to = "delta") %>%
  ggplot(aes(drift + 1, delta, colour = participant)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_point(alpha = 0.8) +
  geom_smooth(aes(group = 1), method = "lm", formula = y ~ x, colour = "black", se = TRUE) +
  scale_x_log10() + facet_wrap(~ difference) +
  labs(x = "SBP drift at end of session + 1 (frames, log scale)",
       y = "Session mean RMSE difference (intended DOFs)",
       title = "Does SBP get worse as its frame drift grows?",
       subtitle = "Left: SBP minus NFR. Right: control (DWT minus NFR; unaffected by SBP drift).") +
  theme_bw(base_size = 11)
ggsave(file.path(out_dir, "sensitivity_sbp_dose_response.png"), p, width = 9, height = 4.5, dpi = 150)

# ---- 2. confirmatory model A on subsets -------------------------------------------------
re <- "(1 | session) + (1 | trial) + (1 | session:feature_set) +
       (1 | session:feature_set:dof_config)"
fA <- as.formula(paste(
  "rmse ~ feature_set * days100 + feature_set * dof_config +",
  "feature_set * participant +", re))
ctrl <- lmerControl(calc.derivs = FALSE)

summarise_fit <- function(m, label, dd) {
  sd_total <- sqrt(sum(as.data.frame(VarCorr(m))$vcov))
  emm <- emmeans(m, ~ feature_set, weights = "proportional")
  tr  <- emtrends(m, ~ feature_set, var = "days100", weights = "proportional")
  bind_rows(
    as.data.frame(emm) %>% transmute(family = "EMM", term = as.character(feature_set),
                                     estimate = emmean, SE, p.value = NA_real_),
    as.data.frame(pairs(emm, adjust = "tukey")) %>%
      transmute(family = "F2 mean diff (Tukey)", term = contrast, estimate, SE, p.value,
                d_total = estimate / sd_total),
    as.data.frame(test(tr, adjust = "holm")) %>%
      transmute(family = "F3 slope vs 0 (Holm)", term = as.character(feature_set),
                estimate = days100.trend, SE, p.value),
    as.data.frame(pairs(tr, adjust = "tukey")) %>%
      transmute(family = "F4 slope diff (Tukey)", term = contrast, estimate, SE, p.value)
  ) %>%
    mutate(analysis = label,
           n_sessions = n_distinct(dd$session),
           n_sessions_with_sbp = n_distinct(dd$session[dd$feature_set == "SBP-RAW"]),
           .before = 1)
}

subsets <- list(
  "S0 all data (poster)"                 = d,
  "S1 SBP rows dropped, drift > 12"      = d %>% filter(!(feature_set == "SBP-RAW" & drift > 12)),
  "S2 SBP rows dropped, drift > 5"       = d %>% filter(!(feature_set == "SBP-RAW" & drift > 5)),
  "S3 whole sessions dropped, drift > 12" = d %>% filter(drift <= 12),
  "S4 only sessions with drift <= 1"     = d %>% filter(drift <= 1)
)
saved <- file.path(out_dir, "models_A_B.rds")

res <- imap_dfr(subsets, function(dd, label) {
  t0 <- Sys.time()
  dd <- droplevels(dd)
  m <- if (startsWith(label, "S0") && file.exists(saved)) readRDS(saved)$mA
       else lmer(fA, data = dd, REML = FALSE, control = ctrl)
  cat(sprintf("\nfitted %-40s %5d rows, %3d sessions (%3d with SBP), %.1f min\n", label, nrow(dd),
              n_distinct(dd$session), n_distinct(dd$session[dd$feature_set == "SBP-RAW"]),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  summarise_fit(m, label, dd)
})
res <- res %>% mutate(stars = as.character(cut(p.value, c(-Inf, .001, .01, .05, Inf),
                                               c("***", "**", "*", "ns"))))
write_csv(res, file.path(out_dir, "sensitivity_sbp_model_A.csv"))

# ---- compact view: everything that involves SBP, side by side ---------------------------
key <- res %>%
  filter(str_detect(term, "SBP")) %>%
  mutate(value = if_else(is.na(p.value), sprintf("%.3f", estimate),
                         sprintf("%+.3f %s", estimate, stars))) %>%
  select(family, term, analysis, value) %>%
  mutate(analysis = str_extract(analysis, "^S\\d")) %>%
  pivot_wider(names_from = analysis, values_from = value)
cat("\n== SBP results across analyses (estimate and significance; S0 = poster) ==\n")
cat("EMM = mean intended RMSE; F2 = difference in mean RMSE; F3/F4 = RMSE change per 100 days.\n")
print(key, n = Inf, width = Inf)
cat("\nS1/S2: SBP rows dropped in drifting sessions; S3: whole drifting sessions dropped;",
    "S4: no-drift sessions only.\nDone. Outputs in", out_dir, "\n")

