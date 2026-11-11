# SfN 2026: data table per participant, using exactly the rows the model uses.
# Filters (same as fit_and_plot_sfn.r): !null_result, rmse and correlation present,
# intended DOFs only, participants with > 1 session (drops P202201).
#
# A "trial" = one test-set trial of a session (original_trial_index). The same
# trials are decoded in every DOF configuration, so trials are counted once per
# session, not once per configuration.
#
# Run from the repo root:  Rscript scripts/r/table_sessions_sfn.r
# Writes reports/stats_sfn2026/final/table_sessions.csv

suppressPackageStartupMessages(library(tidyverse))
subj_label <- c(P201501 = "S5", P201601 = "S6", P201701 = "S7")   # CHECK mapping

raw <- read_csv("reports/aggregated_decoder_metrics.csv", show_col_types = FALSE)
d <- raw %>%
  filter(!null_result, !is.na(rmse), !is.na(correlation), intended) %>%
  group_by(participant_id) %>% filter(n_distinct(job_id) > 1) %>% ungroup()

per_session <- d %>%
  group_by(participant_id, job_id) %>%
  summarise(trials = n_distinct(original_trial_index),
            day = first(days_since_implant),
            date = first(session_date),
            configs = paste(sort(unique(dof_config)), collapse = "/"),
            .groups = "drop")

tab <- per_session %>%
  group_by(participant_id) %>%
  summarise(
    subject = subj_label[first(participant_id)],
    sessions = n(),
    sessions_3DOF = sum(str_detect(configs, "3DOF")),
    sessions_6DOF = sum(str_detect(configs, "6DOF")),
    sessions_8DOF = sum(str_detect(configs, "8DOF")),
    trials_median = median(trials),
    trials_q1 = quantile(trials, 0.25),
    trials_q3 = quantile(trials, 0.75),
    first_date = min(date), last_date = max(date),
    day_first = min(day), day_last = max(day),
    span_days = max(day) - min(day),
    .groups = "drop"
  )

total <- per_session %>%
  summarise(participant_id = "All", subject = "", sessions = n(),
            sessions_3DOF = sum(str_detect(configs, "3DOF")),
            sessions_6DOF = sum(str_detect(configs, "6DOF")),
            sessions_8DOF = sum(str_detect(configs, "8DOF")),
            trials_median = median(trials),
            trials_q1 = quantile(trials, 0.25), trials_q3 = quantile(trials, 0.75))
out <- bind_rows(tab, total)

dir.create("reports/stats_sfn2026/final", showWarnings = FALSE, recursive = TRUE)
write_csv(out, "reports/stats_sfn2026/final/table_sessions.csv")
print(out %>% select(participant_id, subject, sessions, sessions_3DOF, sessions_6DOF,
                     sessions_8DOF, trials_median, trials_q1, trials_q3,
                     first_date, last_date, span_days), width = Inf)

# Cross-checks against the model's counts
cat(sprintf("\nRows: %d | sessions: %d | model-style trials (job x config x trial): %d\n",
            nrow(d), n_distinct(d$job_id),
            n_distinct(paste(d$job_id, d$dof_config, d$original_trial_index))))
cat("Note: 'days' are days since each participant's first recorded session in the",
    "\nmetrics file (implant dates in participant_info.yaml are placeholders, issue #6).\n")

