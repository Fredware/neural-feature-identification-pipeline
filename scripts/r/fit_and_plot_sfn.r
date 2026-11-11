# SfN 2026: final model + results figures (all from one fitted model).
#
# Run from the repo root on a compute node:
#   Rscript scripts/r/fit_and_plot_sfn.R
# Outputs -> reports/stats_sfn2026/final/
#   tables_*.csv            numbers for the poster text
#   fig_complexity.pdf/svg/png  feature x decoder complexity (all sessions)
#   fig_time.pdf/svg/png        feature x days, per participant
#   fig_summary.pdf/svg/png     overall + per-participant error per feature
#
# Uses every decoded session: no session is dropped because it lacks a
# DOF configuration. The model handles unequal session sets per config.

suppressPackageStartupMessages({
  library(tidyverse); library(lme4); library(emmeans)
})
emm_options(lmer.df = "asymptotic", msg.interaction = FALSE)
has_svglite <- requireNamespace("svglite", quietly = TRUE)
if (!has_svglite) message("svglite not installed: SVG text may become outlines. ",
                          "Install once with install.packages('svglite').")

out_dir <- "reports/stats_sfn2026/final"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- labels (CHECK: participant -> paper subject ID) -----------------------
subj_label <- c(P201501 = "S5", P201601 = "S6", P201701 = "S7", P202201 = "S8")
feat_levels <- c("NFR", "SBP-RAW", "DWT-DB4", "MAV")
# Intended RMSE of a do-nothing decoder (always 0), scored by the pipeline's own
# evaluate script on session 36 (3/6/8 DOF: .552/.557/.562). One session only.
null_ref <- 0.557
# One RMSE axis range for every figure, so panels are visually comparable.
# coord_cartesian() only zooms: nothing outside is dropped from any statistic.
rmse_lims <- c(0.3, 0.8)
report_hidden <- function(v, name) {
  n_out <- sum(v < rmse_lims[1] | v > rmse_lims[2], na.rm = TRUE)
  cat(sprintf("%s: %d of %d session points outside the y-range (still in the stats)\n",
              name, n_out, sum(!is.na(v))))
}
feat_cols   <- c("NFR" = "#E69F00", "SBP-RAW" = "#56B4E9",
                 "DWT-DB4" = "#009E73", "MAV" = "#CC79A7")   # Okabe-Ito

# ---- data ------------------------------------------------------------------
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
  # Time is centred within each participant: "average" comparisons are made
  # at each participant's own mid-study point, never extrapolated outside the
  # days they were recorded (P201501 spans only ~65 days). Slopes unchanged.
  group_by(participant) %>%
  mutate(days_mid = mean(days_since_implant),
         days100  = (days_since_implant - days_mid) / 100) %>%   # per 100 days
  ungroup()
days_mid <- d %>% distinct(participant, days_mid)
cat(sprintf("%d rows | %d sessions | %d participants\n",
            nrow(d), n_distinct(d$session), n_distinct(d$participant)))

re <- "(1 | session) + (1 | trial) + (1 | session:feature_set) +
       (1 | session:feature_set:dof_config)"

# Model A: one complexity effect and one time effect per feature, shared by
#          all participants.
# Model B: lets each participant have their own complexity and time pattern
#          per feature (what the exploratory plots suggest).
fA <- as.formula(paste(
  "rmse ~ feature_set * days100 + feature_set * dof_config +",
  "feature_set * participant +", re))
fB <- as.formula(paste(
  "rmse ~ feature_set * dof_config * participant +",
  "feature_set * days100 * participant +", re))

ctrl <- lmerControl(calc.derivs = FALSE)
mA <- lmer(fA, data = d, REML = FALSE, control = ctrl)
mB <- lmer(fB, data = d, REML = FALSE, control = ctrl)

cmp <- anova(mA, mB)
print(cmp)
p_AB <- cmp$`Pr(>Chisq)`[2]
m <- if (p_AB < 0.05) mB else mA
model_used <- if (p_AB < 0.05) "B (participant-specific patterns)" else "A (shared patterns)"
cat("\nModel used for figures:", model_used, "\n")
cat("Singular fit:", isSingular(m), "\n")
print(VarCorr(m))   # a component at ~0 is what makes the fit 'singular'

# ---- tests: likelihood-ratio test for each highest-order term -------------
lrt <- drop1(m, test = "Chisq") %>% as.data.frame() %>%
  rownames_to_column("term") %>% filter(term != "<none>")
print(lrt)
write_csv(lrt, file.path(out_dir, "tables_lrt.csv"))
write_csv(as.data.frame(cmp) %>% rownames_to_column("model"),
          file.path(out_dir, "tables_modelA_vs_B.csv"))

# Effect-size scale: total SD (all variance components + residual)
vc <- as.data.frame(VarCorr(m))
sd_total <- sqrt(sum(vc$vcov))
sd_resid <- sigma(m)
cat(sprintf("SD total = %.3f | SD residual = %.3f\n", sd_total, sd_resid))

# ---- feature means and pairwise differences -------------------------------
# Overall = average over participants weighted by their number of rows.
emm_all <- emmeans(m, ~ feature_set, weights = "proportional")
emm_pp  <- emmeans(m, ~ feature_set | participant, weights = "proportional")

pairs_all <- pairs(emm_all, adjust = "tukey") %>% as.data.frame() %>%
  mutate(d_total = estimate / sd_total, d_resid = estimate / sd_resid)
pairs_pp  <- pairs(emm_pp, adjust = "tukey") %>% as.data.frame() %>%
  mutate(d_total = estimate / sd_total)

write_csv(as.data.frame(confint(emm_all)), file.path(out_dir, "tables_emm_overall.csv"))
write_csv(as.data.frame(confint(emm_pp)),  file.path(out_dir, "tables_emm_by_participant.csv"))
write_csv(pairs_all, file.path(out_dir, "tables_pairs_overall.csv"))
write_csv(pairs_pp,  file.path(out_dir, "tables_pairs_by_participant.csv"))
print(confint(emm_all)); print(pairs_all)

# ---- degradation slopes (RMSE per 100 days) -------------------------------
slope_all <- emtrends(m, ~ feature_set, var = "days100", weights = "proportional")
slope_pp  <- emtrends(m, ~ feature_set | participant, var = "days100")
write_csv(as.data.frame(confint(slope_all)), file.path(out_dir, "tables_slopes_overall.csv"))
write_csv(as.data.frame(confint(slope_pp)),  file.path(out_dir, "tables_slopes_by_participant.csv"))
write_csv(as.data.frame(pairs(slope_all, adjust = "tukey")),
          file.path(out_dir, "tables_slope_pairs_overall.csv"))
print(confint(slope_all))

# ---- plotting helpers -------------------------------------------------------
lab_p <- function(p) ifelse(p == "All", "All participants",
                            paste0(subj_label[as.character(p)], " (", p, ")"))
theme_poster <- theme_classic(base_size = 16, base_family = "Arial") +
  theme(legend.position = "top", strip.background = element_blank(),
        strip.text = element_text(face = "bold", size = 16),
        panel.grid.major.y = element_line(colour = "grey92"))
# PDF fallback: "Arial" is mapped to Helvetica metrics (near-identical). Text
# stays live text in Illustrator (tagged Helvetica): select all -> set Arial.
if (!"Arial" %in% names(pdfFonts()))
  pdfFonts(Arial = Type1Font("Arial", pdfFonts()$Helvetica$metrics))
save_fig <- function(p, name, w, h) {
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = w, height = h,
         device = cairo_pdf_or_pdf)
  if (has_svglite) ggsave(file.path(out_dir, paste0(name, ".svg")), p,
                          width = w, height = h, device = svglite::svglite)
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = w, height = h, dpi = 150)
}

cairo_pdf_or_pdf <- function(filename, ...) grDevices::pdf(filename, ..., useDingbats = FALSE)

# Session means (one point per session x feature x config) for the backdrop
sess <- d %>%
  group_by(participant, session, days_since_implant, feature_set, dof_config) %>%
  summarise(rmse = mean(rmse), .groups = "drop")

# ---- Figure: complexity ----------------------------------------------------
emm_cx_pp  <- emmeans(m, ~ feature_set * dof_config | participant,
                      weights = "proportional") %>% confint() %>% as.data.frame()
emm_cx_all <- emmeans(m, ~ feature_set * dof_config, weights = "proportional") %>%
  confint() %>% as.data.frame() %>% mutate(participant = "All")
emm_cx <- bind_rows(emm_cx_all, emm_cx_pp %>% mutate(participant = as.character(participant))) %>%
  mutate(panel = factor(lab_p(participant), levels = lab_p(c("All", levels(d$participant)))))
sess_cx <- bind_rows(sess %>% mutate(participant = "All"),
                     sess %>% mutate(participant = as.character(participant))) %>%
  mutate(panel = factor(lab_p(participant), levels = levels(emm_cx$panel)))

pd <- position_dodge(width = 0.45)
p_cx <- ggplot(emm_cx, aes(dof_config, emmean, colour = feature_set, group = feature_set)) +
  geom_point(data = sess_cx, aes(y = rmse), alpha = 0.10, size = 0.9, stroke = 0,
             position = position_jitterdodge(jitter.width = 0.10, dodge.width = 0.45)) +
  geom_hline(yintercept = null_ref, linetype = "dashed", colour = "grey40") +
  geom_line(position = pd, linewidth = 1) +
  geom_pointrange(aes(ymin = asymp.LCL, ymax = asymp.UCL), position = pd,
                  size = 0.5, linewidth = 1) +
  facet_wrap(~ panel, nrow = 1) +
  scale_colour_manual(values = feat_cols) +
  coord_cartesian(ylim = rmse_lims) +
  labs(x = "Decoder complexity (degrees of freedom)", y = "RMSE (intended DOFs)",
       colour = NULL) +
  theme_poster
report_hidden(sess_cx$rmse, "fig_complexity")
save_fig(p_cx, "fig_complexity", 15, 5.5)

# ---- Figure: time ----------------------------------------------------------
grid_pp <- d %>% group_by(participant) %>%
  summarise(lo = min(days100), hi = max(days100), .groups = "drop")
pred_t <- pmap_dfr(grid_pp, function(participant, lo, hi) {
  g <- seq(lo, hi, length.out = 40)
  emmeans(m, ~ feature_set * days100,
          at = list(days100 = g, participant = as.character(participant)),
          weights = "proportional") %>% confint() %>% as.data.frame() %>%
    mutate(participant = participant)
}) %>% left_join(days_mid, by = "participant") %>%
  mutate(days_plot = days100 * 100 + days_mid,
         panel = factor(lab_p(as.character(participant)),
                              levels = lab_p(levels(d$participant))))
sess_t <- sess %>% group_by(participant, session, days_since_implant, feature_set) %>%
  summarise(rmse = mean(rmse), .groups = "drop") %>%
  mutate(panel = factor(lab_p(as.character(participant)), levels = levels(pred_t$panel)))

p_t <- ggplot(pred_t, aes(days_plot, emmean, colour = feature_set, fill = feature_set)) +
  geom_point(data = sess_t, inherit.aes = FALSE,
             mapping = aes(x = days_since_implant, y = rmse, colour = feature_set),
             alpha = 0.25, size = 1.1, stroke = 0) +
  geom_hline(yintercept = null_ref, linetype = "dashed", colour = "grey40") +
  geom_ribbon(aes(ymin = asymp.LCL, ymax = asymp.UCL), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ panel, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = feat_cols, aesthetics = c("colour", "fill")) +
  coord_cartesian(ylim = rmse_lims) +
  labs(x = "Days since first session", y = "RMSE (intended DOFs)",
       colour = NULL, fill = NULL) +
  theme_poster
report_hidden(sess_t$rmse, "fig_time")
save_fig(p_t, "fig_time", 15, 5.5)

# ---- Figure: summary (overall + per participant) ---------------------------
sum_df <- bind_rows(
  as.data.frame(confint(emm_all)) %>% mutate(participant = "All"),
  as.data.frame(confint(emm_pp)) %>% mutate(participant = as.character(participant))
) %>% mutate(group = factor(lab_p(participant),
                            levels = lab_p(c("All", levels(d$participant)))))

p_s <- ggplot(sum_df, aes(group, emmean, colour = feature_set)) +
  scale_x_discrete() +   # declare x as discrete before the numeric annotate() position
  geom_hline(yintercept = null_ref, linetype = "dashed", colour = "grey40") +
  annotate("text", x = 0.5, y = null_ref, label = "do-nothing decoder (approx.)",
           hjust = 0, vjust = -0.5, size = 4, family = "Arial", colour = "grey30") +
  geom_pointrange(aes(ymin = asymp.LCL, ymax = asymp.UCL),
                  position = position_dodge(width = 0.6), size = 0.6, linewidth = 1.1) +
  scale_colour_manual(values = feat_cols) +
  labs(x = NULL, y = "RMSE (intended DOFs)\nmodel estimate \u00b1 95% CI", colour = NULL) +
  coord_cartesian(ylim = rmse_lims) +
  theme_poster
save_fig(p_s, "fig_summary", 9, 5.5)

# ---- Figure: intended vs unintended error (descriptive) ---------------------
# Intended RMSE: error on the DOF the participant was asked to move.
# Unintended RMSE: error on the DOFs that should stay still in that trial.
# A do-nothing decoder sits at (0, ~0.557); a perfect decoder at (0, 0).
tr <- raw %>%
  filter(!null_result, !is.na(rmse)) %>%
  group_by(participant_id) %>% filter(n_distinct(job_id) > 1) %>% ungroup() %>%
  mutate(feature_set = factor(feature_set, levels = feat_levels),
         kind = if_else(intended, "intended", "unintended")) %>%
  group_by(participant_id, job_id, feature_set, kind) %>%
  summarise(rmse = mean(rmse), .groups = "drop") %>%
  pivot_wider(names_from = kind, values_from = rmse) %>%
  filter(!is.na(intended), !is.na(unintended))
tr_long <- bind_rows(tr %>% mutate(participant_id = "All"), tr) %>%
  mutate(panel = factor(lab_p(participant_id),
                        levels = lab_p(c("All", levels(d$participant)))))
tr_sum <- tr_long %>% group_by(panel, feature_set) %>%
  summarise(n = n(),
            ui = mean(unintended), ui_se = sd(unintended) / sqrt(n),
            it = mean(intended),   it_se = sd(intended) / sqrt(n), .groups = "drop")
write_csv(tr_sum, file.path(out_dir, "tables_intended_vs_unintended.csv"))

p_tr <- ggplot(tr_sum, aes(ui, it, colour = feature_set)) +
  geom_point(data = tr_long, aes(unintended, intended), alpha = 0.15, size = 0.9,
             stroke = 0) +
  geom_errorbar(aes(ymin = it - 1.96 * it_se, ymax = it + 1.96 * it_se), width = 0) +
  geom_errorbar(aes(xmin = ui - 1.96 * ui_se, xmax = ui + 1.96 * ui_se), width = 0,
                orientation = "y") +
  geom_point(size = 3) +
  annotate("point", x = 0, y = null_ref, shape = 4, size = 4, stroke = 1.5) +
  annotate("text", x = 0.005, y = null_ref, label = "do nothing", hjust = 0,
           vjust = -0.8, size = 3.8, family = "Arial") +
  facet_wrap(~ panel, nrow = 1) +
  scale_colour_manual(values = feat_cols) +
  coord_cartesian(xlim = c(0, max(tr_long$unintended, na.rm = TRUE)), ylim = rmse_lims) +
  labs(x = "Unintended RMSE (DOFs that should stay still)",
       y = "Intended RMSE (DOF being moved)", colour = NULL) +
  theme_poster
report_hidden(tr_long$intended, "fig_intended_vs_unintended")
save_fig(p_tr, "fig_intended_vs_unintended", 15, 5.5)
print(tr_sum %>% filter(panel == "All participants"))

# ---- Confirmatory tests (the abstract's hypotheses), with family-wise corrections ----
# Confirmatory model = model A: feature x time + feature x complexity (+ feature x
# participant, needed because participant is fixed). This keeps the abstract's
# structure; model B (participant-specific patterns) is reported as exploratory.
# Families and corrections:
#   F1  model terms (likelihood-ratio tests, one per hypothesis)          no correction (one test per question)
#   F2  feature pairwise differences in mean RMSE (6 pairs)                 Tukey
#   F3  does each feature degrade over time? slope vs 0 (4 tests)          Holm
#   F4  do features degrade at different rates? slope differences (6 pairs) Tukey
saveRDS(list(mA = mA, mB = mB), file.path(out_dir, "models_A_B.rds"))
star <- function(p) as.character(cut(p, c(-Inf, .001, .01, .05, Inf), c("***", "**", "*", "ns")))
tidyp <- function(df) df %>% mutate(p.value = signif(p.value, 2), stars = star(p.value))
mC <- mA

cat("\n== F1: likelihood-ratio tests (confirmatory model A) ==\n")
f1 <- drop1(mC, test = "Chisq") %>% as.data.frame() %>% rownames_to_column("term") %>%
  filter(term != "<none>") %>% rename(df = npar, chisq = LRT, p.value = `Pr(Chi)`) %>% tidyp()
print(f1, row.names = FALSE)
cat("Model A vs B (exploratory: participant-specific patterns):\n"); print(cmp)

emmC <- emmeans(mC, ~ feature_set, weights = "proportional")
trC  <- emtrends(mC, ~ feature_set, var = "days100", weights = "proportional")

cat("\n== F2: feature pairs, mean RMSE (Tukey, 6 comparisons) ==\n")
f2 <- as.data.frame(pairs(emmC, adjust = "tukey")) %>%
  mutate(d_total = estimate / sd_total) %>% tidyp()
print(confint(emmC)); print(f2, row.names = FALSE)

cat("\n== F3: degradation per feature, slope vs 0 per 100 days (Holm, 4 tests) ==\n")
f3 <- as.data.frame(test(trC, adjust = "holm")) %>% tidyp()
print(f3, row.names = FALSE)

cat("\n== F4: slope differences between features (Tukey, 6 comparisons) ==\n")
f4 <- as.data.frame(pairs(trC, adjust = "tukey")) %>% tidyp()
print(f4, row.names = FALSE)

write_csv(f1, file.path(out_dir, "confirm_F1_lrt.csv"))
write_csv(f2, file.path(out_dir, "confirm_F2_feature_pairs.csv"))
write_csv(f3, file.path(out_dir, "confirm_F3_slopes_vs_zero.csv"))
write_csv(f4, file.path(out_dir, "confirm_F4_slope_pairs.csv"))

# Exploratory (participant-level), from model B, one family across all 18 comparisons
cat("\n== Exploratory: feature pairs per participant (model B; Holm across all 18) ==\n")
ex <- as.data.frame(pairs(emm_pp, adjust = "none")) %>%
  mutate(p.value = p.adjust(p.value, "holm")) %>% tidyp()
print(ex, row.names = FALSE)
write_csv(ex, file.path(out_dir, "exploratory_pairs_by_participant.csv"))

cat("\nDone. Outputs in", out_dir, "\n")

