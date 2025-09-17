function [trial_starts_idx, trial_stops_idx] = compute_trial_idxs(trial_struct, nip_time_kdf)
trial_starts_nip = [trial_struct.TargOnTS];
trial_starts_idx = project_utils.find_closest_idxs_frm(nip_time_kdf, trial_starts_nip);
trial_starts_idx = find(trial_starts_idx);

trial_stops_nip = [trial_struct.TrialTS];
trial_stops_idx = project_utils.find_closest_idxs_frm(nip_time_kdf, trial_stops_nip);
trial_stops_idx = find(trial_stops_idx);
end

