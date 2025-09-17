function closest_idxs = find_closest_idxs_frm(nip_time_vector, target_nip_stamps)
closest_idxs = [];
for ii = 1:length(target_nip_stamps)
    [~, closest_idxs(ii)] = min(abs(nip_time_vector - target_nip_stamps(ii)));
end
    closest_idxs = ismember(nip_time_vector, nip_time_vector(closest_idxs));
end
