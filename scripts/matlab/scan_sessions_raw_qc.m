function scan_sessions_raw_qc(varargin)
% SCAN_SESSIONS_RAW_QC  Per-session raw-signal quality scan to pick a session
% whose USEA channels show local, movement-modulated spiking (for Fig 2/3) and
% to seed the channel-provenance work in issue #14.
%
% For each session: read the first `win_s` seconds of the training recording
% (all channels), and for USEA channels 1:192 compute
%   noise      MAD noise of the 750 Hz high-pass (uV)
%   rate_rest / rate_move   threshold crossings/s (-5 x noise, 1 ms refractory)
%                           at rest (all kinematics 0) vs during movement (any |kin| > 0.1)
%   amp        median crossing trough / noise
%   env_common correlation of the channel's spike-band envelope (|HP|, 30 ms)
%              with the median envelope across all 192 channels (shared = not local)
%   line       60 Hz power vs 40-80 Hz neighbours
%   max_uv     largest |raw| (saturation/artifact)
% "Good" channel: rate_move >= 5/s, rate_move >= 2 x rate_rest, amp >= 6,
%   env_common < 0.5, max_uv < 2000, line < 20.
% Rows are appended per session, so a preempted job resumes where it stopped.
%
%   matlab -batch "addpath('scripts/matlab'); scan_sessions_raw_qc()"

p = inputParser;
addParameter(p, 'list', 'reports/stats_sfn2026/session_scan_list.csv');
addParameter(p, 'out', 'reports/stats_sfn2026/session_scan_qc.csv');
addParameter(p, 'per_channel_dir', 'reports/stats_sfn2026/session_scan_channels');
addParameter(p, 'data_root', '/uufs/chpc.utah.edu/common/home/george-group1/data-usea');
addParameter(p, 'win_s', 90);
parse(p, varargin{:}); a = p.Results;
FS = 30000; NCH = 192;

L = readtable(a.list, 'TextType', 'string', 'Delimiter', ',');
done = [];
if isfile(a.out), done = readtable(a.out).job_id; end
if ~exist(a.per_channel_dir, 'dir'), mkdir(a.per_channel_dir); end
if ~isfile(a.out)
    fid = fopen(a.out, 'w');
    fprintf(fid, ['job_id,participant,session_dir,n_good,n_modulated,median_noise_uv,' ...
        'frac_common,frac_line,frac_saturated,rmse_NFR,rmse_SBP,rmse_DWT,rmse_MAV,status\n']);
    fclose(fid);
end
[z, pp, k] = butter(4, 750/(FS/2), 'high'); [sosH, gH] = zp2sos(z, pp, k);

for i = 1:height(L)
    J = L(i, :);
    if ismember(J.job_id, done), continue; end
    status = "ok"; row = nan(1, 7);
    try
        spath = char(fullfile(a.data_root, J.session_dir));
        [~, sname] = fileparts(spath);
        f5 = dir(fullfile(spath, '*.ns5'));
        [~, big] = max([f5.bytes]); ns5 = char(fullfile(spath, f5(big).name));
        try
            off = project_utils.CalculateNIPOffset_bhm(regexprep(ns5, '.ns5', '.ns2'), ...
                fullfile(spath, ['RecStart_' char(sname) '.mat']));
        catch
            off = project_utils.CalculateNIPOffset_bhm(regexprep(ns5, '.ns5', '.ns2'), ...
                fullfile(spath, ['Kalman_SSStruct_' char(sname) '.mat']));
        end
        [kin, ~, ~, ~, knip] = unrl_utils.readKDF_jag(char(fullfile(spath, J.training_file)));
        knip = double(knip(:));
        if size(kin, 1) ~= numel(knip), kin = kin'; end
        t_end = min(knip(end), knip(1) + a.win_s * FS);
        [hdr, raw] = unrl_utils.fastNSxRead2022('File', ns5, 'Range', [knip(1), t_end] + off);
        sf = (double(hdr.MaxAnlgVal(1)) - double(hdr.MinAnlgVal(1))) / ...
             (double(hdr.MaxDigVal(1)) - double(hdr.MinDigVal(1)));
        n = size(raw, 2); ts = knip(1) + (0:n-1)';
        fi = discretize(ts, [knip; inf]); fi(isnan(fi)) = numel(knip);
        moving = any(abs(kin(fi, :)) > 0.1, 2);
        resting = all(kin(fi, :) == 0, 2);
        nch = min(NCH, size(raw, 1));
        C = nan(nch, 8); ENV = zeros(floor(n / 300), nch, 'single');
        for ch = 1:nch
            x = double(raw(ch, :)') * sf;
            hp = filtfilt(sosH, gH, x);
            noise = median(abs(hp)) / 0.6745; thr = -5 * noise;
            xc = find(hp(2:end) < thr & hp(1:end-1) >= thr) + 1;
            if ~isempty(xc), xc = xc([true; diff(xc) > 30]); end
            xc = xc(xc < n - 16);
            amp = arrayfun(@(j) min(hp(j:j+15)), xc);
            e = movmean(abs(hp), 900); ENV(:, ch) = e(150:300:300*size(ENV, 1));
            seg = x(1:min(n, 10 * FS));
            [pxx, f] = pwelch(seg, 2^14, [], [], FS);
            lineP = mean(pxx(f >= 58 & f <= 62)) / mean(pxx((f >= 40 & f < 58) | (f > 62 & f <= 80)));
            C(ch, :) = [noise, sum(resting(xc)) / max(sum(resting) / FS, 1), ...
                sum(moving(xc)) / max(sum(moving) / FS, 1), ...
                median(abs(amp)) / max(noise, eps), nan, lineP, max(abs(x)), ch];
        end
        med = median(ENV, 2);
        for ch = 1:nch, C(ch, 5) = corr(double(ENV(:, ch)), double(med)); end
        good = C(:,3) >= 5 & C(:,3) >= 2 * C(:,2) & C(:,4) >= 6 & C(:,5) < 0.5 & C(:,7) < 2000 & C(:,6) < 20;
        modulated = C(:,3) >= 5 & C(:,3) >= 2 * C(:,2);
        row = [sum(good), sum(modulated), median(C(:,1)), mean(C(:,5) >= 0.5), ...
               mean(C(:,6) >= 20), mean(C(:,7) >= 8000), nan];
        T = array2table(C, 'VariableNames', {'noise_uv','rate_rest','rate_move','amp', ...
            'env_common','line','max_uv','channel'});
        T.good = good;
        writetable(T, fullfile(a.per_channel_dir, sprintf('job_%03d.csv', J.job_id)));
        clear raw ENV
    catch ME
        status = "error: " + replace(string(ME.message), [",", newline], [";", " "]);
    end
    fid = fopen(a.out, 'a');
    fprintf(fid, '%d,%s,%s,%g,%g,%.2f,%.3f,%.3f,%.3f,%g,%g,%g,%g,%s\n', J.job_id, J.participant, ...
        J.session_dir, row(1), row(2), row(3), row(4), row(5), row(6), ...
        J.rmse_NFR, J.rmse_SBP, J.rmse_DWT, J.rmse_MAV, status);
    fclose(fid);
    fprintf('[%d/%d] job %d: good=%g modulated=%g (%s)\n', i, height(L), J.job_id, row(1), row(2), status);
end
end

