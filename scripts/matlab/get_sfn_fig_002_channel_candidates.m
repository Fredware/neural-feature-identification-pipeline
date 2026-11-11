function get_sfn_fig_002_channel_candidates(varargin)
% FIG2_CHANNEL_CANDIDATES  Rank candidate channels for Figure 2 by stated
% criteria, then draw a contact sheet so the final pick is made by eye.
%
% For each DOF: take the 12 channels whose NFR feature best tracks the DOF
% (|r| over the session), read the same window as Figure 2, and score them:
%   spk_mod   threshold crossings during movement (0..1.25 s) / at rest (-0.75..0 s)
%   snr       median crossing amplitude / noise (noise = MAD of 750 Hz high-pass)
%   emg_mod   10-300 Hz power, movement / rest   (high = EMG-like crosstalk)
%   max_uv    largest |raw| in the window       (> 1000 uV = artifact)
%   line      power 58-62 Hz / power 40-80 Hz (excl. 58-62) (high = line noise)
%   common    correlation of the channel's 10-300 Hz signal with the mean of all
%             192 channels (high = common-mode: reference/EMG/motion, not local units)
% Pass rule (stated in the caption): spk_mod >= 2, snr >= 4, max_uv < 1000,
% emg_mod < 3, line < 5, common < 0.5. Survivors are sorted by spk_mod.
% Filters are second-order sections (the 10-300 Hz band is numerically unstable
% as a single transfer function at 30 kHz).
%
% Run from the repo root (same node/memory as Figure 2):
%   matlab -nodisplay -r "addpath('scripts/matlab'); fig2_channel_candidates(); exit"
% Then:  fig2_signal_flow('channels', [a b c])

p = inputParser;
addParameter(p, 'sess_root', fullfile(getenv('HOME'), 'sfn_session36'));
addParameter(p, 'data_root', '/uufs/chpc.utah.edu/common/home/george-group1/data-usea');
addParameter(p, 'session_dir', 'P201601/20160831-102216');
addParameter(p, 'ns5_file', '20160831-102216-001.ns5');
addParameter(p, 'dofs', [12 10 1]);
addParameter(p, 'n_cand', 24);
addParameter(p, 'out', 'reports/stats_sfn2026/final');
parse(p, varargin{:}); a = p.Results;
FS = 30000; PRE = 0.75; POST = 1.75;
NAMES = containers.Map({1,2,3,4,5,6,10,12}, {'Thumb F/E','Index F/E','Middle F/E', ...
    'Ring F/E','Little F/E','Thumb Intrinsic','Wrist Pitch','Wrist Roll'});

kin = h5read(fullfile(a.sess_root, 'kinematics.h5'), '/kinematics');
nip = double(h5read(fullfile(a.sess_root, 'kinematics.h5'), '/nip_time')); nip = nip(:);
nfr = h5read(fullfile(a.sess_root, 'features', 'NFR.h5'), '/features');
T = numel(nip); dt = median(diff(nip)) / FS;

spath = fullfile(a.data_root, a.session_dir); ns5 = fullfile(spath, a.ns5_file);
[~, sname] = fileparts(spath); %#ok<ASGLU>
try
    off = project_utils.CalculateNIPOffset_bhm(regexprep(ns5, '.ns5', '.ns2'), ...
        fullfile(spath, ['RecStart_' sname '.mat']));
catch
    off = project_utils.CalculateNIPOffset_bhm(regexprep(ns5, '.ns5', '.ns2'), ...
        fullfile(spath, ['Kalman_SSStruct_' sname '.mat']));
end
[z, pp, k] = butter(4, 750/(FS/2), 'high');        [sosH, gH] = zp2sos(z, pp, k);
[z, pp, k] = butter(4, [10 300]/(FS/2), 'bandpass'); [sosL, gL] = zp2sos(z, pp, k);

if ~exist(a.out, 'dir'), mkdir(a.out); end
fid = fopen(fullfile(a.out, 'fig2_channel_candidates.csv'), 'w');
fprintf(fid, 'dof,channel,abs_r,spk_mod,snr,emg_mod,max_uv,line,common,passes\n');
set(groot, 'defaultAxesFontName', 'Arial', 'defaultTextFontName', 'Arial');

for s = 1:numel(a.dofs)
    dof = a.dofs(s);
    r = abs(corr(nfr, abs(kin(end - size(nfr, 1) + 1:end, dof)), 'rows', 'complete'));
    r(isnan(r)) = 0;
    [rs, order] = sort(r, 'descend'); cand = order(1:a.n_cand);
    onset = find(kin(:, dof) > 0.1, 1, 'first');
    pad = round(0.25 * FS);
    rng = [nip(max(1, onset - round(PRE/dt))) - pad, nip(min(T, onset + round(POST/dt))) + pad] + off;
    [hdr, raw] = unrl_utils.fastNSxRead2022('File', ns5, 'Range', rng);
    sf = (double(hdr.MaxAnlgVal(1)) - double(hdr.MinAnlgVal(1))) / ...
         (double(hdr.MaxDigVal(1)) - double(hdr.MinDigVal(1)));
    tx = ((0:size(raw, 2)-1)' + rng(1) - off - nip(onset)) / FS;
    keep = tx >= -PRE & tx <= POST; tx = tx(keep);
    rest = tx < 0; move = tx >= 0 & tx <= 1.25;

    allL = filtfilt(sosL, gL, double(raw(1:min(192, size(raw, 1)), :)') * sf);
    cmode = mean(allL(keep, :), 2); clear allL
    M = nan(a.n_cand, 7); W = cell(a.n_cand, 1); X = cell(a.n_cand, 1); H = X; TH = nan(a.n_cand, 1);
    for c = 1:a.n_cand
        ch = cand(c);
        x = double(raw(ch, :)') * sf;
        hp = filtfilt(sosH, gH, x); lb = filtfilt(sosL, gL, x);
        x = x(keep); hp = hp(keep); lb = lb(keep);
        noise = median(abs(hp)) / 0.6745; thr = -5 * noise;
        xc = find(hp(2:end) < thr & hp(1:end-1) >= thr) + 1;
        xc = xc(xc > 30 & xc < numel(hp) - 45);
        nr = sum(rest(xc)) / max(sum(rest), 1); nm = sum(move(xc)) / max(sum(move), 1);
        amp = arrayfun(@(i) min(hp(i:i+15)), xc);
        [pxx, f] = pwelch(x, 2^14, [], [], FS);
        lineP = mean(pxx(f >= 58 & f <= 62)) / mean(pxx((f >= 40 & f < 58) | (f > 62 & f <= 80)));
        nrc = sum(rest(xc)); nmc = sum(move(xc));          % crossing counts
        M(c, :) = [((nmc + 1) / sum(move)) / ((nrc + 1) / sum(rest)), ...
                   median(abs(amp)) / noise, ...
                   mean(lb(move).^2) / mean(lb(rest).^2), max(abs(x)), lineP, rs(c), ...
                   corr(lb, cmode)];
        if isempty(xc), M(c, 2) = 0; end
        W{c} = cell2mat(arrayfun(@(i) hp(i-30:i+45)', xc(1:min(end, 60)), 'UniformOutput', false));
        X{c} = x; H{c} = hp; TH(c) = thr;
    end
    pass = M(:,1) >= 2 & M(:,2) >= 4 & M(:,4) < 1000 & M(:,3) < 3 & M(:,5) < 5 & M(:,7) < 0.5;
    for c = 1:a.n_cand
        fprintf(fid, '%d,%d,%.3f,%.2f,%.2f,%.2f,%.0f,%.2f,%.2f,%d\n', dof, cand(c), M(c,6), ...
            M(c,1), M(c,2), M(c,3), M(c,4), M(c,5), M(c,7), pass(c));
    end
    [~, srt] = sortrows([~pass, -M(:,1)]); show = srt(1:min(6, end));
    fprintf('%s: passing channels %s\n', NAMES(dof), mat2str(cand(pass)'));

    % contact sheet: one row per candidate (raw | 750 Hz HP + threshold | crossings overlay)
    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [0 0 14 2.1*numel(show)], 'Visible', 'off');
    tl = tiledlayout(fig, numel(show), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:numel(show)
        c = show(k);
        nexttile(tl); plot(tx, X{c}, 'k', 'LineWidth', 0.3); xlim([-PRE POST]); box off
        ylabel(sprintf('ch %d%s', cand(c), repmat(' *', 1, pass(c))));
        title(sprintf('|r| %.2f  spk %.1fx  SNR %.1f  EMG %.1fx  max %.0f \\muV  line %.1f  common %.2f', ...
            M(c,6), M(c,1), M(c,2), M(c,3), M(c,4), M(c,5), M(c,7)), 'FontWeight', 'normal', 'FontSize', 9);
        nexttile(tl); plot(tx, H{c}, 'Color', [.5 .5 .5], 'LineWidth', 0.3); hold on
        yline(TH(c), '--', 'Color', [230 159 0]/255); xlim([-PRE POST]); box off
        nexttile(tl);
        if ~isempty(W{c}), plot((-30:45)/FS*1e3, W{c}', 'Color', [0 0 0 0.15]); end
        xlabel('ms'); box off
    end
    title(tl, sprintf('%s candidates (* = passes all criteria)', NAMES(dof)));
    exportgraphics(fig, fullfile(a.out, sprintf('fig2_candidates_dof%d.png', dof)), 'Resolution', 120);
    close(fig); clear raw
end
fclose(fid);
fprintf('Wrote %s and fig2_candidates_dof*.png\n', fullfile(a.out, 'fig2_channel_candidates.csv'));
end

