function plot_sfn_fig_002_signal_flow(varargin)
% FIG2_SIGNAL_FLOW  Raw USEA signal -> intermediate stage -> decoder feature,
% for the three best-decoded DOFs of one session, stitched side by side.
%
% Rows (shared time axis within each segment):
%   1 Target kinematics of the moving DOF
%   2 Raw broadband voltage (30 kHz)
%   3 NFR  intermediate: 750 Hz high-pass + threshold + spike ticks. Threshold rule
%          from makeNeuralFeatures_NS5: std(high-passed TRAINING data) * -5, here
%          estimated from the first 20 s of the training recording.
%   4 NFR  feature (from features/NFR.h5 = exactly what the decoder saw)
%   5 SBP  intermediate: 300-1000 Hz band-pass (config.yaml params; illustrative)
%   6 SBP  feature (features/SBP-RAW.h5)
%   7 MAV  intermediate: 750 Hz high-pass, rectified
%   8 MAV  feature (features/MAV.h5, baseline-subtracted by the pipeline)
%   9 DWT  intermediate: db4 detail signals at three levels (2: 3.75-7.5 kHz,
%          5: 469-938 Hz, 8: 59-117 Hz), reconstructed with wrcoef
%  10 DWT  feature: thresholded detail energy per db4 level (features/DWT-DB4.h5)
%
% One channel per segment: the channel whose NFR feature (threshold crossings,
% i.e. spiking units) correlates best with that DOF's kinematics over the
% session, unless 'channels' is given. Stated in the panel title.
% Raw/intermediate traces are min/max-decimated for plotting (shape preserved,
% vector files stay small); numbers are unaffected.
%
% Run from the repo root on a compute node (reads ~3 short NS5 slices + baseline):
%   module load matlab/R2024b
%   matlab -nodisplay -r "addpath('scripts/matlab'); fig2_signal_flow(); exit"
% Options (name/value): 'dofs',[12 10 1], 'config','8DOF', 'pre',0.75, 'post',1.75,
%   'out','reports/stats_sfn2026/final'

p = inputParser;
addParameter(p, 'sess_root', fullfile(getenv('HOME'), 'sfn_session36'));
addParameter(p, 'data_root', '/uufs/chpc.utah.edu/common/home/george-group1/data-usea');
addParameter(p, 'session_dir', 'P201601/20160831-102216');
addParameter(p, 'ns5_file', '20160831-102216-001.ns5');
addParameter(p, 'baseline_kdf', 'BaselineData_20160831-102216_103956.kdf');
addParameter(p, 'config', '8DOF');
addParameter(p, 'dofs', []);
addParameter(p, 'channels', []);
addParameter(p, 'pre', 0.75);
addParameter(p, 'post', 1.75);
addParameter(p, 'out', 'reports/stats_sfn2026/final');
addParameter(p, 'label', 'Session 36 (S6, day 96)');
parse(p, varargin{:}); a = p.Results;

FS = 30000;
NFR_THRESH_SD = -5;                  % config.yaml nfr.spike_threshold_std
SBP_BAND = [300 1000]; SBP_ORDER = 2;% config.yaml sbp
FEATS = {'NFR', 'SBP-RAW', 'DWT-DB4', 'MAV'};
COL = struct('NFR', [230 159 0]/255, 'SBP', [86 180 233]/255, ...
             'DWT', [0 158 115]/255, 'MAV', [204 121 167]/255);
NAMES = containers.Map({1,2,3,4,5,6,10,12}, {'Thumb F/E','Index F/E','Middle F/E', ...
    'Ring F/E','Little F/E','Thumb Intrinsic','Wrist Pitch','Wrist Roll'});

% ---- kinematics + features (MATLAB h5read: time is rows) -----------------
kin = h5read(fullfile(a.sess_root, 'kinematics.h5'), '/kinematics');   % T x 12
nip = double(h5read(fullfile(a.sess_root, 'kinematics.h5'), '/nip_time')); nip = nip(:);
T = numel(nip);
F = struct();
for k = 1:numel(FEATS)
    x = h5read(fullfile(a.sess_root, 'features', [FEATS{k} '.h5']), '/features');
    if size(x, 1) ~= T && size(x, 2) >= T - 5, x = x'; end
    F.(matlab.lang.makeValidName(FEATS{k})) = x;          % rows = frames
end
% NFR has n_frames-1 rows: it is aligned to nip(2:end) (pipeline trims the first sample)
nfr_nip = nip(end - size(F.NFR, 1) + 1:end);

% ---- choose DOFs: lowest mean intended RMSE across features -----------------
if isempty(a.dofs)
    cand = []; score = [];
    for k = 1:numel(FEATS)
        m = jsondecode(fileread(fullfile(a.sess_root, 'current_eval', ...
            sprintf('%s_%s_metrics.json', a.config, FEATS{k}))));
        fn = fieldnames(m.aggregate_per_dof);
        for f = 1:numel(fn)
            d = str2double(regexprep(fn{f}, '\D', ''));
            r = m.aggregate_per_dof.(fn{f}).intended.rmse_mean;
            i = find(cand == d, 1);
            if isempty(i), cand(end+1) = d; score(end+1) = r; %#ok<AGROW>
            else, score(i) = score(i) + r; end
        end
    end
    [~, order] = sort(score); a.dofs = cand(order(1:3));
    fprintf('Best-decoded DOFs (%s, mean intended RMSE over features): %s\n', ...
        a.config, mat2str(a.dofs));
end

% ---- NIP offset (same logic as extract_features.m) ---------------------------
spath = fullfile(a.data_root, a.session_dir);
ns5 = fullfile(spath, a.ns5_file);
ns2 = regexprep(ns5, '.ns5', '.ns2');
[~, sname] = fileparts(spath);
try
    off = project_utils.CalculateNIPOffset_bhm(ns2, fullfile(spath, ['RecStart_' sname '.mat']));
catch
    off = project_utils.CalculateNIPOffset_bhm(ns2, fullfile(spath, ['Kalman_SSStruct_' sname '.mat']));
end

% ---- per-DOF segment, channel, raw slice ----------------------------------------
dt = median(diff(nip)) / FS;
sbp = F.SBP_RAW;
seg = struct([]);
for s = 1:numel(a.dofs)
    dof = a.dofs(s);
    onset = find(kin(:, dof) > 0.1, 1, 'first');
    fr = max(1, onset - round(a.pre/dt)) : min(T, onset + round(a.post/dt));
    r = abs(corr(F.NFR, abs(kin(end - size(F.NFR, 1) + 1:end, dof)), 'rows', 'complete'));
    if isempty(a.channels), [~, ch] = max(r); else, ch = a.channels(s); end
    pad = round(0.25 * FS);                                   % filter edge padding
    rng = [nip(fr(1)) - pad, nip(fr(end)) + pad] + off;
    [hdr, raw] = unrl_utils.fastNSxRead2022('File', ns5, 'Range', rng);
    sf = (double(hdr.MaxAnlgVal(1)) - double(hdr.MinAnlgVal(1))) / ...
         (double(hdr.MaxDigVal(1)) - double(hdr.MinDigVal(1)));
    x = double(raw(ch, :)') * sf;
    tx = ((0:numel(x)-1)' + rng(1) - off - nip(onset)) / FS;  % s re: movement onset
    seg(s).dof = dof; seg(s).ch = ch; seg(s).fr = fr; seg(s).onset = onset;
    seg(s).x = x; seg(s).tx = tx; seg(s).corr = r(ch);
    fprintf('Segment %d: %s, channel %d (|r| NFR vs kinematics = %.2f)\n', ...
        s, NAMES(dof), ch, r(ch));
end

% ---- NFR threshold: std(high-passed training data) * -5 (makeNeuralFeatures_NS5)
[thdr, traw] = unrl_utils.fastNSxRead2022('File', ns5, 'Range', [nip(1), nip(1) + 20*FS] + off);
tsf = (double(thdr.MaxAnlgVal(1)) - double(thdr.MinAnlgVal(1))) / ...
      (double(thdr.MaxDigVal(1)) - double(thdr.MinDigVal(1)));
[Bh, Ah] = butter(4, 750/(FS/2), 'high');
[Bb, Ab] = butter(SBP_ORDER, SBP_BAND/(FS/2), 'bandpass');
DWT_SHOW = [2 5 8];
for s = 1:numel(seg)
    seg(s).thr = std(filtfilt(Bh, Ah, double(traw(seg(s).ch, :)') * tsf)) * NFR_THRESH_SD;
    seg(s).hp  = filtfilt(Bh, Ah, seg(s).x);
    seg(s).bp  = filtfilt(Bb, Ab, seg(s).x);
    [c, l] = wavedec(seg(s).x, 10, 'db4');
    seg(s).dw = zeros(numel(seg(s).x), numel(DWT_SHOW));
    for k = 1:numel(DWT_SHOW), seg(s).dw(:, k) = wrcoef('d', c, l, 'db4', DWT_SHOW(k)); end
    keep = seg(s).tx >= -a.pre & seg(s).tx <= a.post;
    for fld = {'x', 'tx', 'hp', 'bp'}, seg(s).(fld{1}) = seg(s).(fld{1})(keep); end
    seg(s).dw = seg(s).dw(keep, :);
    xc = find(seg(s).hp(2:end) < seg(s).thr & seg(s).hp(1:end-1) >= seg(s).thr);
    seg(s).spk = seg(s).tx(xc + 1);
end
clear traw

% ---- plot ---------------------------------------------------------------------
set(groot, 'defaultAxesFontName', 'Arial', 'defaultTextFontName', 'Arial');
nS = numel(seg); nR = 10;
ZERO_GREY = [0.80 0.80 0.80]; dwt_ax = gobjects(0); dwt_lims = zeros(0, 2);
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [0 0 6*nS 13], 'Visible', 'off');
tl = tiledlayout(fig, nR, nS, 'TileSpacing', 'compact', 'Padding', 'compact', ...
    'TileIndexing', 'columnmajor');
rowlab = {'Target', 'Raw (\muV)', 'NFR: 750 Hz HP', 'NFR (spikes/frame)', ...
    'SBP: 300-1000 Hz BP', 'SBP', 'MAV: 750 Hz HP, |x|', 'MAV', 'DWT levels 2/5/8', 'DWT band'};
for s = 1:nS
    S = seg(s); fr = S.fr;
    tf = (nip(fr) - nip(S.onset)) / FS;                     % frame times
    ax = gobjects(nR, 1);
    ax(1) = nexttile(tl); plot(tf, kin(fr, S.dof), 'Color', [.45 .45 .45], 'LineWidth', 2);
    title(sprintf('%s  (channel %d)', NAMES(S.dof), S.ch), 'FontWeight', 'bold');
    ax(2) = nexttile(tl); envplot(S.tx, S.x, [.2 .2 .2]);
    ax(3) = nexttile(tl); envplot(S.tx, S.hp, [.6 .6 .6]); hold on
        yline(S.thr, '--', 'Color', COL.NFR, 'LineWidth', 1);
        plot(S.spk, S.thr * 1.6 * ones(size(S.spk)), '|', 'Color', COL.NFR, 'MarkerSize', 6);
    ax(4) = nexttile(tl);
        nf = ismember(nfr_nip, nip(fr));
        plot((nfr_nip(nf) - nip(S.onset)) / FS, F.NFR(nf, S.ch), 'Color', COL.NFR, 'LineWidth', 1.8);
    ax(5) = nexttile(tl); envplot(S.tx, S.bp, [.6 .6 .6]);
    ax(6) = nexttile(tl); plot(tf, F.SBP_RAW(fr, S.ch), 'Color', COL.SBP, 'LineWidth', 1.8);
    ax(7) = nexttile(tl); envplot(S.tx, abs(S.hp), [.6 .6 .6]);
    ax(8) = nexttile(tl); plot(tf, F.MAV(fr, S.ch), 'Color', COL.MAV, 'LineWidth', 1.8);
    ax(9) = nexttile(tl); hold on
        sc = max(abs(S.dw), [], 1); sc(sc == 0) = 1;
        for k = 1:numel(DWT_SHOW)
            envplot(S.tx, S.dw(:, k) / sc(k) + 2.2 * (numel(DWT_SHOW) - k), [.6 .6 .6]);
        end
        yticks(2.2 * (0:numel(DWT_SHOW)-1)); yticklabels(compose('L%d', fliplr(DWT_SHOW)));
    ax(10) = nexttile(tl);
        E = F.DWT_DB4(fr, (S.ch - 1) * 10 + (1:10))';         % channel-major -> 10 x frames
        L = log10(E); L(E == 0) = NaN;                         % zero = hard-thresholded
        h = imagesc(tf, 1:10, L, 'AlphaData', ~isnan(L));
        set(ax(10), 'YDir', 'reverse', 'Color', ZERO_GREY);    % level 1 (highest band) on top
        colormap(ax(10), interp1([0 1], [0.86 0.95 0.91; COL.DWT * 0.75], linspace(0, 1, 256)));
        dwt_ax(end+1) = ax(10); dwt_lims(end+1, :) = [min(L(:), [], 'omitnan'), max(L(:), [], 'omitnan')]; %#ok<AGROW>
        yticks([1 4 7 10]);
        yticklabels({'7.5-15 kHz', '0.94-1.9 kHz', '117-234 Hz', '15-29 Hz'});
        hold on                                                % faint level boundaries: binned data
        for yb = 1.5:1:9.5, yline(yb, '-', 'Color', [1 1 1], 'LineWidth', 0.5); end
        xlabel('Time from movement onset (s)');
    for r = 1:nR
        xlim(ax(r), [-a.pre a.post]); box(ax(r), 'off');
        if r < nR, ax(r).XTickLabel = []; end
        if s == 1, ylabel(ax(r), rowlab{r}); end
        xline(ax(r), 0, ':', 'Color', [.5 .5 .5]);
    end
    linkaxes(ax, 'x');
end
% shared DWT colour scale + one colorbar; grey = below threshold (explicit category)
cl = [min(dwt_lims(:, 1)), max(dwt_lims(:, 2))];
for k = 1:numel(dwt_ax), clim(dwt_ax(k), cl); end
cb = colorbar(dwt_ax(end)); cb.Layout.Tile = 'east';
cb.Label.String = {'DWT energy (log_{10})', 'grey = below baseline threshold (set to 0)'};
title(tl, sprintf('%s, %s decoder''s best-decoded DOFs', a.label, a.config), ...
    'FontName', 'Arial');

if ~exist(a.out, 'dir'), mkdir(a.out); end
stem = fullfile(a.out, 'fig2_signal_flow');
exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector');
exportgraphics(fig, [stem '.png'], 'Resolution', 150);
try
    exportgraphics(fig, [stem '.svg'], 'ContentType', 'vector');    % R2024b+ supports SVG
catch
    print(fig, [stem '.svg'], '-dsvg', '-painters');
end
save([stem '_data.mat'], 'seg', 'a');
fprintf('Saved %s.{pdf,svg,png} and %s_data.mat\n', stem, stem);
end

function envplot(t, y, col)
% Min/max envelope decimation to ~3000 bins: keeps spike shapes visible while
% keeping vector output small. Plotting only; the data are unchanged.
nb = 3000; n = numel(y);
if n <= 2 * nb, plot(t, y, 'Color', col, 'LineWidth', 0.5); return; end
e = round(linspace(0, n, nb + 1));
tt = zeros(2 * nb, 1); yy = tt;
for i = 1:nb
    idx = e(i) + 1:e(i + 1);
    [mn, imn] = min(y(idx)); [mx, imx] = max(y(idx));
    if imn < imx, yy(2*i-1:2*i) = [mn mx]; else, yy(2*i-1:2*i) = [mx mn]; end
    tt(2*i-1:2*i) = t(idx(1)) + [0 1] * (t(idx(end)) - t(idx(1)));
end
plot(tt, yy, 'Color', col, 'LineWidth', 0.5);
end

