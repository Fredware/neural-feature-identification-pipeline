function NIPOffset = CalculateNIPOffset_bhm(NS2File, RecStartFile)
% caluculates NIP offset for corrupted nsx data taken after 5/19/2016
% works on data recorded using FeedbackDecode
% inputs: NS2 (full file path string, *.ns2), RecStart.mat or
% SSStruct.mat (fullfile path string)
% outputs: NIPOffset : offset in NIP samples.  This number is the number of NIP samples the NSx files are leading the KDF
% file
% example
% RecStartFile = 'E:\Data\P201601\20160608-124342\RecStart_20160608-124342.mat';  NS2File = 'E:\Data\P201601\20160608-124342\20160608-124342-0010002.ns2';

% init
NIPOffset = 0;

load(RecStartFile); % load RecStart or SSstruct.mat
[hdr, D] = unrl_utils.fastNSxRead2022('File', NS2File, 'Range', [0, 1000]);

if exist('RecStart', 'var')
    NIPOffset = double(RecStart) - double(hdr.NIPStart) ;
elseif exist('SS', 'var')
    if isfield(SS, 'RecStart')
        NIPOffset = double(SS.RecStart) - double(hdr.NIPStart);
    end
end


%% old stuff for synching based on first event in NEV file and events file
% NEV = openNEV(NEVFile);
% DData = regexp(char(NEV.Data.SerialDigitalIO.UnparsedData),'*','split'); DData = DData(2:end);
% DDataTS = NEV.Data.SerialDigitalIO.TimeStamp(regexp(char(NEV.Data.SerialDigitalIO.UnparsedData),'*'));
% DDataTS = DDataTS(cellfun(@isempty,regexp(DData,'^\d+')))';
% DData = DData(cellfun(@isempty,regexp(DData,'^\d+')))';
% IDIdx = find(~cellfun(@isempty,regexp(DData,'ID')));
% 
% 
% fid = fopen(EventsFile);
% EParamsCell = textscan(fid,'%s','delimiter','\n\r'); EParamsCell = EParamsCell{:};
% fclose(fid);
% 
% IDCellIdx = ~cellfun(@isempty,regexp(EParamsCell,'EventID'));