function [Header,varargout] = fastNSxRead2022(varargin)

% Reads NSx files over a specified range and writes back to an NSx file if
% "FileOut" is provided.
%
% Example: [Header, D] = fastNSxRead('File', NS5file);
% Example: [Header, D] = fastNSxRead('File', NS5file, 'Range', SampleRange);
%
% SampleRange = [startSample, endSample]; 
% Samples are with respect to sample rate of file
%
% Version Date: 20170428
% Author: Tyler Davis
%
% Changelog:
% 20160614-td: Added NIPStart to header
% 20170414-td: Added code to load data files that are automatically split due to system overload in Blackrock
% 20170428-td: Added ability to save a portion of file back to nsx format
% 20220803-td: Changed *char to uint8=>char in fread calls - seems to be faster for some reason


% Parsing input
p = inputParser;
defaultRange = [];
defaultFile = '';
defaultMode = 'Read'; %Write
addParameter(p,'Range',defaultRange,@(x)size(x,2)==2);
addParameter(p,'File',defaultFile,@(x)exist(x,'file'));
addParameter(p,'Mode',defaultMode,@(x)any(validatestring(x,{'Read','Write'})));

parse(p,varargin{:});

% Defining variables
Range = p.Results.Range;
FNameIn = char(p.Results.File);
Mode = p.Results.Mode;

if regexpi(computer,'win')
    SystemType = 'Windows';
else
    SystemType = 'Other';
end

if isempty(FNameIn)
    [PathIn,NameIn,ExtIn] = lastPath('\*.ns?','Choose nsx file...');
    FNameIn = fullfile(PathIn,[NameIn,ExtIn]);
end

if strcmp(Mode,'Write') && ~isempty(Range)
    FNameOut = regexprep(FNameIn,'\.ns\d{1}$',['_',regexprep(num2str(Range),'\s+','-'),'.',FNameIn(end-2:end)]);
else
    FNameOut = '';
end

switch FNameIn(end-2:end)
    case {'ns5','ns6'}
        Header.Fs = 30000;
    case 'ns4'
        Header.Fs = 10000;
    case 'ns3'
        Header.Fs = 2000;
    case 'ns2'
        Header.Fs = 1000;
    otherwise
        disp('Choose an NSx file')
        return
end

% Getting fileid
FID = fopen(FNameIn, 'r', 'l');
Header.FileID = fread(FID, [1,8], 'uint8=>char');
if ~isempty(FNameOut)
    FIDOut = fopen(FNameOut, 'w', 'l');
    fwrite(FIDOut, Header.FileID, '*char');
end

% Reading NSx file header
if strcmp(Header.FileID,'NEURALSG') %v2.1    
    Header.FsStr        = fread(FID, [1,16],  'uint8=>char');
    Header.Period       = fread(FID, [1,1],   '*uint32');
    Header.ChannelCount = fread(FID, [1,1],   'uint32=>double');
    for k = 1:Header.ChannelCount
        Header.ChannelID(k,:) = fread(FID, [1,1],  '*uint32');
    end
    if ~isempty(FNameOut)
        fwrite(FIDOut, Header.FsStr,  '*char');
        fwrite(FIDOut, Header.Period,   '*uint32');
        fwrite(FIDOut, Header.ChannelCount,   '*uint32');
        for k = 1:Header.ChannelCount
            fwrite(FIDOut, Header.ChannelID(k,:),  '*uint32');
        end
    end    
else %v2.2 or v2.3 (NEURALCD)
    Header.FileSpec     = fread(FID, [1,2],   '*uchar');
    Header.HeaderBytes  = fread(FID, [1,1],   '*uint32');
    Header.FsStr        = fread(FID, [1,16],  'uint8=>char');
    Header.Comment      = fread(FID, [1,252], 'uint8=>char');
    Header.NIPStart     = fread(FID, [1,1],   '*uint32');
    Header.Period       = fread(FID, [1,1],   '*uint32');
    Header.Resolution   = fread(FID, [1,1],   '*uint32');
    Header.TimeOrigin   = fread(FID, [1,8],   '*uint16');
    Header.ChannelCount = fread(FID, [1,1],   'uint32=>double');    
    for k = 1:Header.ChannelCount
        Header.Type(k,:)           = fread(FID, [1,2],  'uint8=>char');
        Header.ChannelID(k,:)      = fread(FID, [1,1],  '*uint16');
        Header.ChannelLabel(k,:)   = fread(FID, [1,16], 'uint8=>char');
        Header.PhysConnector(k,:)  = fread(FID, [1,1],  '*uint8');
        Header.ConnectorPin(k,:)   = fread(FID, [1,1],  '*uint8');
        Header.MinDigVal(k,:)      = fread(FID, [1,1],  '*int16');
        Header.MaxDigVal(k,:)      = fread(FID, [1,1],  '*int16');
        Header.MinAnlgVal(k,:)     = fread(FID, [1,1],  '*int16');
        Header.MaxAnlgVal(k,:)     = fread(FID, [1,1],  '*int16');
        Header.Units(k,:)          = fread(FID, [1,16], 'uint8=>char');
        Header.HighFreqCorner(k,:) = fread(FID, [1,1],  '*uint32');
        Header.HighFreqOrder(k,:)  = fread(FID, [1,1],  '*uint32');
        Header.HighFiltType(k,:)   = fread(FID, [1,1],  '*uint16');
        Header.LowFreqCorner(k,:)  = fread(FID, [1,1],  '*uint32');
        Header.LowFreqOrder(k,:)   = fread(FID, [1,1],  '*uint32');
        Header.LowFiltType(k,:)    = fread(FID, [1,1],  '*uint16');
    end
    if ~isempty(FNameOut)
        fwrite(FIDOut, Header.FileSpec,   '*uchar');
        fwrite(FIDOut, Header.HeaderBytes,   '*uint32');
        fwrite(FIDOut, Header.FsStr,  '*char');
        fwrite(FIDOut, Header.Comment, '*char');
        fwrite(FIDOut, Header.NIPStart,   '*uint32');
        fwrite(FIDOut, Header.Period,   '*uint32');
        fwrite(FIDOut, Header.Resolution,   '*uint32');
        fwrite(FIDOut, Header.TimeOrigin,   '*uint16');
        fwrite(FIDOut, Header.ChannelCount,   '*uint32');
        for k = 1:Header.ChannelCount
            fwrite(FIDOut, Header.Type(k,:),  '*char');
            fwrite(FIDOut, Header.ChannelID(k,:),  '*uint16');
            fwrite(FIDOut, Header.ChannelLabel(k,:), '*char');
            fwrite(FIDOut, Header.PhysConnector(k,:),  '*uint8');
            fwrite(FIDOut, Header.ConnectorPin(k,:),  '*uint8');
            fwrite(FIDOut, Header.MinDigVal(k,:),  '*int16');
            fwrite(FIDOut, Header.MaxDigVal(k,:),  '*int16');
            fwrite(FIDOut, Header.MinAnlgVal(k,:),  '*int16');
            fwrite(FIDOut, Header.MaxAnlgVal(k,:),  '*int16');
            fwrite(FIDOut, Header.Units(k,:), '*char');
            fwrite(FIDOut, Header.HighFreqCorner(k,:),  '*uint32');
            fwrite(FIDOut, Header.HighFreqOrder(k,:),  '*uint32');
            fwrite(FIDOut, Header.HighFiltType(k,:),  '*uint16');
            fwrite(FIDOut, Header.LowFreqCorner(k,:),  '*uint32');
            fwrite(FIDOut, Header.LowFreqOrder(k,:),  '*uint32');
            fwrite(FIDOut, Header.LowFiltType(k,:),  '*uint16');
        end
    end
end

BegOfDataHeader = ftell(FID);
fseek(FID, 0, 'eof');
EndOfFile = ftell(FID);
fseek(FID, BegOfDataHeader, 'bof');

% Checking for multiple data headers (v2.2 and v2.3 only) and calculating
% channel length
if strcmp(Header.FileID,'NEURALSG')
    Header.DataBytes = EndOfFile - BegOfDataHeader;
    Header.ChannelSamples = Header.DataBytes/Header.ChannelCount/2;
else
    k = 1;
    while ftell(FID)~=EndOfFile
        DataHeader(k,1) = fread(FID, [1,1], '*uint8'); %This value should always be 1
        if DataHeader(k,1)~=1
            disp('Error reading data headers!')
            return
        end
        DataTimestamp(k,1) = fread(FID, [1,1], '*uint32');
        ChannelSamples(k,1) = fread(FID, [1,1], 'uint32=>double');
        BegOfData(k,1) = ftell(FID); %Location of data after data header
        if ~ChannelSamples(k,1) %Stop if length data is zero
            DataBytes(k,1) = EndOfFile - BegOfData(k,1);
            ChannelSamples(k,1) = DataBytes(k,1)/Header.ChannelCount/2;
            break;
        end
        if ChannelSamples(k,1)*Header.ChannelCount*2>(EndOfFile-BegOfData(k,1)) %ChannelSamples(k,1)*Header.ChannelCount*2 is longer than length of file (i.e. ran out of disk space)
            fseek(FID,0,'eof');        
            DataBytes(k,1) = ftell(FID) - BegOfData(k,1);
            ChannelSamples(k,1) = floor(DataBytes(k,1)/(Header.ChannelCount*2));
            DataBytes(k,1) = ChannelSamples(k,1)*Header.ChannelCount*2;
            break;
        else
            fseek(FID,ChannelSamples(k,1)*Header.ChannelCount*2,'cof');
            DataBytes(k,1) = ftell(FID) - BegOfData(k,1);
        end
        k = k+1;
    end
    
    % Writing data header for file out
    if ~isempty(FNameOut)
        fwrite(FIDOut, DataHeader(1,1), '*uint8');
        fwrite(FIDOut, DataTimestamp(1,1), '*uint32');
        fwrite(FIDOut, diff(Range)+1, '*uint32'); %samples per channel
    end    
    
    % Check if pauses exist in data
    if (length(DataHeader)==2 && ChannelSamples(1)~=1) || length(DataHeader)>2
        disp('Pauses exist in this data set! Will only use data before the pause!')
        DataHeader(2:end) = [];
        DataTimestamp(2:end) = [];
        ChannelSamples(2:end) = [];
        BegOfData(2:end) = [];
        DataBytes(2:end) = [];
    end
    
    % Check if data length in header is equal to calculated data length
    if DataBytes(end)~=ChannelSamples(end)*Header.ChannelCount*2
        disp('Header and calculated data lengths are different!')
        fclose(FID);
        if ~isempty(FNameOut)
            fclose(FIDOut);
        end
        return
    end
    
    % Move back to beginning of last data segment
    fseek(FID,BegOfData(end),'bof');
    
    % Discarding info from extra data header and updating main header
    % An extra data header and a single sample of data are found when files are automatically split using firmware 6.03
    Header.DataBytes = DataBytes(end);
    Header.ChannelSamples = floor(ChannelSamples(end));    
end

if nargout>1
    % Determining system memory to maximize data segments
    switch SystemType
        case 'Windows'
            SystemMemory = regexp(evalc('feature memstats'),'\d*(?= MB)','match');
            SystemMemory = str2double(SystemMemory{2})*1e6; % Units bytes
        case 'Other'
            [~,SystemMemory] = system('free -m');
            SystemMemory = regexp(SystemMemory,'\d+','match');
            SystemMemory = str2double(SystemMemory{3})*1e6;
    end
    
    if isempty(Range)
        MaxSamples = floor((0.75*SystemMemory)/Header.DataBytes*Header.ChannelSamples);
        Range = double([1,min(MaxSamples,Header.ChannelSamples)]);
        if MaxSamples<Header.ChannelSamples
            fprintf('Warning!! Data is too large to load. Only the 1st %0.0f samples will be loaded.\n',MaxSamples)
        end
    end
    
    for k=1:size(Range,1) %If more than one range, send multiple data segments to an output cell
        % Seeking to beginning of data segment
        if exist('BegOfData','var')
            fseek(FID,BegOfData(end)+(Range(k,1)-1)*2*Header.ChannelCount,'bof');
        else
            fseek(FID,BegOfDataHeader(end)+(Range(k,1)-1)*2*Header.ChannelCount,'bof');
        end
        
        % Reading data
        fprintf('Loading %0.1f GB of data\n',(diff(Range(k,:))+1)*2*Header.ChannelCount/1e9)
        if size(Range,1)>1
            varargout{1}(k) = {fread(FID,[Header.ChannelCount,(diff(Range(k,:))+1)],'*int16')};
            if ~isempty(FNameOut)
                fwrite(FIDOut,varargout{1}{k},'*int16');
            end
        else
            varargout{1} = fread(FID,[Header.ChannelCount,(diff(Range(k,:))+1)],'*int16');
            if ~isempty(FNameOut)
                fwrite(FIDOut,varargout{1},'*int16');
            end
        end        
    end
end

% Closing file
fclose(FID);
if ~isempty(FNameOut)
    fclose(FIDOut);
end






