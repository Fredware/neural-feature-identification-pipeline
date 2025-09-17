function [FlMask, ExMask, Mask] = isolate_mask_bhm(Mask,FlMask,ExMask,Kinematics,ChosenKins)
% Find Train Windows
StartStops = diff(Mask);
StartInds = find(StartStops == 1);
StopInds = find(StartStops == -1);
if(length(StopInds) < length(StartInds))
    StopInds = [StopInds length(Mask)];
end

if(length(StartInds) ~= length(StopInds))
    StartInds = [1 StartInds];
end

% Remove Train windows of unused kinematics
for k = 1:length(StartInds)
    wind = [StartInds(k):StopInds(k)];
    if sum(Kinematics(ChosenKins,wind)) <= 0
        FlMask(wind) = zeros(1,numel(wind));   % Get rid of 0 and Extensions for Flexion Train Mask
    end
    if sum(Kinematics(ChosenKins,wind)) >= 0
        ExMask(wind) = zeros(1,numel(wind));   % Get rid of 0 and Flexions for Extension Train Mask
    end
    if sum(Kinematics(ChosenKins,wind)) == 0
        Mask(wind) = zeros(1,numel(wind));   % Get rid of 0 and Extensions for Flexion Train Mask
    end
end
end

