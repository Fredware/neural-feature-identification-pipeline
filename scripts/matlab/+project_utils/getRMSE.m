function [movementRMSEs, xTalkRMSEs, varargout] = getRMSE(trueValue, predictedValue, varargin)
% % function [movementRMSEs, minWind, medWind, maxWind] = getRMSE(trueValue, predictedValue, varargin)
% % inputs:
% %       trueValue - either hand kinematics during training or target location
% %       predictedValue - hand kinematics predicted by decode
% %       type - analysis, either: 'targets' or 'training'.   Default is
% %       targets.
% % outputs:
% %       movementRMSEs - intended movements
% %       xTalkRMSEs - unintended movements
% 
% %     [d1,~] = size(trueValue);
% %     if(d1 > 12) trueValue = trueValue'; end
% %     [d1,~] = size(predictedValue);
% %     if(d1 > 12) predictedValue = predictedValue'; end
% 
% 
%     corrCoefs = corr(trueValue',predictedValue');
%     [DOFnum,~] = size(trueValue);
%     corrCoefs = corrCoefs(eye(DOFnum) == 1); % was originally eye(12). Update by Tara.
%     tempTarg = sign(diff(sign(sum(abs(trueValue),1))));
%     TargetStartIdxs = find(tempTarg == 1);
%     TargetStopIdxs = find(tempTarg == -1);
%     if(length(TargetStopIdxs) < length(TargetStartIdxs))
%         TargetStopIdxs = [TargetStopIdxs length(trueValue)];
%     end
%     if(length(TargetStartIdxs) ~= length(TargetStopIdxs))
%         TargetStartIdxs = [1 TargetStartIdxs];
%     end
%     movementRMSEs = zeros(1,length(TargetStartIdxs));
%     xTalkRMSEs = {};%zeros(1,length(TargetStartIdxs));
%     for qq = 1:length(TargetStartIdxs)
%         %window of data to look at
%         wind = TargetStartIdxs(qq):TargetStopIdxs(qq);
%         %which targets are active?
%         temp = sum(trueValue(:,wind),2);
%         %get RMSEs for all DOFs
%         decodeError = trueValue(:,wind) - predictedValue(:,wind);
%         squares = decodeError.^2;
%         tempRMSE = sqrt(mean(squares,2));
%         %which DOFs have targets (intended movement)
%         whichDOF = (temp ~= 0);
%         %which DOFs have x-talk? (non-intended movements)
%         anyMovement = (sum(abs(predictedValue),2) > 0);
%         xTalkDOFs = (anyMovement & ~whichDOF); %moves, but not intendeded movement
%         %update results
%         movementRMSEs(qq) = mean(tempRMSE(whichDOF));
% %         xTalkRMSEs(qq) = mean(tempRMSE(xTalkDOFs));
%         xTalkRMSEs{qq} = tempRMSE(xTalkDOFs);
%     end
%     %%Bret Added%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     [~, indicies] = sort(movementRMSEs);
%     [~,maxInd] = max(movementRMSEs);
%     [~,minInd] = min(movementRMSEs);
%     midInd = indicies(round(numel(indicies)/2));
%     minWind = TargetStartIdxs(minInd):TargetStopIdxs(minInd);
%     medWind = TargetStartIdxs(midInd):TargetStopIdxs(midInd);
%     maxWind = TargetStartIdxs(maxInd):TargetStopIdxs(maxInd);
%     if movementRMSEs(qq) == min(movementRMSEs(1:qq))
%         minWind = wind;
%     end
%     if movementRMSEs(qq) == max(movementRMSEs(1:qq))
%         maxWind = wind;
%     end
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     varargout{1} = corrCoefs;
% end





    corrCoefs = corr(trueValue',predictedValue');
    [DOFnum,~] = size(trueValue);
    corrCoefs = corrCoefs(eye(DOFnum) == 1); % was originally eye(12). Update by Tara.
    tempTarg = sign(diff(sign(sum(abs(trueValue),1))));
    TargetStartIdxs = find(tempTarg == 1);
    TargetStopIdxs = find(tempTarg == -1);
    if(length(TargetStopIdxs) < length(TargetStartIdxs))
        TargetStopIdxs = [TargetStopIdxs length(trueValue)];
    end
    if(length(TargetStartIdxs) ~= length(TargetStopIdxs))
        TargetStartIdxs = [1 TargetStartIdxs];
    end
    movementRMSEs = cell(size(trueValue,1),1);
    xTalkRMSEs = cell(size(trueValue,1),1);
    for qq = 1:length(TargetStartIdxs)
        %window of data to look at
        wind = TargetStartIdxs(qq):TargetStopIdxs(qq);
        %which targets are active?
        temp = sum(trueValue(:,wind),2);
        %get RMSEs for all DOFs
        decodeError = trueValue(:,wind) - predictedValue(:,wind);
        squares = decodeError.^2;
        tempRMSE = sqrt(mean(squares,2));
        %which DOFs have targets (intended movement)
        whichDOF = (temp ~= 0);
        %update results
        for i = 1:length(whichDOF)
            if whichDOF(i)
                movementRMSEs{i} = [movementRMSEs{i} tempRMSE(i)];
            else
                xTalkRMSEs{i} = [xTalkRMSEs{i} tempRMSE(i)];
            end
        end
    end
    varargout{1} = corrCoefs;
end