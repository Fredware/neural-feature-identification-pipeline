function [TRAIN, X, Z]=kalman3_pm_train(x,z)
% A third order Kalman Filter based off a muscle pulling against an
% antagonist modeled as a spring
% MAT 20250525

% Set up X to contain positions and velocities
% SG FILTER
x_sg = sgolayfilt(x, 2, 9, [], 2);
Ts = 1./30;
vel_sg = diff(x_sg').*Ts;
vel_sg = vel_sg';
vel_sg = sgolayfilt(vel_sg, 2, 9, [], 2);
acl_sg = diff(vel_sg').*Ts;
acl_sg = acl_sg';
vel_sg = [vel_sg zeros(size(x,1),1) ]; % append a zero to keep the size
acl_sg = [acl_sg zeros(size(x,1),2) ]; % append a zero to keep the size
X = zeros(size(x_sg,1) + size(vel_sg,1) + size(acl_sg,1), size(x,2));
X(1:3:end,:) = x_sg;
X(2:3:end,:) = vel_sg;
X(3:3:end,:) = acl_sg;

Z = z;

% version = mfilename;
% calculate mu's, the means of kinematics and features, and subtract off
xmu=zeros(size(X,1),1); %mean(x,2); x=x-repmat(PARAM.xmu,[1 size(x,2)]);
zmu=zeros(size(z,1),1); %mean(z,2); z=z-repmat(PARAM.zmu,[1 size(z,2)]);

% length of the signals
N=size(X,1);
M=size(X,2);

% calculate k and a, the state-to-state transformation for velocity
a = (X(:,1:end-1)'\acl_sg(:,2:end)')';
ts_mat = circshift(eye(3.*size(a,1)).*Ts,-1);
ts_mat(:,1) = 0;
A = eye(3.*size(a,1)) + ts_mat; 
A(3:3:end) = a;
TRAIN.A=A;
% SSTRAIN.A = x(:,2:M)/x(:,1:(M-1));  %%

% calculate W, the covariance of the noise in the kinematics
W1=X(:,2:M)*X(:,2:M)';
W2=X(:,1:(M-1))*X(:,2:M)';
W=(1/(M-1))*(W1-TRAIN.A*W2);
W = (W+W')/2; %Added to force symmetry (needed to use dare to solve for steady state).
TRAIN.W = W;

% cross-correlation and autocorrelations of x and z
TRAIN.Pzx=z(:,1:M)*X(:,1:M)';
TRAIN.Rxx=X(:,1:M)*X(:,1:M)';
TRAIN.Rzz=z(:,1:M)*z(:,1:M)';

% calculate H, the transformation matrix from measured features to state
% SSTRAIN.H=SSTRAIN.Pzx/(SSTRAIN.Rxx); %%
TRAIN.H=TRAIN.Pzx*pinv(TRAIN.Rxx);

% calculate Q, the covariance of noise in the measured features
TRAIN.Q=(1/M)*(TRAIN.Rzz-TRAIN.H*TRAIN.Pzx');

TRAIN.N = N; %# of kinematics (not counting velocities)