 function [TRAIN,x,z]=kalman_train(x,z)

% length of the signals
M=size(x,2); %number of samples

% calculate A, the state-to-state transformation (hand kinematics)
A1=x(:,2:M)*x(:,1:(M-1))';
A2=x(:,1:(M-1))*x(:,1:(M-1))';
TRAIN.A=A1*pinv(A2);

% calculate W, the covariance of the noise in the kinematics
W1=x(:,2:M)*x(:,2:M)';
W2=x(:,1:(M-1))*x(:,2:M)';
TRAIN.W=(1/(M-1))*(W1-TRAIN.A*W2);

% cross-correlation and autocorrelations of x and z
TRAIN.Pzx=z(:,1:M)*x(:,1:M)';
TRAIN.Rxx=x(:,1:M)*x(:,1:M)';
TRAIN.Rzz=z(:,1:M)*z(:,1:M)';

% calculate H, the transformation matrix from measured features to state
TRAIN.H=TRAIN.Pzx*pinv(TRAIN.Rxx);

% calculate Q, the covariance of noise in the measured features
TRAIN.Q=(1/M)*(TRAIN.Rzz-TRAIN.H*TRAIN.Pzx');
