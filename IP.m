clear; clc;

%% Inputs

dffIns = importdata('avgORNs.mat'); % deltaF/F traces of input units, specified as TxN matrix
dffOuts = importdata('avgPNs.mat'); % deltaF/F traces of output units, specified as TxN matrix

dt = .098; % sampling time interval, s
train_intervals = 1:320; % intervals of points to include in training

% Constraints on feedforward excitatory interactions
lbsi = -20; % lower bound self-interaction
ubsi = 0; % upper bound self-interaction
lb = 0; % lower bound of input units to output units weights
ub = 10; % upper bound of input units to output units weights
lambda = 10; % lambda parameter for l1 regularization

save_flag = true; % save the (nIns+1)x(nOuts) interaction matrix Z? 
plot_flag = true; % plot predictions?
which_plots = 1:size(dffOuts,2); % which output units to plot?

%% Inverse problem

nIns = size(dffIns,2); % number of input units
nOuts = size(dffOuts,2); % number of output units

t = dt:dt:(size(dffIns,1)-1)*dt; % time, in s
T = length(t); % number of time points

Z = zeros(1+nIns, nOuts); % interaction matrix

% set constraints
boundIns = zeros(1,nIns);
C = zeros(nIns+1, nIns+1);
v = [0 boundIns+1];
C(1,:) = v;
d = [lambda; boundIns'];
lb = [lbsi boundIns+lb]';
ub = [ubsi boundIns+ub]';

for out_index = 1:nOuts
     
    % align time points
    dffIns_shift = dffIns(2:end,:);
    dffOuts_shift = dffOuts(2:end,:);
    dffOuts_shift(:,out_index) = dffOuts(1:end-1,out_index);
    
    A = zeros(T,nIns+nOuts);

    A(:,1) = dffOuts_shift(:,out_index);
    for i=1:size(dffIns_shift,2)
        A(:,i+1) = dffIns_shift(:,i);
    end
    for i=1:size(dffOuts_shift,2)
        A(:,nIns+i+1) = dffOuts_shift(:,i);
    end
    
    A(:,nIns+1+out_index) = []; % remove out_index unit because it was already added in A(:,1)

    y = ( dffOuts(2:end,out_index) - dffOuts(1:end-1,out_index) )/dt;

    % train on part of the protocol
    y = y(train_intervals);
    A = A(train_intervals,:);
    
    % compute weights
    z = lsqlin(A(:,1:1+nIns),y,C,d,[],[],lb,ub);
        
    Z(:,out_index) = z;
end

Z( abs(Z)<1e-3 ) = 0;

if save_flag
    save('Z.mat','Z')
end

%% Visualize reconstruction

predictions = []; % TxN matrix of predictions

if plot_flag 
    for out_index = which_plots
        dffOut = dffOuts(1:end-1,out_index);
        testState_Out = dffOut(1);
        test_dffOut = zeros(T,1);
        test_dffOut(1) = testState_Out;

        B = zeros(T,nIns);
        for i=1:size(dffIns_shift,2)
            B(:,i) = dffIns_shift(:,i);
        end

        for i = 2:T
            testState_Out = testState_Out + dt*( B(i-1,:)*Z(2:end,out_index) ) + dt*( testState_Out*Z(1,out_index) );
            test_dffOut(i) = testState_Out;
        end

        predictions(:,end+1) = test_dffOut;
        figure
        hold on
        plot(t, test_dffOut, 'r.-', 'LineWidth', 3, 'MarkerSize', 20)
        plot(t, dffOut, 'b-', 'LineWidth', 1)
        plot(t(train_intervals), dffOut(train_intervals), 'g-', 'LineWidth', 1)
        hold off
        legend('Reconstructed trace','Recorded trace','Location','northeast')
        if out_index>3
            xlabel('Time (s)')
        end
        if out_index==1 || out_index==4
            ylabel('\DeltaF/F0')
        end
        set(gca,'FontSize',20)
    end
end

save('predOuts.mat','predictions')
