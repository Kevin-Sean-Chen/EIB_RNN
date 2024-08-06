% PSet1_PCA

load('HatsopoulosReachTask.mat')
%% numNeurons x numTimebins x 8
neuron_time_direction = zeros(numNeurons, numTimebins, 8);

for d = 1:8
    dir_pos = find(direction==d);
    neuron_time_direction(:,:,d) = neuron_time_direction(:,:,d) + mean(firingRate(:,:,dir_pos),3);
end

neuronbytime8 = reshape(neuron_time_direction, numNeurons, numTimebins*8);
neuronbytime8 = neuronbytime8 - mean(neuronbytime8,2);  % correct way
% neuronbytime8 = neuronbytime8 - mean(neuronbytime8,1);  % for columns wise demean!
% neuronbytime8 = (neuronbytime8 - mean(neuronbytime8,2)) ./ std(neuronbytime8,[], 2);  % z-scoring
figure
imagesc(neuronbytime8)
xlabel('time by directions');
ylabel('neurons')

%% covariance
cov_matrix = neuronbytime8 * neuronbytime8' / size(neuronbytime8,2);
figure
imagesc(cov_matrix)

%% decompution
[uu,ss,vv] = svd(cov_matrix);
figure()
plot(diag(ss),'-o')

var_scale = cumsum(diag(ss))/sum(ss(:));
figure()
plot(var_scale)

temp = find(var_scale>0.9);
cutoff = temp(1)  % the cutoff for 90%

%% projections
PCs = uu(:,1:3);  % first three PCs

projec = PCs'*neuronbytime8;
low_d_trajectory = reshape(projec, 3, numTimebins, 8);
colors = jet(8);

figure
for ii = 1:8
    dir_id = ii;
    plot3(low_d_trajectory(1,:,dir_id), low_d_trajectory(2,:,dir_id), low_d_trajectory(3,:,dir_id),'Color',colors(ii,:))
    hold on
end

%%% make color continiuous

%% random tests..
%% column demean
%% z-scoring

%% SVD
X = neuronbytime8*1;
[uu,ss_svd,vv] = svd(X);

figure;
plot(diag(ss_svd)); hold on
plot(diag(ss))
plot(diag(ss_svd).^2/numTimebins/8,'--')
legend({'SVD','PCA','convert'})

%% projection with singular vectors
proj_svd = uu(:,1:3)'*X;
low_d_trajectory_svd = reshape(proj_svd, 3, numTimebins, 8);

figure
for ii = 1:8
    dir_id = ii;
    plot3(low_d_trajectory_svd(1,:,dir_id), low_d_trajectory_svd(2,:,dir_id), low_d_trajectory_svd(3,:,dir_id));
    hold on
end

%% decoding
newX = reshape(firingRate, numNeurons, numTrials*numTimebins)';
region_id = [];
for ii = 1:length(brainRegion)
    if strcmp(brainRegion{ii}, 'PMd')==1  %MI,,PMd
        region_id = [region_id ii];
    end
end
feature_matrix = squeeze(firingRate(region_id, 15, :))';  %%% using the middle time point for now
% feature_matrix(:, 24) = [];
%%% naive split
reach_binary = direction*0;
reach_binary(direction<=4) = 0;
reach_binary(direction>4) = 1;
%%% XOR split
% reach_binary = direction*0;
% pos = find(direction==3 | direction==4 | direction==7 | direction==8);
% reach_binary(pos) = ones(1,length(pos));

% mdl = fitclinear(feature_matrix, reach_binary);
% mdl = fitcdiscr(feature_matrix, reach_binary);

n_repeats = 50;
all_cvs = zeros(1, n_repeats);

for nn = 1:n_repeats

    % CVmodel = fitclinear(feature_matrix, reach_binary, 'DiscrimType', 'pseudoLinear', 'KFold', 10);
    CVmodel = fitclinear(feature_matrix, reach_binary, 'KFold', 5);
    % Calculate the cross-validation loss (misclassification error)
    cvloss = 1 - kfoldLoss(CVmodel);
    all_cvs(nn) = cvloss;
    % [predictions, scores] = predict(mdl, feature_matrix);
    % sum(predictions==reach_binary)/length(reach_binary)

end

figure;plot(all_cvs)

%% 
