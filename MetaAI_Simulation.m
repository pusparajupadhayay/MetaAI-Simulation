%% MetaAI: Metasurface-Driven Physical Neural Networks Simulation



clear; close all; clc;


%% Simulation Parameters
rng(42); % For reproducibility


% Network parameters

U = 64;        % Input dimension 
R = 10;        % Output classes
M = 256;       % Number of meta-atoms
num_samples = 1000; % Number of test samples


% Wireless parameters

SNR_dB = 20;   % Signal-to-noise ratio
symbol_rate = 1e6; % Symbol rate (1 MHz)


% Metasurface parameters

phase_states = [0, pi/2, pi, 3*pi/2]; % 2-bit phase states
num_states = length(phase_states);


%% Generate Synthetic Dataset (MNIST-like)
fprintf('Generating synthetic dataset...\n');


% Create synthetic input data 
X = randn(U, num_samples); % Input features
X = X ./ max(abs(X(:)));   % Normalize to [-1, 1]


% Generate random labels for demo
true_labels = randi([1, R], num_samples, 1);
Y = full(ind2vec(true_labels'))'; % One-hot encoding


% Split into training and testing
train_ratio = 0.8;
num_train = floor(train_ratio * num_samples);
train_indices = 1:num_train;
test_indices = (num_train+1):num_samples;


X_train = X(:, train_indices);
Y_train = Y(train_indices, :);
X_test = X(:, test_indices);
Y_test = Y(test_indices, :);


%% Train Complex-Valued Neural Network (Digital)
fprintf('Training complex-valued neural network...\n');


% Initialize complex weights
W_desired = (randn(R, U) + 1j*randn(R, U)) / sqrt(U);


% Training parameters
learning_rate = 8e-3;
num_epochs = 60;
batch_size = 64;


% Training loop
for epoch = 1:num_epochs
    epoch_loss = 0;
    
    for batch = 1:ceil(num_train/batch_size)
        % Get batch
        start_idx = (batch-1)*batch_size + 1;
        end_idx = min(batch*batch_size, num_train);
        batch_size_actual = end_idx - start_idx + 1;
        
        X_batch = X_train(:, start_idx:end_idx);
        Y_batch = Y_train(start_idx:end_idx, :);
        
        % Forward pass
        Z = W_desired * X_batch; % Complex matrix multiplication
        Y_pred = abs(Z); % Magnitude for classification
        Y_pred = Y_pred ./ sum(Y_pred, 1); % Softmax-like normalization
        
        % Compute loss (cross-entropy)
        loss = -sum(sum(Y_batch' .* log(Y_pred + 1e-8))) / batch_size_actual;
        epoch_loss = epoch_loss + loss;
        
        % Backward pass (complex gradient)
        dL_dZ = (Y_pred - Y_batch') .* sign(Z); % Complex gradient
        dL_dW = dL_dZ * X_batch' / batch_size_actual;
        
        % Update weights
        W_desired = W_desired - learning_rate * dL_dW;
    end
    
    if mod(epoch, 10) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, epoch_loss/batch);
    end
end


%% Metasurface Configuration Optimization
fprintf('Optimizing metasurface configuration...\n');


% Initialize meta-atom phases randomly
phi_m = randi([1, num_states], M, 1); % Random initial phases


% Propagation phase offsets (simplified far-field model)
theta = 30 * pi/180; % Angle in radians
d_s = 0.05; % Spacing between meta-atoms (wavelengths)
k0 = 2*pi; % Wave number


phi_m_rho = zeros(M, 1);
for m = 1:M
    phi_m_rho(m) = k0 * (m-1) * d_s * cos(theta);
end


% Amplitude offset (same for all meta-atoms in far-field)
alpha_rho = 1.0;


% Optimize metasurface configuration for each output class
H_mts = zeros(R, 1); % Metasurface channel responses
phi_optimized = zeros(M, R);


for r = 1:R
    % Target weight for this output
    W_target = W_desired(r, :);
    
    % Optimize phase configuration
    best_loss = inf;
    best_phi = phi_m;
    
    % Simple optimization: try multiple random configurations
    num_configs = 1000;
    for config = 1:num_configs
        % Random phase configuration
        phi_candidate = randi([1, num_states], M, 1);
        phi_vals = phase_states(phi_candidate)';
        
        % Compute metasurface channel response
        H_candidate = alpha_rho * sum(exp(1j*(phi_vals + phi_m_rho)));
        
        % Compute loss (approximation error)
        loss = mean(abs(H_candidate - mean(W_target)).^2);
        
        if loss < best_loss
            best_loss = loss;
            best_phi = phi_candidate;
            H_mts(r) = H_candidate;
        end
    end
    
    phi_optimized(:, r) = best_phi;
end


fprintf('Metasurface optimization completed.\n');


%% Over-the-Air Computation Simulation
fprintf('Simulating over-the-air computation...\n');


% Sequential transmission and computation
Y_pred_physical = zeros(size(Y_test));


for sample = 1:size(X_test, 2)
    x = X_test(:, sample); % Input sample
    
    % Simulate sequential transmission and computation
    y_physical = zeros(R, 1);
    
    for r = 1:R
        % Get metasurface configuration for this output
        phi_r = phi_optimized(:, r);
        phi_vals = phase_states(phi_r)';
        
        % Compute channel response
        H_r = alpha_rho * sum(exp(1j*(phi_vals + phi_m_rho)));
        
        % Sequential computation: accumulate over input symbols
        accumulation = 0;
        for i = 1:U
            % Apply channel response to input symbol
            symbol_output = H_r * x(i);
            
            % Add noise (simulating wireless channel)
            noise_power = 10^(-SNR_dB/10);
            noise = sqrt(noise_power/2) * (randn(1) + 1j*randn(1));
            symbol_output_noisy = symbol_output + noise;
            
            % Accumulate results
            accumulation = accumulation + symbol_output_noisy;
        end
        
        y_physical(r) = abs(accumulation);
    end
    
    % Normalize outputs (softmax-like)
    y_physical = y_physical / sum(y_physical);
    Y_pred_physical(sample, :) = y_physical';
end



%% Performance Evaluation
fprintf('Evaluating performance...\n');
% Digital baseline (for comparison)
Y_pred_digital = W_desired * X_test; % Likely R x N result from W_desired (R x U) * X_test (U x N)
Y_pred_digital = abs(Y_pred_digital);
Y_pred_digital = Y_pred_digital ./ sum(Y_pred_digital, 1);
Y_pred_digital = Y_pred_digital'; 

% Convert to predicted labels
% All outputs from max(..., [], 2) are now N x 1 column vectors
[~, pred_labels_physical] = max(Y_pred_physical, [], 2);
[~, pred_labels_digital] = max(Y_pred_digital, [], 2);
[~, true_labels_test] = max(Y_test, [], 2);

pred_labels_physical = pred_labels_physical(:);
pred_labels_digital  = pred_labels_digital(:);
true_labels_test     = true_labels_test(:);

% Calculate accuracies
% These lines now compare N x 1 vectors
accuracy_physical = sum(pred_labels_physical == true_labels_test) / length(true_labels_test);
accuracy_digital = sum(pred_labels_digital == true_labels_test) / length(true_labels_test);

fprintf('Physical (MetaAI) Accuracy: %.2f%%\n', accuracy_physical * 100);
fprintf('Digital Baseline Accuracy: %.2f%%\n', accuracy_digital * 100);



%% Visualization
figure('Position', [100, 100, 1200, 800]);


% Accuracy comparison
subplot(2, 3, 1);
methods = {'MetaAI (Physical)', 'Digital Baseline'};
accuracies = [accuracy_physical, accuracy_digital] * 100;
bar(accuracies);
set(gca, 'XTickLabel', methods);
ylabel('Accuracy (%)');
title('Classification Accuracy Comparison');
grid on;


for i = 1:length(accuracies)
    text(i, accuracies(i)+1, sprintf('%.1f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end


% Weight distribution
subplot(2, 3, 2);
scatter(real(W_desired(:)), imag(W_desired(:)), 20, 'filled', 'b');
xlabel('Real Part');
ylabel('Imaginary Part');
title('Desired Weights Distribution');
grid on;
axis equal;


% Metasurface phase configuration
subplot(2, 3, 3);
phase_config = reshape(phi_optimized(:, 1), [16, 16]); % First output
imagesc(phase_config);
colorbar;
title('Metasurface Phase Configuration');
xlabel('Meta-atom X');
ylabel('Meta-atom Y');


% Confusion matrix (physical)
subplot(2, 3, 4);
conf_mat_physical = confusionmat(true_labels_test, pred_labels_physical);
imagesc(conf_mat_physical);
colorbar;
title('Physical Model Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');


% Confusion matrix (digital)
subplot(2, 3, 5);
conf_mat_digital = confusionmat(true_labels_test, pred_labels_digital);
imagesc(conf_mat_digital);
colorbar;
title('Digital Model Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');


% Output magnitude distribution
subplot(2, 3, 6);
sample_outputs = Y_pred_physical(1:min(50, size(Y_pred_physical,1)), :);
imagesc(sample_outputs');
colorbar;
title('Sample Output Magnitudes');
xlabel('Sample Index');
ylabel('Class');


sgtitle('MetaAI Simulation Results');


%% Parallelism Analysis (Subcarrier-based)
fprintf('Analyzing subcarrier-based parallelism...\n');


num_subcarriers = [1, 2, 4, 8]; % Different parallelism levels
parallel_accuracies = zeros(size(num_subcarriers));


for p_idx = 1:length(num_subcarriers)
    K = num_subcarriers(p_idx);
    
    % Simplified parallelism simulation
    % In practice, this would involve more complex optimization
    Y_pred_parallel = zeros(size(Y_test));
    
    for sample = 1:size(X_test, 2)
        x = X_test(:, sample);
        y_parallel = zeros(R, 1);
        
        % Process in parallel groups
        symbols_per_group = ceil(U / K);
        
        for group = 1:K
            start_idx = (group-1)*symbols_per_group + 1;
            end_idx = min(group*symbols_per_group, U);
            
            if start_idx > U
                break;
            end
            
            % Use different output for each group (simplified)
            output_idx = mod(group-1, R) + 1;
            phi_group = phi_optimized(:, output_idx);
            phi_vals = phase_states(phi_group)';
            
            H_group = alpha_rho * sum(exp(1j*(phi_vals + phi_m_rho)));
            
            % Process symbols in this group
            for i = start_idx:end_idx
                symbol_output = H_group * x(i);
                noise = sqrt(noise_power/2) * (randn(1) + 1j*randn(1));
                y_parallel(output_idx) = y_parallel(output_idx) + abs(symbol_output + noise);
            end
        end
        
        Y_pred_parallel(sample, :) = y_parallel' / sum(y_parallel);
    end
    
    [~, pred_labels_parallel] = max(Y_pred_parallel, [], 2);
    parallel_accuracies(p_idx) = sum(pred_labels_parallel == true_labels_test) / length(true_labels_test);
end


figure;
plot(num_subcarriers, parallel_accuracies*100, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Subcarriers');
ylabel('Accuracy (%)');
title('Parallelism vs Accuracy Trade-off');
grid on;


%% Performance Summary
fprintf('\n=== METAAI SIMULATION SUMMARY ===\n');
fprintf('Physical Model Accuracy: %.2f%%\n', accuracy_physical * 100);
fprintf('Digital Baseline Accuracy: %.2f%%\n', accuracy_digital * 100);
fprintf('Accuracy Gap: %.2f%%\n', (accuracy_digital - accuracy_physical) * 100);
fprintf('Number of Meta-atoms: %d\n', M);
fprintf('Input Dimension: %d\n', U);
fprintf('Output Classes: %d\n', R);


% Calculate theoretical speedup
sequential_time = U * R; % Sequential operations
parallel_time = U * ceil(R/max(num_subcarriers)); % With parallelism
speedup = sequential_time / parallel_time;


fprintf('Theoretical Speedup with Parallelism: %.2fx\n', speedup);
