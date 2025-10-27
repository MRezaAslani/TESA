% Generate test signals
fs = 16000;
t = 0:1/fs:2;
x_clean = sin(2*pi*440*t') + 0.5*sin(2*pi*880*t'); % Clean: 440Hz + 880Hz
x_noisy = x_clean + 0.1*randn(size(x_clean)); % Add noise

% Compute target spectrogram
S_target = abs(stft(x_clean, fs, 'Window', hamming(256), 'OverlapLength', 224, 'FFTLength', 256));

% Run TESA
stft_params = struct('fs', fs, 'window', hamming(256), 'noverlap', 224, 'nfft', 256);
tesa_params = struct('lambda', 0.001, 'alpha', 0.1, 'num_iter', 1000, 'beta1', 0.9, 'beta2', 0.999);

tic;
[x_opt, loss] = tesa(x_noisy, S_target, stft_params, tesa_params);
toc;

% Visualize results
figure('Position', [100, 100, 1200, 800]);

% Time domain signals
subplot(3,1,1);
plot(t, x_clean, 'g-', 'LineWidth', 1.5); hold on;
plot(t, x_noisy, 'r--', 'LineWidth', 1);
plot(t, x_opt, 'b-', 'LineWidth', 1.5);
legend('Clean', 'Noisy', 'TESA Output', 'Location', 'best');
title('Time Domain Signals');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Spectrograms
subplot(3,1,2);
imagesc(abs(stft(x_noisy, fs))); axis xy; colorbar; title('Input Spectrogram');

subplot(3,1,3);
imagesc(abs(stft(x_opt, fs))); axis xy; colorbar; title('TESA Output Spectrogram');

sgtitle('TESA Reconstruction Results', 'FontSize', 14, 'FontWeight', 'bold');

% Loss convergence
figure;
semilogy(loss); 
xlabel('Iteration'); ylabel('Loss (log scale)');
title('TESA Optimization Convergence');
grid on;
