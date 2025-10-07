% Chaotic_2D_EI_network
close all
clc

%% drawing options
%%%  drawing=1 for plotting video; drawing=0 for not
%%%  press space to paus during the video
drawing = 1;
% drawing = 0;

%% setup network
%%% important parameters and size of the 2D sheet
L = 50;  % size of the square 2D space
N = L^2;  % total number of neurons
tau_e = 0.005;  % in seconds, so this is 5ms
sig_e = 0.1;  % in between 0-1
tau_i = 3.0*tau_e;  % write this to show the ratio (tau_i/tau_e)
sig_i = 2.0*sig_e;  % (sig_i/sig_e)
rescale = 1.;  % tune this to bring all strength up and down by a factor, if needed for finite size
%%%% emperical note for parameters %%%
%%% tau, sigma
%%% 6ms, 0.16  ### grid/lattice parameter
%%% 15ms, 0.2  ### chaotic droplets
%%% 10ms, 0.14 ### coherent waves/strips
%%% 10ms,  0.2  ### blinking dots
%%%% stabalizes ~>50-100 steps

%% scaling for EI network
% EI weights scaled by the spatial kernel size to have sqrt(K) signal
% the condition is chosen to have positive firing and keep EI balance
%%%  mu_e/mu_i > Wei/Wii > Wee/Wie  %%%
Wee = 1  * (L^2*sig_e^2*pi)^0.5 *rescale;
Wei = -2  *  (L^2*sig_i^2*pi)^0.5 *rescale;
Wie = .99  *  (L^2*sig_e^2*pi)^0.5 *rescale;
Wii = -1.8  *  (L^2*sig_i^2*pi)^0.5 *rescale;
mu_e = 1  *rescale;  % assume uniform baseline
mu_i = .8  *rescale;

%% initializations
%%% define time scale and vectors
dt = 0.001;  % 1ms time steps
T = .5;  % a few seconds of simulation would work
time = 0:dt:T;
lt = length(time);
re_xy = zeros(L,L,lt);
ri_xy = zeros(L,L,lt);
kernel_size = 23; % pick a prime number for more stable numerical convolution

%%% initalize matrix for E,I rate and current signal
re_init = rand(L,L);
ri_init = rand(L,L);
re_xy(:,:,1) = re_init;
ri_xy(:,:,1) = ri_init;
he_xy = re_xy*1;
hi_xy = ri_xy*1;

%%% measure current for one cell
measure_e = zeros(1,lt);
measure_i = zeros(1,lt);

%%% measure for the the field to compute balance index
measure_mu = zeros(L,L,lt);
measure_mu_ex = zeros(L,L,lt);

%% neural dynamcs
hFig = figure; 
% Set up the figure to detect key presses
set(hFig, 'KeyPressFcn', @(src, event) setappdata(src, 'KeyPressed', true));
setappdata(hFig, 'KeyPressed', false);  % Initialize the key press flag

for tt = 1:lt-1
    %%% 2D rate EI-RNN dynamics
    ge_conv_re = spatial_convolution(re_xy(:,:,tt), g_kernel(sig_e, kernel_size));
    gi_conv_ri = spatial_convolution(ri_xy(:,:,tt), g_kernel(sig_i, kernel_size));
    he_xy(:,:,tt+1) = he_xy(:,:,tt) + dt/tau_e*( -he_xy(:,:,tt) + Wee*ge_conv_re + Wei*gi_conv_ri + mu_e);
    hi_xy(:,:,tt+1) = hi_xy(:,:,tt) + dt/tau_i*( -hi_xy(:,:,tt) + Wie*ge_conv_re + Wii*gi_conv_ri + mu_i);
    re_xy(:,:,tt+1) = phi(he_xy(:,:,tt+1));
    ri_xy(:,:,tt+1) = phi(hi_xy(:,:,tt+1));

    %%% make measurements
    measure_e(tt+1) = Wee*ge_conv_re(20,20) + mu_e;
    measure_i(tt+1) = Wei*gi_conv_ri(20,20) + mu_i;

    measure_mu(:,:,tt+1) = abs(Wee*ge_conv_re + Wei*gi_conv_ri + mu_e);
    measure_mu_ex(:,:,tt+1) = Wee*ge_conv_re + mu_e;

    %%% drawing
    if drawing==1
        imagesc(squeeze(re_xy(:,:,tt+1)));
        colormap('gray');      % Set the colormap (optional)
        colorbar;              % Show the colorbar (optional)
        title(['Frame ' num2str(tt)]);  % Display the frame number
        axis equal;            % Keep axis proportions
        axis tight;            % Remove any extra white space
        drawnow;               % Update the figure window
        pause(0.01); 
    end
    
    if getappdata(hFig, 'KeyPressed')
        disp('Key pressed. Exiting loop.');
        break;  % Exit the loop if any key is pressed
    end

end

%% some analysis
offset = 50;  % cutoff inital segment for stationary analysis
%%% EI currents
figure();
plot(measure_e(offset:end),'b', 'DisplayName', 'E'); hold on
plot(measure_i(offset:end),'r', 'DisplayName', 'I')
plot(measure_e(offset:end) + measure_i(offset:end),'k', 'DisplayName', 'total')
title('E-I inputs');
ylabel('current input')
xlabel('time steps')
legend()

%%% rate examples
figure()
subplot(211)
temp = reshape(re_xy,N,lt);
samp_neurons = randperm(N-1, 10);
plot(temp(samp_neurons,offset:end)')
title('Time Evolution of Selected r_e Neurons');
ylabel('rate');
xlabel('time steps');
subplot(212)
temp = reshape(ri_xy,N,lt);
plot(temp(samp_neurons,offset:end)')
title('Time Evolution of Selected r_i Neurons');
ylabel('rate');
xlabel('time steps');

%%% the spectrum
figure()
[power,frequency] = group_spectrum(re_xy(:,:,offset:end), dt);
loglog(frequency, power);
title('population power spectrum')
xlabel('Frequency (Hz)')
ylabel('Power')

%%% balance index
figure()
beta_t = abs(measure_mu)./measure_mu_ex;
beta_t = squeeze(mean(mean(beta_t(:,:,offset:end),1),2));
plot(beta_t, 'DisplayName','measurement'); hold on
plot([0, length(beta_t)], [0.1, 0.1],'k--','DisplayName','tight-balance')
plot([0, length(beta_t)], [1, 1],'g--','DisplayName','loose-balance')
title('balance index')
xlabel('time steps')
ylabel('beta'); legend()

%%% firing distribution
figure()
temp = reshape(re_xy,N,lt);
hist(mean(temp,2),50)
title('r_e histogram')
xlabel('rate')
ylabel('count')

%% functions
function nl = phi(x)
    nl = zeros(size(x));  % Initialize output array
    nl(x > 0) = tanh(x(x > 0)) * 1;  % Apply tanh(x) for positive elements
    % nl(x > 0) = (x(x > 0)) * 1;  % for ReLU, works for moderate ratios
end

function kernel = g_kernel(sigma, size)
    % Generates a 2D Gaussian kernel.
    
    sigma = sigma * size; % Adjust sigma by size
    [x, y] = meshgrid(0:size-1, 0:size-1); % Create meshgrid for 2D coordinates
    
    % Calculate the 2D Gaussian kernel
    kernel = (1 / (2 * pi * sigma^2)) * exp(-((x - (size - 1) / 2).^2 + (y - (size - 1) / 2).^2) / (2 * sigma^2));

    % Normalize the kernel
    kernel = kernel / sum(kernel(:));
end

function gr = spatial_convolution(r, k)
    % 2D spatial convolution with periodic boundary conditions
    % Input:
    %   r: Input matrix (neural field)
    %   k: Convolution kernel
    % Output:
    %   gr: Result of convolution with the same size as r

    % Pad the array with wrapped edges (circular padding)
    pad_size = floor(size(k) / 2);
    r_padded = padarray(r, pad_size, 'circular');

    % Perform the 2D convolution
    gr = conv2(r_padded, k, 'same');

    % Extract the central part to get the same size as the input
    gr = gr(pad_size(1)+1:end-pad_size(1), pad_size(2)+1:end-pad_size(2));
end

function [power_spectrum, frequencies] = spectrum_analysis(time_series, dt)
    % Perform FFT
    fft_result = fft(time_series);
    
    % Compute the power spectrum
    power_spectrum = abs(fft_result).^2;
    
    % Get the corresponding frequencies
    N = length(time_series);          % Number of data points
    frequencies = (0:(N/2)-1) / (N*dt); % Frequencies up to the Nyquist frequency

    % Only return the first half (positive frequencies)
    power_spectrum = power_spectrum(1:N/2);
    frequencies = frequencies(1:N/2);
end

function [mean_spectrum, frequencies] = group_spectrum(data, dt)
    % Get the dimensions of the data
    [N, ~, T] = size(data);
    
    % Perform spectrum analysis on the first element to get frequency vector
    [pp, frequencies] = spectrum_analysis(squeeze(data(1,1,:)), dt);
    
    % Initialize the 3D matrix to store the spectra
    spec_all = zeros(N, N, length(frequencies));
    
    % Loop over the first two dimensions to calculate spectra for each (ii, jj) pair
    for ii = 1:N
        for jj = 1:N
            temp = squeeze(data(ii, jj, :));
            [spec_all(ii, jj, :), ~] = spectrum_analysis(temp, dt);
        end
    end
    
    % Average over the first two dimensions
    mean_spectrum = mean(mean(spec_all, 1), 2);
    mean_spectrum = squeeze(mean_spectrum); % Remove singleton dimensions
end
