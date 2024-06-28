[num_rows, num_cols, num_bands] = size(hsi_normalized);
denoised_hsi_blf = zeros(num_rows, num_cols, num_bands);
for band = 1:num_bands
    % Read the noisy image
    input_image = hsi_normalized(:, :, band);
    
    % Set parameters
    w = 1; % Half-size of the window
    sigma_d = 8.5; % Spatial domain standard deviation
    sigma_r = 3; % Intensity range standard deviation
    
    % Apply the bilateral filter
    denoised_hsi_blf(:,:,band) = bilateral_filter_(input_image, w, sigma_d, sigma_r);
end
%%
function output = bilateral_filter_(input, w, sigma_d, sigma_r)
    % Read the input image
    input = im2double(input);
    
    % Get the dimensions of the input image
    [rows, cols, channels] = size(input);

    % Initialize the output image
    output = zeros(rows, cols, channels);

    % Create the spatial Gaussian filter
    [X, Y] = meshgrid(-w:w, -w:w);
    G = exp(-(X.^2 + Y.^2) / (2 * sigma_d^2));

    % Apply the bilateral filter to each channel
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                % Extract the local region
                iMin = max(i-w, 1);
                iMax = min(i+w, rows);
                jMin = max(j-w, 1);
                jMax = min(j+w, cols);
                region = input(iMin:iMax, jMin:jMax, c);

                % Compute the intensity Gaussian filter
                H = exp(-(region - input(i, j, c)).^2 / (2 * sigma_r^2));

                % Compute the bilateral filter response
                F = H .* G((iMin:iMax)-i+w+1, (jMin:jMax)-j+w+1);
                output(i, j, c) = sum(F(:) .* region(:)) / sum(F(:));
            end
        end
    end

    % Convert the output to the original image type
    output = im2uint8(output);
end


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
