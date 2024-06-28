% Parameters
block_size = 8; % Size of the blocks (patches)
search_window = 21; % Size of the search window
threshold_ = 2.7; % Hard thresholding parameter
%%
[num_rows, num_cols, num_bands] = size(hsData);
denoised_hsi_bm3d = zeros(num_rows, num_cols, num_bands);
for band = 1:num_bands %[output:group:23d4c112]
    % Load the noisy image
    noisy_image = hsData(:, :, band);
    
    % Get image dimensions
    [height, width] = size(noisy_image);
    
    % Basic estimation stage
    basic_estimate = bm3d_basic_estimate(noisy_image, block_size, search_window, threshold_); %[output:912dc040]
    
    % Final estimation stage
    denoised_hsi_bm3d(:,:,band) = bm3d_final_estimate(noisy_image, basic_estimate, block_size, search_window);
end %[output:group:23d4c112]
%%
%[text] Basic esti
%%
function basic_estimate = bm3d_basic_estimate(image, block_size, search_window, threshold)
    [height, width] = size(image);
    basic_estimate = zeros(size(image));
    weight = zeros(size(image));
    
    % Loop over each block in the image
    for i = 1:block_size:height-block_size+1
        for j = 1:block_size:width-block_size+1
            % Extract the reference block
            ref_block = image(i:i+block_size-1, j:j+block_size-1);
            
            % Find similar blocks within the search window
            [similar_blocks, positions] = find_similar_blocks(image, ref_block, i, j, block_size, search_window);
            
            % Stack similar blocks into a 3D array
            stack = stack_blocks(similar_blocks);
            
            % Apply 3D transform (e.g., DCT)
            transformed_stack = dct3(stack);
            
            % Hard thresholding
            transformed_stack(abs(transformed_stack) < threshold) = 0;
            
            % Inverse 3D transform
            denoised_stack = idct3(transformed_stack);
            
            % Aggregate the denoised blocks back into the image
            for k = 1:size(positions, 1)
                pos = positions(k, :);
                basic_estimate(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) = ...
                    basic_estimate(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) + denoised_stack(:, :, k);
                weight(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) = ...
                    weight(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) + 1;
            end
        end
    end
    
    % Normalize the aggregated image
    basic_estimate = basic_estimate ./ weight;
end
%%
%[text] Final Esti
%%
function denoised_image = bm3d_final_estimate(noisy_image, basic_estimate, block_size, search_window)
    [height, width] = size(noisy_image);
    denoised_image = zeros(size(noisy_image));
    weight = zeros(size(noisy_image));
    
    % Loop over each block in the image
    for i = 1:block_size:height-block_size+1
        for j = 1:block_size:width-block_size+1
            % Extract the reference block from the basic estimate
            ref_block = basic_estimate(i:i+block_size-1, j:j+block_size-1);
            
            % Find similar blocks within the search window in the basic estimate
            [similar_blocks, positions] = find_similar_blocks(basic_estimate, ref_block, i, j, block_size, search_window);
            
            % Stack similar blocks into a 3D array
            stack = stack_blocks(similar_blocks);
            
            % Apply 3D transform (e.g., DCT)
            transformed_stack = dct3(stack);
            
            % Wiener filtering
            variance = var(noisy_image(:));
            transformed_stack = transformed_stack .* abs(transformed_stack) ./ (abs(transformed_stack).^2 + variance);
            
            % Inverse 3D transform
            denoised_stack = idct3(transformed_stack);
            
            % Aggregate the denoised blocks back into the image
            for k = 1:size(positions, 1)
                pos = positions(k, :);
                denoised_image(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) = ...
                    denoised_image(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) + denoised_stack(:, :, k);
                weight(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) = ...
                    weight(pos(1):pos(1)+block_size-1, pos(2):pos(2)+block_size-1) + 1;
            end
        end
    end
    
    % Normalize the aggregated image
    denoised_image = denoised_image ./ weight;
end
%%
%[text] **Helper Functions**
%[text] Block Matching and Stacking
%%
function [similar_blocks, positions] = find_similar_blocks(image, ref_block, i, j, block_size, search_window)
    [height, width] = size(image);
    half_window = floor(search_window / 2);
    similar_blocks = [];
    positions = [];
    
    % Define the search window boundaries
    row_min = max(i - half_window, 1);
    row_max = min(i + half_window, height - block_size + 1);
    col_min = max(j - half_window, 1);
    col_max = min(j + half_window, width - block_size + 1);
    
    % Loop through the search window
    for m = row_min:row_max
        for n = col_min:col_max
            % Extract the candidate block
            candidate_block = image(m:m+block_size-1, n:n+block_size-1);
            
            % Calculate the similarity (e.g., Euclidean distance)
            distance = norm(ref_block(:) - candidate_block(:));
            
            % If the candidate block is similar enough, add it to the list
            if distance < threshold
                similar_blocks = cat(3, similar_blocks, candidate_block);
                positions = [positions; m, n];
            end
        end
    end
end

function stack = stack_blocks(similar_blocks)
    % Stack the similar blocks into a 3D array
    stack = reshape(similar_blocks, size(similar_blocks, 1), size(similar_blocks, 2), []);
end
%%
%[text] 3D DCT (Discrete cosine Transform)
%%
function transformed_stack = dct3(stack)
    % Apply 3D DCT to the stack of blocks
    transformed_stack = dct(dct(dct(stack, [], 1), [], 2), [], 3);
end

function inverse_stack = idct3(transformed_stack)
    % Apply inverse 3D DCT to the transformed stack of blocks
    inverse_stack = idct(idct(idct(transformed_stack, [], 1), [], 2), [], 3);
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
%[output:912dc040]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"Operator '<' is not supported for operands of type 'threshold'.\n\nError in <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('bm_3d>find_similar_blocks', '\/home\/ch41c1d\/Documents\/MATLAB\/bm_3d.mlx', 123)\" style=\"font-weight:bold\">bm_3d>find_similar_blocks<\/a> (<a href=\"matlab: opentoline('\/home\/ch41c1d\/Documents\/MATLAB\/bm_3d.mlx',123,0)\">line 123<\/a>)\n            if distance < threshold\n\nError in <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('bm_3d>bm3d_basic_estimate', '\/home\/ch41c1d\/Documents\/MATLAB\/bm_3d.mlx', 32)\" style=\"font-weight:bold\">bm_3d>bm3d_basic_estimate<\/a> (<a href=\"matlab: opentoline('\/home\/ch41c1d\/Documents\/MATLAB\/bm_3d.mlx',32,0)\">line 32<\/a>)\n            [similar_blocks, positions] = find_similar_blocks(image, ref_block, i, j, block_size, search_window);"}}
%---
