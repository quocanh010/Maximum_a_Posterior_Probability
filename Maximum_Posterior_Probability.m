S = load('TrainingSamplesDCT_subsets_8.mat');
Prior_1 = load('Prior_1.mat');
Prior_2 = load('Prior_2.mat');
alpha_m = load('Alpha.mat');
alpha = alpha_m.alpha;
%Extract data
% D1_BG = S.D1_BG;
% D2_BG = S.D2_BG;
% D3_BG = S.D3_BG;
% D4_BG = S.D4_BG;
% 
% D1_CT = S.D1_FG;
% D2_CT = S.D2_FG;
% D3_CT = S.D3_FG;
% D4_CT = S.D4_FG;

data_cell = cell(2,4);

data_cell{1,1} = S.D1_BG;
data_cell{1,2} = S.D2_BG;
data_cell{1,3} = S.D3_BG;
data_cell{1,4} = S.D4_BG;

data_cell{2,1} = S.D1_FG;
data_cell{2,2} = S.D2_FG;
data_cell{2,3} = S.D3_FG;
data_cell{2,4} = S.D4_FG;


Prior_1_W0 = Prior_1.W0;
Prior_1_m0_BG = Prior_1.mu0_BG;
Prior_1_m0_CT = Prior_1.mu0_FG;

Prior_2_W0 = Prior_2.W0;
Prior_2_m0_BG = Prior_2.mu0_BG;
Prior_2_m0_CT = Prior_2.mu0_FG;






img = imread('cheetah.bmp');
img = im2double(img);
%Vectorize 8-8 block with zig-zag patter
% Padding to the image
B = padarray(img, [7 7], 'symmetric','pre');
m_Cheetah = zeros(255, 270);
m_Background = zeros(255, 270);
% Compute ans sliding window and decision matrix for 64 features
B_masking_m_Cheetah = zeros(255, 270);
B_masking_m_Background = zeros( 255, 270);
ML_masking_m_Cheetah = zeros( 255, 270);
ML_masking_m_Background = zeros( 255, 270);
MAP_masking_m_Cheetah = zeros( 255, 270);
MAP_masking_m_Background = zeros( 255, 270);

B_e = zeros(9,1);
MAP_e =  zeros(9,1);
ML_e = zeros(9,1);

masked_cheetah = imread('cheetah_mask.bmp');
masked_cheetah = im2double(masked_cheetah);
v = find(masked_cheetah);
v1 = find(~masked_cheetah);

% for al = 1:8
%      sigma0 = diag(alpha(al) .* Prior_1_W0);
%      mun_1_BG = sigma0 * inv(sigma0 + (1/(size(D1_BG, 1))) *  cov_D1_BG)  * me_1_BG' + (1/size(D1_BG, 1)) * cov_D1_BG * inv(sigma0 + (1/size(D1_BG, 1)) * cov_D1_BG) * Prior_1_m0_BG';
%      mun_1_CT = sigma0 * inv(sigma0 + (1/(size(D1_CT,1))) *  cov_D1_CT)  * me_1_CT' + (1/size(D1_CT,1)) * cov_D1_CT * inv(sigma0 + (1/(size(D1_CT,1))) * cov_D1_CT) * Prior_1_m0_CT';
%      cov_n_1_BG = sigma0 * inv(sigma0 + (1/(size(D1_BG, 1))) * cov_D1_BG) * (1/(size(D1_BG, 1))) * cov_D1_BG;
%      cov_n_1_CT = sigma0 * inv(sigma0 + (1/(size(D1_CT, 1))) * cov_D1_CT) * (1/(size(D1_CT, 1))) * cov_D1_CT;
%      B_cov_BG = cov_n_1_BG + cov_D1_BG;
%      B_cov_CT = cov_n_1_CT + cov_D1_CT;
%      for i = 1:size(img, 1)
%          for j = 1:size(img, 2)
%              temp_v = compute_dct_vector(B(i:i+7, j:j+7));
%              temp_v = temp_v';
%              B_masking_m_Cheetah(i, j) = (temp_v - mun_1_CT)' * inv(B_cov_CT) * (temp_v - mun_1_CT) + (log((2 * pi)^64 * det(B_cov_CT)) + Prior_Cheetah_1);
%              B_masking_m_Background(i, j) = (temp_v - mun_1_BG)' * inv(B_cov_BG) * (temp_v - mun_1_BG) + (log((2 * pi)^64 * det(B_cov_BG)) + Prior_Background_1);
%              MAP_masking_m_Cheetah(i, j) = (temp_v - mun_1_CT)' * inv(cov_D1_CT) * (temp_v - mun_1_CT) + (log((2 * pi)^64 * det(cov_D1_CT)) + Prior_Cheetah_1);
%              MAP_masking_m_Background(i, j) = (temp_v - mun_1_BG)' * inv(cov_D1_BG) * (temp_v - mun_1_BG) + (log((2 * pi)^64 * det(cov_D1_BG)) + Prior_Background_1);
%              ML_masking_m_Cheetah(i, j) = (temp_v - me_1_CT')' * inv(cov_D1_CT) * (temp_v - me_1_CT') + (log((2 * pi)^64 * det(cov_D1_CT)) + Prior_Cheetah_1);
%              ML_masking_m_Background(i, j) = (temp_v - me_1_BG')' * inv(cov_D1_BG) * (temp_v - me_1_BG') + (log((2 * pi)^64 * det(cov_D1_BG)) + Prior_Background_1);
%          end
%      end
%      
%      n_img = ~(  B_masking_m_Background <= B_masking_m_Cheetah );
%      %Calcualte error for 64 features
%      flat_v_64 = n_img(:);
%  
%      %Calculating error
%      P_Cheetah_g_Cheetah = nnz(flat_v_64(v) == 1) / numel(flat_v_64(v));
%      P_Cheetah_g_Backgound = nnz(flat_v_64(v1) == 1) / numel(flat_v_64(v1));
%      B_e(al,1) = P_Cheetah_g_Backgound * Prior_Background_1 + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah_1;
%      
%      n_img = ~(   MAP_masking_m_Background <= MAP_masking_m_Cheetah );
%      %Calcualte error for 64 features
%      flat_v_64 = n_img(:);
%  
%      %Calculating error
%      P_Cheetah_g_Cheetah = nnz(flat_v_64(v) == 1) / numel(flat_v_64(v));
%      P_Cheetah_g_Backgound = nnz(flat_v_64(v1) == 1) / numel(flat_v_64(v1));
%      MAP_e(al,1) = P_Cheetah_g_Backgound * Prior_Background_1 + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah_1;
%      
%      n_img = ~(   ML_masking_m_Background <= ML_masking_m_Cheetah);
%      %Calcualte error for 64 features
%      flat_v_64 = n_img(:);
%  
%      %Calculating error
%      P_Cheetah_g_Cheetah = nnz(flat_v_64(v) == 1) / numel(flat_v_64(v));
%      P_Cheetah_g_Backgound = nnz(flat_v_64(v1) == 1) / numel(flat_v_64(v1));
%      ML_e(al,1) = P_Cheetah_g_Backgound * Prior_Background_1 + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah_1;
%      
% 
% 
%      
%      
% end
% % 64 features Gaussian Classifier
for d = 1:4
    cov_BG = cov(data_cell{1,d});
    cov_CT = cov(data_cell{2,d});
    me_BG = mean(data_cell{1,d},1);
    me_CT = mean(data_cell{2,d},1);
    Prior_Cheetah = size(data_cell{2,d}, 1) / (size(data_cell{1,d},1) + size(data_cell{2,d}, 1));
    Prior_Background = size(data_cell{1,d}, 1) / (size(data_cell{1,d},1) + size(data_cell{2,d}, 1));

     for al = 1:9
         sigma0 = diag(alpha(al) .* Prior_1_W0);
         mun_1_BG = sigma0 * ((sigma0 + (1/(size(data_cell{1,d}, 1))) *  cov_BG)  \ me_BG') + (1/size(data_cell{1,d}, 1)) * cov_BG * ((sigma0 + (1/size(data_cell{1,d}, 1)) * cov_BG) \ Prior_1_m0_BG');
         mun_1_CT = sigma0 * ((sigma0 + (1/(size(data_cell{2,d},1))) *  cov_CT)  \ me_CT') + (1/size(data_cell{2,d},1)) * cov_CT * ((sigma0 + (1/(size(data_cell{2,d},1))) * cov_CT) \ Prior_1_m0_CT');
         cov_n_1_BG = sigma0 * ((sigma0 + (1/(size(data_cell{1,d}, 1))) * cov_BG) \ ((1/(size(data_cell{1,d}, 1))) * cov_BG));
         cov_n_1_CT = sigma0 * ((sigma0 + (1/(size(data_cell{2,d}, 1))) * cov_CT) \ ((1/(size(data_cell{2,d}, 1))) * cov_CT));
         B_cov_BG = cov_n_1_BG + cov_BG;
         B_cov_CT = cov_n_1_CT + cov_CT;
         for i = 1:size(img, 1)
             for j = 1:size(img, 2)
                 temp_v = compute_dct_vector(B(i:i+7, j:j+7));
                 temp_v = temp_v';
                 B_masking_m_Cheetah(i, j) = (temp_v - mun_1_CT)' * ((B_cov_CT) \ (temp_v - mun_1_CT)) + (log((2 * pi)^64 * det(B_cov_CT)) + Prior_Cheetah);
                 B_masking_m_Background(i, j) = (temp_v - mun_1_BG)' * ((B_cov_BG) \ (temp_v - mun_1_BG)) + (log((2 * pi)^64 * det(B_cov_BG)) + Prior_Background);
                 MAP_masking_m_Cheetah(i, j) = (temp_v - mun_1_CT)' * ((cov_CT) \ (temp_v - mun_1_CT)) + (log((2 * pi)^64 * det(cov_CT)) + Prior_Cheetah);
                 MAP_masking_m_Background(i, j) = (temp_v - mun_1_BG)' * ((cov_BG) \ (temp_v - mun_1_BG)) + (log((2 * pi)^64 * det(cov_BG)) + Prior_Background);
                 ML_masking_m_Cheetah(i, j) = (temp_v - me_CT')' * ((cov_CT) \ (temp_v - me_CT')) + (log((2 * pi)^64 * det(cov_CT)) + Prior_Cheetah);
                 ML_masking_m_Background(i, j) = (temp_v - me_BG')' * ((cov_BG) \ (temp_v - me_BG')) + (log((2 * pi)^64 * det(cov_BG)) + Prior_Background);
             end
         end

         n_img = ~(  B_masking_m_Background <= B_masking_m_Cheetah );
         %Calcualte error for 64 features
         flat_v_64 = n_img(:);

         %Calculating error
         P_Cheetah_g_Cheetah = nnz(flat_v_64(v) == 1) / numel(flat_v_64(v));
         P_Cheetah_g_Backgound = nnz(flat_v_64(v1) == 1) / numel(flat_v_64(v1));
         B_e(al,1) = P_Cheetah_g_Backgound * Prior_Background + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah;

         m_img = ~(   MAP_masking_m_Background <= MAP_masking_m_Cheetah );
         %Calcualte error for 64 features
         flat_v_64 = m_img(:);

         %Calculating error
         P_Cheetah_g_Cheetah = nnz(flat_v_64(v) == 1) / numel(flat_v_64(v));
         P_Cheetah_g_Backgound = nnz(flat_v_64(v1) == 1) / numel(flat_v_64(v1));
         MAP_e(al,1) = P_Cheetah_g_Backgound * Prior_Background + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah;

         d_img = ~(   ML_masking_m_Background <= ML_masking_m_Cheetah);
         %Calcualte error for 64 features
         flat_v_64 = d_img(:);

         %Calculating error
         P_Cheetah_g_Cheetah = nnz(flat_v_64(v) == 1) / numel(flat_v_64(v));
         P_Cheetah_g_Backgound = nnz(flat_v_64(v1) == 1) / numel(flat_v_64(v1));
         ML_e(al,1) = P_Cheetah_g_Backgound * Prior_Background + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah;

     end
figure;
plot(log10(alpha), B_e','o-','linewidth',3,'markersize',10,'markerfacecolor','g')
hold on;
plot(log10(alpha), MAP_e','o-','linewidth',3,'markersize',10,'markerfacecolor','g')
plot(log10(alpha), ML_e','o-','linewidth',3,'markersize',10,'markerfacecolor','g')
legend('Bayes','MAP','ML');
xlabel('log10(alpha)') ;
ylabel('Probability of error') ;
title(['D',num2str(d)]);
hold off;


end
% 64 features Gaussian Clas
 
 

 


% % Calculating P_Y_g_X
% img = imread('cheetah.bmp');
% % Normalizing Image
% img = im2double(img);
% 
% Vectorize 8-8 block with zig-zag patter
% % Padding to the image
% B = padarray(img, [7 7], 'symmetric','pre');
% result_v = zeros(255, 270, 64);
% m_Cheetah = zeros(255, 270);
% m_Background = zeros(255, 270);
% % Compute ans sliding window and decision matrix for 64 features
%  for i = 1:size(img, 1)
%      for j = 1:size(img, 2)
%          temp_v = compute_dct_vector(B(i:i+7, j:j+7));
%          temp_v = temp_v';
%          m_Cheetah(i,j) = (temp_v - mean_Cheetah')' * inv(cov_Cheetah) * (temp_v - mean_Cheetah') + (log((2 * pi)^64 * det(cov_Cheetah)) + Prior_Cheetah);
%          m_Background(i,j) = (temp_v - mean_Background')' * inv(cov_Background) * (temp_v - mean_Background') + (log((2 * pi)^64 * det(cov_Background)) + Prior_Background);
%      end
%  end
% 
% % 64 features Gaussian Classifier
% n_img = ~(m_Background <= m_Cheetah);
% figure
% subplot(1, 2, 1)
% imagesc(n_img);
% colormap(gray(256));
% title(['64 features masked '])
% axis equal;





% Cheetah = S.TrainsampleDCT_FG;
% Background = S.TrainsampleDCT_BG;

% Calculating Piror Problability 
% Prior_Cheetah = size(Cheetah, 1) / (size(Cheetah,1) + size(Background, 1));
% Prior_Background = size(Background, 1) / (size(Cheetah,1) + size(Background, 1));
% 
% Calculate the mean for each feature
% mean_Cheetah    = mean(Cheetah, 1);
% mean_Background = mean(Background, 1);
% 
% Find covariance matrix for each class
% cov_Cheetah = cov(Cheetah);
% cov_Background = cov(Background);
% 
% Graph marginal distributions
% 
% for i = 1:1:64
%     mu_c = mean_Cheetah(i);
%     sigma_c = sqrt(cov_Cheetah(i,i));
%     mu_b = mean_Background(i);
%     sigma_b = sqrt(cov_Background(i,i));
%     x_c = linspace(min(Cheetah(:,i)), max(Cheetah(:,i)), 1303);
%     x_b = linspace(min(Background(:,i)), max(Background(:,i)), 1303);
%     x_c = linspace(min(mean_Cheetah(:,i)), max(mean_Cheetah(:, i)), 64);
%     x_b = linspace(min(mean_Background(:,i)), max(mean_Background(:, i)), 64);
%     if (mod((i-1), 8) == 0)
%         figure
%         count = 1;
%        
%     end
%     subplot(2, 4, count)
%     plot(x_c, pdf('Normal', x_c, mu_c, sigma_c))
%     plot(x_c, normpdf(x_c, mu_c, sigma_c))
%     hold on
%     plot(x_b, pdf('Normal', x_b, mu_b, sigma_b))
%     plot(x_b, normpdf(x_b, mu_b, sigma_b))
%     title(['Feature: ', num2str(i)])
%     hold off
%     count = count + 1;
%     
% end
% Pick 8 best and worst plot them
% best_8 = [1 11 14 23 25 27 32 40]
% worst_8 = [3 4 5 59 60 62 63 64]
% 8 best
% count = 1;
% figure
% for i = 1:numel(best_8)
%     mu_c = mean_Cheetah(best_8(i));
%     sigma_c = sqrt(cov_Cheetah(best_8(i), best_8(i)));
%     mu_b = mean_Background(best_8(i));
%     sigma_b = sqrt(cov_Background(best_8(i), best_8(i)));
%     x_c = linspace(min(Cheetah(:, best_8(i))), max(Cheetah(:,best_8(i))), 1303);
%     x_b = linspace(min(Background(:, best_8(i))), max(Background(:, best_8(i))), 1303);
%     subplot(2, 4, count)
%     subplot(2, 4, count)
%     plot(x_c, pdf('Normal', x_c, mu_c, sigma_c))
%     plot(x_c, normpdf(x_c, mu_c, sigma_c))
%     hold on
%     plot(x_b, pdf('Normal', x_b, mu_b, sigma_b))
%     plot(x_b, normpdf(x_b, mu_b, sigma_b))
%     title(['Feature: ', num2str(best_8(i))])
%     hold off
%     count = count + 1;
%     
% end
% 8 worst
% count = 1;
% figure
% for i = 1:numel(worst_8)
%     mu_c = mean_Cheetah(worst_8(i));
%     sigma_c = sqrt(cov_Cheetah(worst_8(i), worst_8(i)));
%     mu_b = mean_Background(worst_8(i));
%     sigma_b = sqrt(cov_Background(worst_8(i), worst_8(i)));
%     x_c = linspace(min(Cheetah(:, worst_8(i))), max(Cheetah(:, worst_8(i))), 1303);
%     x_b = linspace(min(Background(:, worst_8(i))), max(Background(:, worst_8(i))), 1303);
%     subplot(2, 4, count)
%     subplot(2, 4, count)
%     plot(x_c, pdf('Normal', x_c, mu_c, sigma_c))
%     plot(x_c, normpdf(x_c, mu_c, sigma_c))
%     hold on
%     plot(x_b, pdf('Normal', x_b, mu_b, sigma_b))
%     plot(x_b, normpdf(x_b, mu_b, sigma_b))
%     title(['Feature: ', num2str(worst_8(i))])
%     hold off
%     count = count + 1;
%     
% end
% 
% 
% Calculating P_Y_g_X
% img = imread('cheetah.bmp');
% Normalizing Image
% img = im2double(img);
% 
% Vectorize 8-8 block with zig-zag patter
% Padding to teh image
% B = padarray(img, [7 7], 'symmetric','pre');
% result_v = zeros(255, 270, 64);
% m_Cheetah = zeros(255, 270);
% m_Background = zeros(255, 270);
% m_8_Cheetah = zeros(255, 270);
% m_8_Background = zeros(255, 270);
% Compute ans sliding window and decision matrix for 64 features
% Pick 8 best features 1,11,14,23,25,27,32,40
% Find Covariance Matrixs and mean vectors for 8 best features
% cov_8_Cheetah = cov_Cheetah([1 11 14 23 25 27 32 40],[1 11 14 23 25 27 32 40]);
% cov_8_Background = cov_Background([1 11 14 23 25 27 32 40],[1 11 14 23 25 27 32 40]);
% mean_8_Cheetah = [mean_Cheetah(1); mean_Cheetah(11); mean_Cheetah(14); mean_Cheetah(23); mean_Cheetah(25); mean_Cheetah(27); mean_Cheetah(32); mean_Cheetah(40)];
% mean_8_Background = [mean_Background(1); mean_Background(11); mean_Background(14); mean_Background(23); mean_Background(25); mean_Background(27); mean_Background(32); mean_Background(40)];
%  for i = 1:size(img, 1)
%      for j = 1:size(img, 2)
%          temp_v = compute_dct_vector(B(i:i+7, j:j+7));
%          temp_v = temp_v';
%          temp_8 = [temp_v(1); temp_v(11); temp_v(14); temp_v(23); temp_v(25); temp_v(27); temp_v(32); temp_v(40)];
%          m_Cheetah(i,j) = (temp_v - mean_Cheetah')' * inv(cov_Cheetah) * (temp_v - mean_Cheetah') + (log((2 * pi)^64 * det(cov_Cheetah)) + Prior_Cheetah);
%          m_Background(i,j) = (temp_v - mean_Background')' * inv(cov_Background) * (temp_v - mean_Background') + (log((2 * pi)^64 * det(cov_Background)) + Prior_Background);
%          m_8_Cheetah(i,j) = (temp_8 - mean_8_Cheetah)' * inv(cov_8_Cheetah) * (temp_8 - mean_8_Cheetah) + (log((2 * pi)^64 * det(cov_8_Cheetah)) + Prior_Cheetah);
%          m_8_Background(i,j) = (temp_8 - mean_8_Background)' * inv(cov_8_Background) * (temp_8 - mean_8_Background) + (log((2 * pi)^64 * det(cov_8_Background)) + Prior_Background);
%      end
%  end
% result = nlfilter(  B , [8 8], @zig_zag_v);
% 64 features Gaussian Classifier
% n_img = ~(m_Background <= m_Cheetah);
% figure
% subplot(1, 2, 1)
% imagesc(n_img);
% colormap(gray(256));
% title(['64 features masked '])
% axis equal;
% 
% n_8_img = ~(m_8_Background <= m_8_Cheetah);
% hold on
% subplot(1, 2, 2)
% imagesc(n_8_img);
% colormap(gray(256));
% title(['8 features masked '])
% axis equal;
% hold off
% masked_cheetah = imread('cheetah_mask.bmp');
% masked_cheetah = im2double(masked_cheetah);
% v = find(masked_cheetah);
% v1 = find(~masked_cheetah);
% 
% Calcualte error for 64 features
% flat_v_64 = n_img(:);
% flat_v_8  = n_8_img(:);
% Calculating error
% P_Cheetah_g_Cheetah = nnz(flat_v(v) == 1) / numel(flat_v_64(v));
% P_Cheetah_g_Backgound = nnz(flat_v(v1) == 1) / numel(flat_v_64(v1));
% 
% P_e_64 = P_Cheetah_g_Backgound * Prior_Background + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah;
% 
% Calculating error for 8 features
% P_Cheetah_g_Cheetah = nnz(flat_v_8(v) == 1) / numel(flat_v_8(v));
% P_Cheetah_g_Backgound = nnz(flat_v_8(v1) == 1) / numel(flat_v_8(v1));
% 
% P_e_8 = P_Cheetah_g_Backgound * Prior_Background + (1 - P_Cheetah_g_Cheetah) * Prior_Cheetah;