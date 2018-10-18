function plot_accuracy_keyframe
% Plots SilhoNet results agains PoseCNN for tests on the keyframe image
% set.

color = {'r', '-.c', 'g', 'y', 'm', '--b', 'g', '-.m'};
leng = {'PoseCNN', 'PoseCNN+ICP', 'PoseCNN+Multiview', 'PoseCNN+ICP+Multiview', ...
    '3D Coordinate Regression', 'PoseCNN-SilhoNet', 'SilhoNet-YCB', 'SilhoNet-FasterRCNN'};
aps = zeros(5, 1);
lengs = cell(5, 1);
close all;

% load results
object = load('results_keyframe.mat');
distances_sys = object.distances_sys;
distances_non = object.distances_non;
rotations = object.errors_rotation;
translations = object.errors_translation;
cls_ids = object.results_cls_id;

object_silhonet_gt = load('results_SilhoNet/angle_errors_gt.mat');
object_silhonet_pred = load('results_SilhoNet/angle_errors_pred.mat');
silhonet_names_gt = fieldnames(object_silhonet_gt);
silhonet_names_pred = fieldnames(object_silhonet_pred);

% index_plot = [4, 2, 5, 3, 1];
index_plot = [1, 2];

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
classes = C{1};
classes{end+1} = 'All 21 objects';
fclose(fid);

hf = figure('units','normalized','outerposition',[0 0 1 1]);
font_size = 30;
max_distance = 0.1;

silhonet_all_classes_gt = [];
silhonet_all_classes_pred = [];
mean_errors = zeros(numel(classes), numel(leng));
std_dev_errors = zeros(numel(classes), numel(leng)); 

% for each class
for k = 1:numel(classes)
    index = find(cls_ids == k);
    disp([num2str(k),': ',num2str(length(index))]);
    if isempty(index)
        index = 1:size(distances_sys,1);
    end
    
    % rotation
%     subplot(2, 1, 1);
    for i = index_plot
        D = rotations(index, i);
        inf_ind=D==Inf;
        D = D(inf_ind==0);
        mean_errors(k,i) = mean(D);
        std_dev_errors(k,i) = std2(D);
        d = sort(D);
        n = numel(d);
        accuracy = cumsum(ones(1, n)) / n;
        plot(d, accuracy, color{i}, 'LineWidth', 8);
        hold on;
    end
    % gt rois
    if k == length(classes)
        D_gt = silhonet_all_classes_gt;
        D_pred = silhonet_all_classes_pred;
    else
        D_gt = cell2mat(object_silhonet_gt.(classes{k}(5:end))');
        D_pred = cell2mat(object_silhonet_pred.(classes{k}(5:end))');
        silhonet_all_classes_gt = [silhonet_all_classes_gt; D_gt];
        silhonet_all_classes_pred = [silhonet_all_classes_pred; D_pred];
    end
    mean_errors(k,7) = mean(D_gt);
    std_dev_errors(k,7) = std2(D_gt);
    d = sort(D_gt);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) /n;
    plot(d, accuracy, color{7}, 'LineWidth', 8);
    mean_errors(k,8) = mean(D_pred);
    std_dev_errors(k,8) = std2(D_pred);
    d = sort(D_pred);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) /n;
    plot(d, accuracy, color{8}, 'LineWidth', 8);
    hold off;
    %h = legend('network', 'refine tranlation only', 'icp', 'stereo translation only', 'stereo full', '3d coordinate');
    %set(h, 'FontSize', 16);
    h = legend(leng([index_plot, 7, 8]), 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Rotation angle threshold');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(classes{k}, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)
    set(gcf,'units','centimeters','position',[0,0,30,20])
    xlim([0 180])
    ylim([0 1])
    xticks([0,40,80,120,160])
    
    % rotation silhonet
    

%     % translation
%     subplot(2, 2, 4);
%     for i = index_plot
%         D = translations(index, i);
%         D(D > max_distance) = inf;
%         d = sort(D);
%         n = numel(d);
%         accuracy = cumsum(ones(1, n)) / n;
%         plot(d, accuracy, color{i}, 'LineWidth', 4);
%         hold on;
%     end
%     hold off;
%     h = legend(leng(index_plot), 'Location', 'southeast');
%     set(h, 'FontSize', font_size);
%     h = xlabel('Translation threshold in meter');
%     set(h, 'FontSize', font_size);
%     h = ylabel('accuracy');
%     set(h, 'FontSize', font_size);
%     h = title(classes{k}, 'Interpreter', 'none');
%     set(h, 'FontSize', font_size);
%     xt = get(gca, 'XTick');
%     set(gca, 'FontSize', font_size)
    
    filename = sprintf('plots/%s.png', classes{k});
    hgexport(hf, filename, hgexport('factorystyle'), 'Format', 'png');
    
%     pause;
end

classes
mean_errors
% std_dev_errors

function ap = VOCap(rec, prec)

index = isfinite(rec);
rec = rec(index);
prec = prec(index)';

mrec=[0 ; rec ; 0.1];
mpre=[0 ; prec ; prec(end)];
for i = 2:numel(mpre)
    mpre(i) = max(mpre(i), mpre(i-1));
end
i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) * 10;
    