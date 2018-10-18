function show_viewpoint(idx, viewpoint, yaw)
% idx = 10;
% viewpoint = 2437;
% yaw = 335;

opt = globals();

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
object_names = C{1};
fclose(fid);
num_objects = numel(object_names);

% open file for saving pose metadata
dir = fullfile(opt.root, 'models', 'rendered_viewpoints', object_names{idx});
filename = fullfile(dir, 'viewpoints.mat');
viewpoints = load(filename);

mask = squeeze(viewpoints.viewpoints.masks(viewpoint,:,:));
mask = imrotate(mask,yaw);

% show the mask
figure(2);
imshow(mask);
axis off;  
axis equal;
pause;