% mask: 64x64 occluded silhouette prediction from SilhoNet
% quat: Quaternion prediction of 3D pose from SilhoNet
% translation: 3D translation of object for visualization purposes (arbitrary for grasp point filtering)
% grasp_points: List of x,y,z grasp point coordinates in object frame
%   formatted as [x1,y1,z1; x2,y2,z2; ...]
% class_idx: Integer id of object class
% image_path: Path to image input to SilhoNet for visualization of grasp
%   points
function filter_grasps_by_silhouette(mask, quat, translation, grasp_points, class_idx, image_path)
% Filters pre-computed grasp points of an object based on the predicted
% object occlusion mask output by SihloNet. Displays the grasp points
% projected back into the input image.

% 0 = no symmetry - angle ranges: roll = (-179.5,179.5), pitch = (-89.5,89.5)
% 1 = planar symmetry - angle ranges: roll = (0.5,179.5), pitch = (-89.5,89.5)
% 2 = 2 x planar symmetry - angle ranges: roll = (0.5,89.5), pitch = (-89.5,89.5)
% 3 = infinite symmetry - angle ranges: roll = 0, pitch = (-89.5,89.5)
% 4 = infinite symmetry + planar symmetry: roll = 0, pitch = (0.5,89.5)
symmetry    = [ 4, 2, 2,  4,   1,  4,  2,  2, 1, 0, 0, 0,  3, 0, 0,    2, 0,  3,  1,   1, 2];
rot_offsets = [90, 0, 0, 90,   0, 90,  0,  0, 0, 0, 0, 0, 90, 0, 0,    0, 0,  0, 94,  90, 0;
                0, 0, 0,  0,   0,  0,  0,  0, 0, 0, 0, 0,  0, 0, 0,    0, 0,  0,  9, -84, 0;
               90, 0, 0, 90, -22, 90, 28, 13, 4, 0, 0, 0, 90, 0, 0,  -12, 0, 92, -5,  -1, 0];

opt = globals();

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
object_names = C{1};
fclose(fid);
num_objects = numel(object_names);

% load CAD models
disp('loading 3D models...');
models = cell(num_objects, 1);
for i = 1:num_objects
    filename = sprintf('models/%s.mat', object_names{i});
    if exist(filename, 'file')
        object = load(filename);
        obj = object.obj;
    else
        file_obj = fullfile(opt.root, 'models', object_names{i}, 'textured.obj');
        obj = load_obj_file(file_obj);
        save(filename, 'obj');
    end
    disp(filename);
    models{i} = obj;
end

intrinsic_matrix_color = opt.intrinsic_matrix_color;
w = 640;
h = 480;
crop_size = 64;
% use translation to get apparant orientation of object in the
% camera view.
R_adj = SpinCalc('EA123toDCM',rot_offsets(:,class_idx)',.0001,0);
RT_tmp = zeros(3,4);

quat_tmp([1,2,3,4]) = rad2deg(quat([2,3,4,1]));
RT_tmp(:,1:3) = SpinCalc('QtoDCM',quat_tmp,.0001,0)*R_adj;

%% generate full silhouettes
% projection
x3d = models{class_idx}.v';
scale = 1.05;
obj_w = abs(max(x3d(:,1))-min(x3d(:,1)));
obj_d = abs(max(x3d(:,2))-min(x3d(:,2)));
obj_h = abs(max(x3d(:,3))-min(x3d(:,3)));
dim = sqrt(obj_w^2 + obj_h^2 + obj_d^2);
min_dim = min([w,h]);
fov = atan(min_dim/intrinsic_matrix_color(1,1));
dist = dim/(2*tan(fov/2))*scale;
RT_tmp(3,4) = dist;

x2d_grasp_points = project(grasp_points, intrinsic_matrix_color, RT_tmp);
RT_tmp(:,4) = translation;
x2d_image_points = project(grasp_points, intrinsic_matrix_color, RT_tmp);

tmp = zeros(h,w);
dim_diff = w-h;
w_min = floor(dim_diff/2);
w_max = w_min + h;
tmp = tmp(:,w_min:w_max);
tmp = imresize(tmp,[crop_size,crop_size],'nearest');

tmp = zeros(h,w);
tmp(:,w_min:w_max) = imresize(mask,[h,h+1],'nearest');
hold off;
figure(1);
imshow(tmp);
hold on;
valid_list = zeros(1, size(x2d_grasp_points,1));
for i = 1:size(x2d_grasp_points,1)
    x = x2d_grasp_points(i,1);
    y = x2d_grasp_points(i,2);
    if tmp(floor(y),floor(x)) == 1
        valid_list(i) = 1;
        c = 'g.';
    else
        valid_list(i) = 0;
        c = 'r.';
    end
    plot(x2d_grasp_points(i,1), x2d_grasp_points(i,2), c, 'LineWidth', 2, 'MarkerSize', 25);
end
figure(2);
I = imread(image_path);
hold off;
imshow(I);
hold on;
for i = 1:size(x2d_grasp_points,1)
    if valid_list(i)
        c = 'g.';
    else
        c = 'r.';
    end
    plot(x2d_image_points(i,1), x2d_image_points(i,2), c, 'LineWidth', 2, 'MarkerSize', 25);
end
