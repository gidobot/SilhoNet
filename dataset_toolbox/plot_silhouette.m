% RT: Object 6D pose as a transformation matrix.
% class_idx: Integer id of object class
function plot_silhouette(RT, class_idx)

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
pitch_angle = atan2d(RT(1,4),RT(3,4));
roll_angle = -atan2d(RT(2,4),RT(3,4));
eul = [roll_angle, pitch_angle, 0];
rot = SpinCalc('EA123toDCM',eul,.0001,1);
RT_tmp = zeros(3,4);
RT_tmp(:,1:3) = rot*RT(:,1:3);

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
x2d = project(x3d, intrinsic_matrix_color, RT_tmp);

face = models{class_idx}.f3';

vertices = [x2d(face(:,1),2) x2d(face(:,1),1) ...
            x2d(face(:,2),2) x2d(face(:,2),1) ...
            x2d(face(:,3),2) x2d(face(:,3),1)];
% BW is the mask
BW = mesh_test(double(vertices), h, w);
tmp = zeros(h,w);
tmp(BW) = 1;
dim_diff = w-h;
w_min = floor(dim_diff/2);
w_max = w_min + h;
tmp = tmp(:,w_min:w_max);
tmp = imresize(tmp,[crop_size,crop_size],'nearest');
imshow(tmp);