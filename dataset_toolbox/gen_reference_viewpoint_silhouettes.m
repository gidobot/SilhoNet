% seq_id: 0 ~ 91
% The *-meta.mat file in the YCB-Video dataset contains the following fields:
% center: 2D location of the projection of the 3D model origin in the image
% cls_indexes: class labels of the objects
% factor_depth: divde the depth image by this factor to get the actual
% depth value
% poses: 6D poses of objects in the image
% intrinsic_matrix: camera intrinsics
% rotation_translation_matrix: RT of the camera motion in 3D
% vertmap: coordinates in the 3D model space of each pixel in the image
function gen_reference_pose_silhouettes
% Generates set of discreete reference viewpoint silhouettes for methods
% that predict viewpoint and viewpoint rotation.

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

parpool('AttachedFiles',{'mesh_test.mexa64'});
parfor (idx = 1:num_objects)
    gen_model_silhouettes(idx,opt,models,object_names,symmetry,rot_offsets);
end
end

function gen_model_silhouettes(idx,opt,models,object_names,symmetry,rot_offsets)
%     intrinsic_matrix_color = opt.intrinsic_matrix_color_cmu;    
    intrinsic_matrix_color = opt.intrinsic_matrix_color;
    
    % image size
    h = 480;
    w = 640;
    crop_size = 64;
    
    % open file for saving pose metadata
    dir = fullfile(opt.root, 'models', 'rendered_viewpoints', object_names{idx});
    if ~exist(dir, 'dir')
        mkdir(dir);
    end
    filename = fullfile(dir, 'viewpoints.mat');
    
    % 0 = no symmetry - angle ranges: roll = (-179.5,179.5), pitch = (-89.5,89.5)
    % 1 = planar symmetry - angle ranges: roll = (0.5,179.5), pitch = (-89.5,89.5)
    % 2 = 2 x planar symmetry - angle ranges: roll = (0.5,89.5), pitch = (-89.5,89.5)
    % 3 = infinite symmetry - angle ranges: roll = 0, pitch = (-89.5,89.5)
    % 4 = infinite symmetry + planar symmetry: roll = 0, pitch = (0.5,89.5)
    step = 5;
    if symmetry(idx) == 0
        angles_roll = -179.5:step:179.5;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(idx) == 1
        angles_roll = 0.5:step:179.5;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(idx) == 2
        angles_roll = 0.5:step:89.5;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(idx) == 3
        angles_roll = 0;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(idx) == 4
        angles_roll = 0;
        angles_pitch = 0.5:step:89.5;
    end
        
    num_angles = length(angles_roll)*length(angles_pitch);
    viewpoints.poses = zeros(num_angles,3);
    viewpoints.masks = zeros(num_angles,crop_size,crop_size);
    viewpoints.symmetry = symmetry;
    viewpoints.quaternion = zeros(num_angles,4);
    counter = 0;
    for i = 1:length(angles_roll)
        for j = 1:length(angles_pitch)
            counter = counter+1;
            % set RT_o2c
            RT_o2c = zeros(3,4);
            RT_o2c(:,1:3) = eye(3);
            
            R_adj = SpinCalc('EA123toDCM',rot_offsets(:,idx)',.0001,0);
            
            % set viewpoint angle
            roll = angles_roll(i);
            pitch = angles_pitch(j);
            yaw = 0;
            eul = [roll, pitch, yaw];
            
            viewpoints.poses(counter,:) = eul;
            
            rot = SpinCalc('EA123toDCM',eul,.0001,0);
            RT_o2c(:,1:3) = rot*R_adj*RT_o2c(:,1:3);
            
            quat = SpinCalc('DCMtoQ',rot,.0001,0);
            quat([1,2,3,4]) = quat([4,1,2,3]);
            viewpoints.quaternion(counter,:) = quat;
            
            [roll, pitch, yaw]

            x3d = models{idx}.v';

            % projection crop
            scale = 1.05;
            obj_w = abs(max(x3d(:,1))-min(x3d(:,1)));
            obj_d = abs(max(x3d(:,2))-min(x3d(:,2)));
            obj_h = abs(max(x3d(:,3))-min(x3d(:,3)));
            dim = sqrt(obj_w^2 + obj_h^2 + obj_d^2);
            % assume height is minimum dimension
            min_dim = min([w,h]);
            fov = atan(min_dim/intrinsic_matrix_color(1,1));
            dist = dim/(2*tan(fov/2))*scale;
            RT_o2c(3,4) = dist;

            % projection
            x2d = project(x3d, intrinsic_matrix_color, RT_o2c);

            face = models{idx}.f3';

            vertices = [x2d(face(:,1),2) x2d(face(:,1),1) ...
                        x2d(face(:,2),2) x2d(face(:,2),1) ...
                        x2d(face(:,3),2) x2d(face(:,3),1)];

            % BW is the mask
            BW = mesh_test(vertices, h, w);
            tmp = zeros(h,w);
            tmp(BW) = 1;
            dim_diff = w-h;
            w_min = floor(dim_diff/2);
            w_max = w_min + h;
            tmp = tmp(:,w_min:w_max);
            tmp = imresize(tmp,[crop_size,crop_size],'nearest');
            
            viewpoints.masks(counter,:,:) = tmp;

%             % show the mask
%             imshow(tmp);
%             axis off;  
%             axis equal;
% 
%             pause;
        end
    end
    save(filename,'viewpoints');
end