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
function gen_full_silhouettes_synthetic()
% Output same as gen_full_silhouettes(), but for synthetic images in the
% YCB-Video dataset.

% 0 = no symmetry - angle ranges: roll = (-179.5,179.5), pitch = (-89.5,89.5)
% 1 = planar symmetry - angle ranges: roll = (0.5,179.5), pitch = (-89.5,89.5)
% 2 = 2 x planar symmetry - angle ranges: roll = (0.5,89.5), pitch = (-89.5,89.5)
% 3 = infinite symmetry - angle ranges: roll = 0, pitch = (-89.5,89.5)
% 4 = infinite symmetry + planar symmetry: roll = 0, pitch = (0.5,89.5)
symmetry    = [ 4, 2, 2,  4,   1,  4,  2,  2, 1, 0, 0, 0,  3, 0, 0,    2, 0,  3,  1,   1, 2];
rot_offsets = [90, 0, 0, 90,   0, 90,  0,  0, 0, 0, 0, 0, 90, 0, 0,    0, 0,  0, 94,  90, 0;
                0, 0, 0,  0,   0,  0,  0,  0, 0, 0, 0, 0,  0, 0, 0,    0, 0,  0,  9, -84, 0;
               90, 0, 0, 90, -22, 90, 28, 13, 4, 0, 0, 0, 90, 0, 0,  -12, 0, 92, -5,  -1, 0];

threads = 10;

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

% run sequences in parallel
seq_set_size = 100;
num_frames = 80000;
parpool('AttachedFiles',{'mesh_test.mexa64'});
parfor (seq_id = 0:(num_frames/seq_set_size)-1)
    gen_seq_silhouettes(seq_id,seq_set_size,opt,models,symmetry,rot_offsets);
end
end

function gen_seq_silhouettes(seq_id,seq_size,opt,models,symmetry,rot_offsets)
    depth2color = opt.depth2color;
    intrinsic_matrix_color = opt.intrinsic_matrix_color;

    % for each frame
    for k = seq_size*seq_id:(1+seq_id)*seq_size-1
        if exist(fullfile(opt.root, 'data_syn', sprintf('%06d-segments.mat', k)), 'file') == 2
            continue;
        end
        fprintf('%04d: iteration %06d\n', seq_id, k);

        % read image
        filename = fullfile(opt.root, 'data_syn', sprintf('%06d-color.png', k));
        I = imread(filename);
%         subplot(1, 3, 1);
%         imshow(I);
%         title('color image');
        [h,w,ch] = size(I);
        
        % read labels
        filename = fullfile(opt.root, 'data_syn', sprintf('%06d-label.png', k));
        label = imread(filename);

        % load meta-data
        filename = fullfile(opt.root, 'data_syn', sprintf('%06d-meta.mat', k));
        object = load(filename);

        % sort objects according to distances
        num = numel(object.cls_indexes);
        distances = zeros(num, 1);
        poses = object.poses;
        for j = 1:num
            distances(j) = poses(3, 4, j);
        end
        [~, index] = sort(distances, 'descend');

        crop_size = 64;
        mask.segments = zeros(num,h,w);
        mask.segments_crop = zeros(num,crop_size,crop_size);
        mask.occluded_segments_crop = zeros(num,crop_size,crop_size);
        mask.proj_distance = zeros(1,num);
        mask.cls_indexes = zeros(1,num);
        mask.viewpoints = zeros(1,num);
        mask.viewpoint_yaw = zeros(1,num);

        % for each object
        for j = 1:num
            ind = index(j);

            mask.cls_indexes(j) = object.cls_indexes(ind);

            % load RT_o2c
            RT_o2c = poses(:,:,ind);

            % use translation to get apparant orientation of object in the
            % camera view.
            pitch_angle = atan2d(RT_o2c(1,4),RT_o2c(3,4));
            roll_angle = -atan2d(RT_o2c(2,4),RT_o2c(3,4));
            eul = [roll_angle, pitch_angle, 0];
            rot = SpinCalc('EA123toDCM',eul,.0001,1);

            RT_o2c_crop = zeros(3,4);
            RT_o2c_crop(:,1:3) = rot*RT_o2c(:,1:3);
            
            % find full silhouette ground truth viewpoint
            class_idx = object.cls_indexes(ind);
            R_adj = SpinCalc('EA123toDCM',rot_offsets(:,class_idx)',.0001,0);
            [viewpoint, yaw] = get_viewpoint(RT_o2c_crop(:,1:3)*inv(R_adj), class_idx, symmetry);
            mask.viewpoints(j) = viewpoint;
            mask.viewpoint_yaw(j) = yaw;

            %% generate full silhouettes
            % projection
            x3d = models{object.cls_indexes(ind)}.v';
            x2d = project(x3d, intrinsic_matrix_color, RT_o2c);

            % projection crop
            scale = 1.05;
            % center objects
            obj_w = abs(max(x3d(:,1))-min(x3d(:,1)));
            obj_d = abs(max(x3d(:,2))-min(x3d(:,2)));
            obj_h = abs(max(x3d(:,3))-min(x3d(:,3)));
            dim = sqrt(obj_w^2 + obj_h^2 + obj_d^2);
            % assume height is minimum dimension
            min_dim = min([w,h]);
            fov = atan(min_dim/intrinsic_matrix_color(1,1));
            dist = dim/(2*tan(fov/2))*scale;
            mask.proj_distance(j) = dist;
            RT_o2c_crop(3,4) = dist;
            x2d_crop = project(x3d, intrinsic_matrix_color, RT_o2c_crop);

            face = models{object.cls_indexes(ind)}.f3';

            vertices = [x2d(face(:,1),2) x2d(face(:,1),1) ...
                        x2d(face(:,2),2) x2d(face(:,2),1) ...
                        x2d(face(:,3),2) x2d(face(:,3),1)];
            vertices_crop = [x2d_crop(face(:,1),2) x2d_crop(face(:,1),1) ...
                        x2d_crop(face(:,2),2) x2d_crop(face(:,2),1) ...
                        x2d_crop(face(:,3),2) x2d_crop(face(:,3),1)];
            % BW is the mask
            BW = mesh_test(double(vertices), h, w);
            BW_crop = mesh_test(double(vertices_crop), h, w);
            tmp = squeeze(mask.segments(j,:,:));
            tmp_crop = zeros(h,w);
            tmp(BW) = 1;
            tmp_crop(BW_crop) = 1;
            mask.segments(j,:,:) = tmp;
            dim_diff = w-h;
            w_min = floor(dim_diff/2);
            w_max = w_min + h;
            tmp_crop = tmp_crop(:,w_min:w_max);
            tmp_crop = imresize(tmp_crop,[crop_size,crop_size],'nearest');
            mask.segments_crop(j,:,:) = tmp_crop;
            
            
            %% generate occluded silhouettes
            occ_mask = zeros(h,w);
            pix_ind = find(label == class_idx);
            occ_mask(pix_ind) = 1;
            occ_mask = padarray(occ_mask, [320,320]);
            pix = project([0,0,0], intrinsic_matrix_color, RT_o2c);
            p_diff_x = pix(1) - w/2;
            p_diff_y = pix(2) - h/2;
            occ_mask = imtranslate(occ_mask, [-p_diff_x,-p_diff_y]);
            dist_offset = sqrt(RT_o2c(1,4)^2 + RT_o2c(2,4)^2 + RT_o2c(3,4)^2);
            scale_offset = dist_offset/dist;
            occ_mask = imresize(occ_mask,scale_offset);
            [om_h, om_w] = size(occ_mask);
            diff_h = om_h - h;
            diff_w = om_w - h;
            h_min = floor(diff_h/2);
            h_max = h_min + h;
            w_min = floor(diff_w/2);
            w_max = w_min + h;
            occ_mask = occ_mask(h_min:h_max,w_min:w_max);
            occ_mask = imresize(occ_mask,[crop_size,crop_size],'nearest');
            tmp_occ = zeros(size(occ_mask));
            pix_ind = find(occ_mask >= 0.5);
            tmp_occ(pix_ind) = 1;
            occ_mask = tmp_occ;
            mask.occluded_segments_crop(j,:,:) = occ_mask;

%             % show the mask
%             subplot(1, 3, 2);
%             imshow(occ_mask);
%             axis off;  
%             axis equal;
% 
%             % show the crop mask
%             subplot(1, 3, 3);
%             imshow(tmp_crop);
%             axis off;  
%             axis equal;
% 
%             pause;
        end
        
        % save segmentation masks
        filename = fullfile(opt.root, 'data_syn', sprintf('%06d-segments.mat', k));
        save(filename,'mask');

        %pause;
    end
end

function [viewpoint, yaw] = get_viewpoint(R, class_idx, symmetry)
    pose = SpinCalc('DCMtoEA123',R,.0001,0);
    if pose(1) > 180
        pose(1) = pose(1) - 360;
    elseif pose(1) < -180
        pose(1) = pose(1) + 360;
    end
    if pose(2) > 180
        pose(2) = pose(2) - 360;
    elseif pose(2) < -180
        pose(2) = pose(2) + 360;
    end

    % 0 = no symmetry - angle ranges: roll = (-179.5,179.5), pitch = (-89.5,89.5)
    % 1 = planar symmetry - angle ranges: roll = (0.5,179.5), pitch = (-89.5,89.5)
    % 2 = 2 x planar symmetry - angle ranges: roll = (0.5,89.5), pitch = (-89.5,89.5)
    % 3 = infinite symmetry - angle ranges: roll = roll = 0, pitch = (-89.5,89.5)
    % 4 = infinite symmetry + planar symmetry: roll = 0, pitch = (0.5,89.5)
    % symmetry = [4, 2, 2, 4, 1, 4, 2, 2, 1, 0, 0, 0, 3, 0, 0, 2, 0, 3, 1, 1, 2];
    roll = pose(1);
    pitch = pose(2);
    yaw = pose(3);
    if symmetry(class_idx) == 1
        if roll < 0
            roll = -roll;
            pitch = -pitch;
            yaw = yaw + 180;
        end
    elseif symmetry(class_idx) == 2
        if roll < 0
            roll = -roll;
            pitch = -pitch;
            yaw = yaw + 180;
        end
        if roll > 90
            roll = 180 - roll;
            pitch = -pitch;
            yaw = yaw + 180;
        end
    elseif symmetry(class_idx) == 3
        roll = 0;
    elseif symmetry(class_idx) == 4
        roll = 0;
        if pitch < 0
            pitch = -pitch;
            yaw = yaw + 180;
        end
    end
    if yaw > 360
        yaw = yaw - 360;
    end
    
    % find viewpoint id
    step = 5;
    if symmetry(class_idx) == 0
        angles_roll = -179.5:step:179.5;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(class_idx) == 1
        angles_roll = 0.5:step:179.5;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(class_idx) == 2
        angles_roll = 0.5:step:89.5;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(class_idx) == 3
        angles_roll = 0;
        angles_pitch = -89.5:step:89.5;
    elseif symmetry(class_idx) == 4
        angles_roll = 0;
        angles_pitch = 0.5:step:89.5;
    end
    
    if length(angles_roll) == 1
        roll_idx = 1;
    else
        roll_d=sort(abs(roll-angles_roll));
        roll_idx = find(abs(roll-angles_roll)==roll_d(1));
    end
    pitch_d=sort(abs(pitch-angles_pitch));
    pitch_idx = find(abs(pitch-angles_pitch)==pitch_d(1));
    
    viewpoint = (roll_idx-1)*length(angles_pitch) + pitch_idx;
end
    