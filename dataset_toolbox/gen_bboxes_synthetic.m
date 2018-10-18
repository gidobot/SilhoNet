function gen_bboxes_synthetic
% Generates ROI coordinate annotation files for the synthetic images in the
% YCB-Video dataset.

opt = globals();

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
object_names = C{1};
fclose(fid);
num_objects = numel(object_names);

seq_set_size = 100;
num_frames = 80000;
parpool('AttachedFiles',{'mesh_test.mexa64'});
parfor (seq_id = 0:(num_frames/seq_set_size)-1)
% for seq_id = 0:1
    gen_seq_bboxes(seq_id,seq_set_size,opt,object_names);
end
end

function gen_seq_bboxes(seq_id,seq_size,opt,object_names)
    for k = seq_size*seq_id:(1+seq_id)*seq_size-1
        fprintf('%04d: iteration %06d\n', seq_id, k);
        
        fid = fopen(fullfile(opt.root, 'data_syn', sprintf('%06d-box.txt', k)),'w');

        % read segments
        filename = fullfile(opt.root, 'data_syn', sprintf('%06d-segments.mat', k));
        object = load(filename);
        classes = object.mask.cls_indexes;
        num = numel(classes);
        for i = 1:num
            segment = squeeze(object.mask.segments(i,:,:));
            [y,x] = ind2sub(size(segment), find(segment>0));
            if isempty(y)
                continue;
            end
            fprintf(fid,'%s %.2f %.2f %.2f %.2f\n',object_names{classes(i)},min(x),min(y),max(x),max(y));
        end
        fclose(fid);
    end
end