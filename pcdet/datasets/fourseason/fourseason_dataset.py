# find Batch{1..10} -type f -name '*_label3d.yaml' | sed 's|.*/||' > /space/userfiles/khatouna/OpenPCDet_FS/data/fourseason/ImageSets/train_summer.txt
# find Batch{1..10} -type f -name '*_label3d.yaml' | \sort -t'_' -k1,1n | \awk 'NR % 4 == 1' > /space/userfiles/khatouna/OpenPCDet_FS/data/fourseason/ImageSets/train_summer.txt
# output_file="/space/userfiles/khatouna/OpenPCDet_FS/data/fourseason/ImageSets/train_summer.txt"
# > "$output_file"  # Clear the output file if it already exists

# # Loop over each Batch folder
# for batch_dir in Batch{1..10}; do
#     # Check if directory exists
#     if [ -d "$batch_dir" ]; then
#         # Find, sort, and sample files within the current Batch folder
#         find "$batch_dir" -type f -name '*_label3d.yaml' | \
#         sort -t'_' -k1,1n | \
#         awk 'NR % 5 == 1' >> "$output_file"  # Append results to the output file
#     fi
# done

# used_classes_fs=['Vehicle', 'Pedestrian', 'Cyclist']
used_classes_fs=['Car', 'Pedestrian', 'Bike']
import copy
import pickle
import glob
import os
import json
import yaml
import string
import numpy as np
import open3d as o3d
from os import listdir
from os.path import exists,isfile
from skimage import io
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.utils import common_utils, box_utils
from pypcd4 import PointCloud
# from pypcd import *
import torch
import multiprocessing
from pathlib import Path
from functools import partial
from tqdm import tqdm
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

class FourSeasonDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )


        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.split_dir = self.root_path / (self.split + '.txt')
        self.points_list = []
        self.labels_list = [] 

        self.path_lidar = dataset_cfg.PATH_LIDAR
        self.path_label = dataset_cfg.PATH_LABEL
        ################################ DA  ###################################################
        self.path_lidar_T = dataset_cfg.PATH_LIDAR_T
        self.fs_infos_T = []
        ###################################################################################

        self.fs_infos = []
        self.labels_list_sampled = []
        self.set_split()
        self.include_fs_data(self.mode)
        self.include_fs_data(self.mode, DA= True)
        


    def include_fs_data(self, mode, DA=False):
        if self.logger is not None:
            self.logger.info('Loading Fourseason dataset')
        fs_infos = []
        if DA is False:
            for info_path in self.dataset_cfg.INFO_PATH[mode]:
                info_path = self.root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    fs_infos.extend(infos)

            self.fs_infos.extend(fs_infos)
        else:
            for info_path in self.dataset_cfg.INFO_PATH[mode]:
                info_path = self.root_path_T / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    fs_infos.extend(infos)

            self.fs_infos_T.extend(fs_infos)

        

        if self.logger is not None:
            self.logger.info('Total samples for Fourseason dataset: %d' % (len(self.fs_infos)))

    def set_split(self, split = None):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        if split is not None:
            self.split_dir = self.root_path / (split + '.txt')
        else:
            self.split_dir = self.root_path / (self.split + '.txt')
        

        fopen = open(self.split_dir, 'r')
        relative_path = fopen.readlines()
        fopen.close()        

        names_with_Batch = [f[0:-14] for f in relative_path]
        names = [f.split('/')[1] for f in names_with_Batch] 
        self.names_list = names          
        self.points_list = [self.path_lidar+'/'+f+'_oust.pcd' for f in names]
        # self.points_list = [self.path_lidar+'/'+f+'_oust.txt' for f in names]
        self.labels_list = [self.path_label+'/'+f+'_label3d.yaml' for f in names_with_Batch]


    def get_lidar(self, sequence_name, path = None, DA = False):
        if path ==None:
            if DA is False:
                lidar_file = Path(self.path_lidar) / Path(sequence_name + '_oust.pcd' )
            else:
                lidar_file = Path(self.path_lidar_T) / Path(sequence_name + '_oust.pcd' )
            points_all = PointCloud.from_path(lidar_file)
            pc = points_all.numpy().astype(np.float64)     
            pointcloud = pc[:,0:3] 


            return pointcloud



    def __len__(self):
        # return len(self.points_list)
        if self._merge_all_iters_to_one_epoch:
            return len(self.fs_infos) * self.total_epochs
        return len(self.fs_infos)

    def __getitem__(self, index): 
        
        # index = 520
        
        index = index % len(self.fs_infos)

        info = copy.deepcopy(self.fs_infos[index])
        ######################################################################
        points_T = None
        if index < len(self.fs_infos_T) and (self.training):
            info_T = copy.deepcopy(self.fs_infos_T[index])
            points_T = self.get_lidar(info_T['point_cloud']['lidar_sequence'], DA= True)


        ######################################################################

        points = self.get_lidar(info['point_cloud']['lidar_sequence'])

        
        gt_boxes = info['annos']['gt_boxes_lidar']
        gt_names = info['annos']['gt_names']
        input_dict = {
            'points': points,
            'gt_boxes':gt_boxes,
            'gt_names':gt_names,
            'frame_id': info['point_cloud']['lidar_sequence'],
            'calib': None,
            'image_shape': 0,
            'points_T': points_T

        }

        if self.training:
            if gt_boxes.shape[0] == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)                        

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            annos.append(single_pred_dict)

        return annos

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'Car': 'Car',
            'Pedestrian': 'Pedestrian',
            'Bike': 'Cyclist',
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.fs_infos]
        return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)

    # def evaluation(self, det_annos, gt_annos, class_names, **kwargs):
    #     eval_det_annos = copy.deepcopy(det_annos)
    #     eval_gt_annos = copy.deepcopy(gt_annos)
    #     return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
    


    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, update_info_only=False):
        from pcdet.datasets.fourseason import fourseason_utils
        print('---------------The fourseason sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.labels_list)))

        process_single_sequence = partial(
            fourseason_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label, update_info_only=update_info_only
        )
        labels_list_sampled = self.labels_list[::sampled_interval]
        labels_list_sampled = [Path(path) for path in labels_list_sampled]
        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, labels_list_sampled),
                                       total=len(labels_list_sampled)))
        # process_single_sequence(labels_list_sampled[23])
        return sequence_infos


    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):

    

        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / ('%s_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        # db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        for k in tqdm(range(0, len(infos), sampled_interval)):
            info = infos[k]
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            points = self.get_lidar(sequence_name)

            annos = info['annos']
            names = annos['gt_names']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue


            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sequence_name, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                if gt_points.shape[0]<3:
                    continue
                gt_points[:, 0:3] -= gt_boxes[i, 0:3]

                if (used_classes is None) or names[i] in used_classes:
                    gt_points = gt_points.astype(np.float64)
                    assert gt_points.dtype == np.float64
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                                'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0]}


                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    


def create_fs_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='fs',
                       workers=min(16, multiprocessing.cpu_count()), update_info_only=False):
    dataset = FourSeasonDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    fs_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, update_info_only=update_info_only
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(fs_infos_train, f)
    print('----------------fourseason info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    fs_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, update_info_only=update_info_only
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(fs_infos_val, f)
    print('----------------fourseason info val file is saved to %s----------------' % val_filename)

    if update_info_only:
        return

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=used_classes_fs, processed_data_tag=processed_data_tag
    )
    print('---------------Data preparation Done---------------')


def create_fs_gt_database(
    dataset_cfg, class_names, data_path, save_path, processed_data_tag='fs',
    workers=min(16, multiprocessing.cpu_count()), use_parallel=False, crop_gt_with_tail=False):
    dataset = FourSeasonDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    train_split = 'train'
    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)

    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=used_classes_fs, processed_data_tag=processed_data_tag
    )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/space/userfiles/khatouna/OpenPCDet_FS/tools/cfgs/dataset_configs/fourseason_dataset.yaml', help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_fs_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='fs', help='')
    parser.add_argument('--update_info_only', action='store_true', default=False, help='')
    parser.add_argument('--use_parallel', action='store_true', default=False, help='')
    parser.add_argument('--wo_crop_gt_with_tail', action='store_true', default=False, help='')

    args = parser.parse_args()

    # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    ROOT_DIR = Path('/space/userfiles/khatouna/OpenPCDet_FS/')

    if args.func == 'create_fs_infos':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_fs_infos(
            dataset_cfg=dataset_cfg,
            class_names=used_classes_fs,
            data_path=ROOT_DIR / 'data' / 'fourseason',
            save_path=ROOT_DIR / 'data' / 'fourseason',
            raw_data_tag='',
            processed_data_tag=args.processed_data_tag,
            update_info_only=args.update_info_only
        )
    elif args.func == 'create_fs_gt_database':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_fs_gt_database(
            dataset_cfg=dataset_cfg,
            class_names=used_classes_fs,
            data_path=ROOT_DIR / 'data' / 'fourseason',
            save_path=ROOT_DIR / 'data' / 'fourseason',
            processed_data_tag=args.processed_data_tag,
            use_parallel=args.use_parallel, 
            crop_gt_with_tail=not args.wo_crop_gt_with_tail
        )
    else:
        raise NotImplementedError