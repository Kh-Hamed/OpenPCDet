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
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import visualize_utils

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
        self.path = dataset_cfg.PATH
        if dataset_cfg.get('FILE_LIST', None) is not None:
            self.file_list = dataset_cfg.FILE_LIST
        else:
            self.file_list = None
        self.shift = dataset_cfg.SHIFT

        self.points_list = []
        self.labels_list = [] 
        self.names_list = []
        self.path_list = []   

        if self.file_list is None:
            names = [os.path.basename(f)[:-9] for f in glob.glob(self.path+'/*_oust.txt')]            
            self.points_list = [self.path+'/'+f+'_oust.txt' for f in names]
            self.path_list = [self.path for f in names]
            self.names_list = names
            if self.training:
                self.labels_list = [self.path+'/'+f+'_label3d.yaml' for f in names]
        else:
            fopen = open(self.file_list, 'r')
            relative_path = fopen.readlines()
            fopen.close()
            # idx= np.random.permutation(len(relative_path))
            # relative_path = np.array(relative_path)
            # fopen = open('/home/xinwei/Downloads/snow_2023_train.txt', 'w')
            # fopen.writelines(relative_path[idx[0:400]])
            # fopen.close()
            # fopen = open('/home/xinwei/Downloads/snow_2023_val.txt', 'w')
            # fopen.writelines(relative_path[idx[400:]])
            # fopen.close()

            names = [f[2:-14] for f in relative_path]            
            self.points_list = [self.path+'/'+f+'_oust.txt' for f in names]
            self.path_list = [self.path for f in names]
            self.names_list = names
            self.labels_list = [self.path+'/'+f+'_label3d.yaml' for f in names]

            
        #self.samples = [os.path.basename(f) for f in glob.glob(self.path+'/label/*ori.npy')]  
        print('Total samples: '+str(len(self.points_list)))

        a = 0

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, index):     
        test_waymo = True

        index = index % len(self.points_list) 

        points = np.loadtxt(self.points_list[index])        
        points = points[:,0:3]
        # change oust x to make x forward
        #points[:,0] *= -1
        #pcd = o3d.io.read_point_cloud(self.points_list[index])
        #points = np.asarray(pcd.points) 

        # 
        if len(self.labels_list):
            data = yaml.safe_load(open(self.labels_list[index]))
            gt_boxes = []
            gt_names = []
            for i in range(len(data['labels'])):
                box3d = data['labels'][i]['box3d']
                category = data['labels'][i]['category']
                gt_boxes.append([box3d['location']['x'],box3d['location']['y'],box3d['location']['z'],
                                    box3d['dimension']['length'],box3d['dimension']['width'],box3d['dimension']['height'],
                                    box3d['orientation']['z_rotation']])
                gt_names.append(string.capwords(category))

            gt_boxes = np.array(gt_boxes)
            gt_names = np.array(gt_names)
            input_dict = {
                'points': points,
                'gt_boxes':gt_boxes,
                'gt_names':gt_names,
                'frame_id': self.names_list[index],
                'calib': None,
                'image_shape': 0,
                'path': self.path_list[index]
            }
        else:
            input_dict = {
                'points': points,
                'frame_id': self.names_list[index],
                'calib': None,
                'image_shape': 0,
                'path': self.path_list[index]
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
            'Vehicle': 'Car',
            'Pedestrian': 'Pedestrian',
            'Cyclist': 'Cyclist',
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
                    gt_boxes_lidar = anno['gt_boxes'].copy()

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

    def evaluation(self, det_annos, gt_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = copy.deepcopy(gt_annos)
        return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)


def generate_train_val_split():
    label_folder = '/home/xinwei/Downloads/four-seasons-labeling-Kookjin/'
    lidar_folder = '/home/xinwei/remote/four_seasons_labellist/'
    full_path = [os.path.join(dp, f) for dp, dn, filenames in os.walk(label_folder) for f in filenames if os.path.splitext(f)[1] == '.yaml']
    relative_path = [p[len(label_folder):] for p in full_path]
    
    random_idx = np.random.permutation(len(relative_path))
    num_train = int(len(relative_path)*0.8)
    num_val = len(relative_path)-num_train
    train_split = [relative_path[idx] for idx in random_idx[:num_train]]
    val_split = [relative_path[idx] for idx in random_idx[num_train:]]

    fp = open(label_folder+'train.txt','w')
    for i in range(len(train_split)):
        # check if points exists
        file_lidar = lidar_folder+train_split[i][:-13]+'_oust.txt'        
        if os.path.exists(file_lidar):
            fp.write(train_split[i]+'\n')
    fp.close()


    fp = open(label_folder+'val.txt','w')
    for i in range(len(val_split)):
        # check if points exists
        file_lidar = lidar_folder+val_split[i][:-13]+'_oust.txt'        
        if os.path.exists(file_lidar):
            fp.write(val_split[i]+'\n')
    fp.close()


def generate_gt_database():
    label_folder = '/home/xinwei/Downloads/four-seasons-labeling-Kookjin/'
    lidar_folder = '/home/xinwei/remote/four_seasons_labellist/'
    file_train = label_folder+'train.txt'
    fp = open(file_train, 'r')
    file_list = fp.readlines()
    fp.close()

    db_info_vechile = []
    db_info_pedestrial = []
    db_info_cyclist = []

    for file in file_list:        
        file = file[:-1]
        file_gt = label_folder+file
        file_lidar = lidar_folder+file[:-13]+'_oust.txt'

        #
        data = yaml.safe_load(open(file_gt))
        gt_boxes = []
        gt_names = []
        for i in range(len(data['labels'])):
            box3d = data['labels'][i]['box3d']
            category = data['labels'][i]['category']
            gt_boxes.append([box3d['location']['x'],box3d['location']['y'],box3d['location']['z'],
                                box3d['dimension']['length'],box3d['dimension']['width'],box3d['dimension']['height'],
                                box3d['orientation']['z_rotation']])
            gt_names.append(string.capwords(category))

        gt_boxes = np.array(gt_boxes)
        gt_names = np.array(gt_names)
        print([file, gt_boxes.shape[0]])

        # 
        points = np.loadtxt(file_lidar)        
        points = points[:,0:3]

        #visualize_utils.draw_scenes(points,gt_boxes)

        for j in range(gt_boxes.shape[0]):      
            box_expand = copy.deepcopy(gt_boxes[j:j+1])
            mask_temp = roiaware_pool3d_utils.points_in_boxes_cpu(points, box_expand).squeeze(0)
            pts_temp = points[mask_temp==1]

            if pts_temp.shape[0]<3:
                continue
            pts_temp[:,0:3] -= gt_boxes[j,0:3]

            info_new = {}
            if gt_names[j] == 'Vehicle':
                info_new['name'] = 'Vehicle'
                info_new['path'] = 'db/veh_'+str(len(db_info_vechile))+'.bin'
                with open(label_folder+'/'+info_new['path'], 'w') as f:
                    pts_temp.tofile(f)
                info_new['box3d_lidar'] = gt_boxes[j,0:7]
                info_new['num_points_in_gt'] = pts_temp.shape[0]
                db_info_vechile.append(info_new)

            if gt_names[j] == 'Pedestrian':
                info_new['name'] = 'Pedestrian'
                info_new['path'] = 'db/ped_'+str(len(db_info_pedestrial))+'.bin'
                with open(label_folder+'/'+info_new['path'], 'w') as f:
                    pts_temp.tofile(f)
                info_new['box3d_lidar'] = gt_boxes[j,0:7]
                info_new['num_points_in_gt'] = pts_temp.shape[0]
                db_info_pedestrial.append(info_new)

            if gt_names[j] == 'Cyclist':
                info_new['name'] = 'Cyclist'
                info_new['path'] = 'db/cyc_'+str(len(db_info_cyclist))+'.bin'
                with open(label_folder+'/'+info_new['path'], 'w') as f:
                    pts_temp.tofile(f)
                info_new['box3d_lidar'] = gt_boxes[j,0:7]
                info_new['num_points_in_gt'] = pts_temp.shape[0]
                db_info_cyclist.append(info_new)                

    db_info_final = {}
    db_info_final['Vehicle'] = db_info_vechile
    db_info_final['Pedestrian'] = db_info_pedestrial
    db_info_final['Cyclist'] = db_info_cyclist
    with open(label_folder +'/dbinfo.pkl', 'wb') as f:
        pickle.dump(db_info_final, f)


if __name__ == '__main__':
    #generate_train_val_split()
    generate_gt_database()