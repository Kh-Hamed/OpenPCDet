from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['gt_names', 'frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ################################################################################
        # data = {}
        # import pickle
        # data['gt_boxes'] = batch_dict['gt_boxes'][0]
        # data['points'] = batch_dict['points'][:, 1:5]
        # # with open('/space/userfiles/khatouna/OpenPCDet_FS/vis_fs.pkl', 'wb') as file:
        # #     pickle.dump(data, file)
        # with open('/space/userfiles/khatouna/OpenPCDet_FS/output/space/userfiles/khatouna/OpenPCDet_FS/tools/cfgs/FS_models/pv_rcnn_plusplus_anchor/default/eval/epoch_30/val/default/result.pkl', 'rb') as file:
        #     data_dic = pickle.load(file)
        # from tools.visual_utils.open3d_vis_utils import draw_scenes
        # draw_scenes(batch_dict['points'][:, 1:5], gt_boxes=batch_dict['gt_boxes'][0, :, 0:7])

        ################################## ##############################################
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
