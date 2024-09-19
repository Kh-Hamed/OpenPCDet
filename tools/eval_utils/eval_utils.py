import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import yaml


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    #################################################################################################################
    # weather_names = ['snow','rain','clear']
    # det_annos = []
    # gt_annos = []
    # weather_annos = []
    # test_waymo = False
    # test_fourseason = True
    # if True:
    #     pred_all = {}
    #     for i, batch_dict in enumerate(dataloader):
    #         if i%100==0:
    #             print(i)
    #         load_data_to_gpu(batch_dict)

    #         if getattr(args, 'infer_time', False):
    #             start_time = time.time()

    #         with torch.no_grad():
    #             pred_dicts, ret_dict = model(batch_dict)
            
    #         pred_all[batch_dict['frame_id'][0]] = pred_dicts[0]

    #         if False:
    #             print(batch_dict['frame_id'][0])
    #             pts_all = batch_dict['points'][:,1:5]             
    #             #gt_boxes = batch_dict['gt_boxes'][0]
    #             pred_boxes = pred_dicts[0]['pred_boxes']
    #             #gt_boxes = batch_dict['gt_boxes'][0][:,0:7]
    #             visualize_utils.draw_scenes(pts_all, pred_boxes)
    #             a = 0  

    #         # save out  
    #         if test_waymo:          
    #             th_score = 0.5
    #             pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy().astype(np.float64)
    #             pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy().astype(np.float64)
    #             pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()-1
    #             pred_boxes = pred_boxes[pred_scores>th_score]
    #             pred_labels = pred_labels[pred_scores>th_score]
    #             pred_boxes[:,2] -= 1.9
    #             pred_boxes = np.round(pred_boxes, 3)

    #             category_list = ['vehicle', 'pedestrian', 'cyclist']            
    #             out_dict = {}
    #             out_dict['name'] = str(batch_dict['frame_id'][0])
    #             out_dict['timestamp'] = 0
    #             out_dict['index'] = str(batch_dict['frame_id'][0])
    #             labels_list = []
    #             for j in range(pred_boxes.shape[0]):
    #                 label_j = {}
    #                 label_j['id'] = j+1
    #                 label_j['category'] = category_list[pred_labels[j]]                
    #                 dimension = {'length':float(pred_boxes[j,3]), 'width':float(pred_boxes[j,4]), 'height':float(pred_boxes[j,5])}
    #                 location = {'x':float(pred_boxes[j,0]), 'y':float(pred_boxes[j,1]), 'z':float(pred_boxes[j,2])}
    #                 orientation = {'x_rotation':0.0,'y_rotation':0.0,'z_rotation':float(pred_boxes[j,6])}
    #                 label_j['box3d'] = {'dimension':dimension, 'location':location, 'orientation':orientation}
    #                 labels_list.append(label_j)
    #             out_dict['labels'] = labels_list

    #             file_pred = batch_dict['path'][0]+'/'+batch_dict['frame_id'][0]+'_label3d.yaml'
    #             with open(file_pred, 'w') as f:
    #                 yaml.safe_dump(out_dict, f)
    #             f.close()             

    #         #
    #         if test_fourseason:
    #             disp_dict = {}

    #             if getattr(args, 'infer_time', False):
    #                 inference_time = time.time() - start_time
    #                 infer_time_meter.update(inference_time * 1000)
    #                 # use ms to measure inference time
    #                 disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

    #             statistics_info(cfg, ret_dict, metric, disp_dict)
    #             det_annos_temp = dataset.generate_prediction_dicts(
    #                 batch_dict, pred_dicts, class_names,
    #                 output_path=final_output_dir if args.save_to_file else None
    #             )
    #             det_annos += det_annos_temp
    #             if weather_names[0] in batch_dict['frame_id'][0]:
    #                 weather_annos.append(0)
    #             elif weather_names[1] in batch_dict['frame_id'][0]:
    #                 weather_annos.append(1)                    
    #             else:
    #                 weather_annos.append(2)                    

    #             # gt
    #             gt_dict = {}
    #             class_list = ['Vehicle', 'Pedestrian', 'Cyclist']
    #             if batch_dict['gt_boxes'].shape[0]* batch_dict['gt_boxes'].shape[1]:
    #                 gt_dict['pred_boxes'] = batch_dict['gt_boxes'][0][:,0:7].cuda()
    #                 gt_dict['pred_scores'] = torch.ones(gt_dict['pred_boxes'].shape[0]).cuda()
    #                 gt_dict['pred_labels'] = torch.ones(gt_dict['pred_boxes'].shape[0]).long().cuda()                  
    #                 for j in range(gt_dict['pred_boxes'].shape[0]):
    #                     if batch_dict['gt_names'][0,j] == 'Vehicle':
    #                         gt_dict['pred_labels'][j] = 0+1
    #                     if batch_dict['gt_names'][0,j] == 'Pedestrian':
    #                         gt_dict['pred_labels'][j] = 1+1
    #                     if batch_dict['gt_names'][0,j] == 'Cyclist':
    #                         gt_dict['pred_labels'][j] = 2+1                
    #             else:         
    #                 gt_dict['pred_boxes'] = torch.empty(0, 7).cuda()
    #                 gt_dict['pred_scores'] = torch.empty(0, 1).cuda()
    #                 gt_dict['pred_labels'] = torch.empty(0, 1).long().cuda()            

    #             gt_annos_temp = dataset.generate_prediction_dicts(
    #                 batch_dict, [gt_dict], class_names,
    #                 output_path=final_output_dir if args.save_to_file else None
    #             )
    #             gt_annos += gt_annos_temp
                
    #             if cfg.LOCAL_RANK == 0:
    #                 progress_bar.set_postfix(disp_dict)
    #                 progress_bar.update()

    #     if test_waymo:
    #         return 0

    ##############################################################################################################
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        #########################################################3        
        # data = {}
        # import pickle
        # data['gt_boxes'] = annos[0]['boxes_lidar']
        # data['gt_scores'] = annos[0]['score']
        # data['points'] = batch_dict['points'][:, 1:5]
        # with open('/home/hamed/FS2/output/home/hamed/FS2/tools/cfgs/FS_models/pv_rcnn_plusplus_anchor/default/eval/epoch_30/val/default/result.pkl', 'rb') as file:
        #     data_dic = pickle.load(file)
        # from tools.visual_utils.open3d_vis_utils import draw_scenes
        # draw_scenes(batch_dict['points'][:, 1:5], gt_boxes=batch_dict['gt_boxes'][0, :, 0:7], ref_boxes=  data_dic[i]['boxes_lidar'], ref_scores=data_dic[i]['score'])
        #################################################################
        # data = {}
        # import pickle
        # mapping = {'Car': 1, 'Pedestrian': 2, 'Bike': 3}
        # data['gt_boxes'] = annos[0]['boxes_lidar']
        # data['gt_scores'] = annos[0]['score']
        # data['labels'] = np.vectorize(mapping.get)(annos[0]['name'])
        # data['points'] = batch_dict['points'][:, 1:5]
        # with open('/space/userfiles/khatouna/OpenPCDet_FS/vis_fs.pkl', 'wb') as file:
        #     pickle.dump(data, file)
        #########################################################
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    # result_str, result_dict = dataset.evaluation(
    #     det_annos, gt_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    #     output_path=final_output_dir
    # )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
