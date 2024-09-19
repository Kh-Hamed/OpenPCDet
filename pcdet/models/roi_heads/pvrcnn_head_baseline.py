import torch.nn as nn
import time
import torch

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
######################################################################
from .diffusion import image_guided_generation
from .diffusion import FullyConnectedNN
import torch.nn.functional as F
import copy
####################################################################


def focal_loss(pred, target, alpha=0.50, gamma=2.0, reduction='mean'):
    """
    Focal Loss for binary classification.
    
    Args:
        pred (torch.Tensor): Predictions from the model (logits or probabilities).
        target (torch.Tensor): Ground truth labels (0 or 1).
        alpha (float): Weighting factor for the class.
        gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
        reduction (str): 'mean' or 'sum'. Specifies the method to reduce the loss.

    Returns:
        torch.Tensor: Computed focal loss.
    """
    # Apply sigmoid to get probabilities if input is logits
    pred = torch.sigmoid(pred)
    
    # Compute binary cross entropy
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Compute the weights
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    
    # Apply weighting to the BCE loss
    focal_loss = focal_weight * bce_loss
    # focal_loss = bce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Print the incoming gradient before reversal
        # print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
        # print(f"Incoming grad_output: {grad_output}")
        
        
        # Reverse the gradient by multiplying by -lambda_
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        
        # Print the reversed gradient
        # print(f"Reversed grad_input: {grad_input}")
        # print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        
        return grad_input, None  # Second return value is None since lambda_ doesn't need a gradient


class PVRCNNHeadBaseline(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.binary_classifier = self.make_fc_layers(
            input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.CLS_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        # batch_size = batch_dict['batch_size']
        batch_size = batch_dict['rois'].shape[0]
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            # targets_dict['rcnn_cls'] = rcnn_cls
            # targets_dict['rcnn_reg'] = rcnn_reg
            ########################################################################
            targets_dict['rcnn_cls'] = rcnn_cls[0:targets_dict['gt_of_rois'].shape[0] * targets_dict['gt_of_rois'].shape[1]]
            targets_dict['rcnn_reg'] = rcnn_reg[0:targets_dict['gt_of_rois'].shape[0] * targets_dict['gt_of_rois'].shape[1]]
            ########################################################################
            self.forward_ret_dict = targets_dict
            ####################################################################################################
            import torch
            # print(shared_features[0, :, 0])
            shared_features_diff = shared_features.permute(0, 2, 1)
            num_source_samples = batch_dict['batch_size'] * 128
            num_target_samples = shared_features.shape[0] - batch_dict['batch_size'] * 128
            mask = (torch.sigmoid(rcnn_cls) >= 0.30).clone().detach()
            mask_target = (torch.sigmoid(rcnn_cls[num_source_samples: ]) >= 0.3).clone().detach()
            mask_source = (torch.sigmoid(rcnn_cls[ :num_source_samples]) >= 0.3).clone().detach()
            shared_features_T = shared_features_diff[num_source_samples: ][mask_target.squeeze(-1), :, :]
            shared_features_S_set = shared_features_diff[: num_source_samples][mask_source.squeeze(-1), :, :]
            indices = torch.randperm(shared_features_S_set.size(0))[:torch.min(mask_source.sum(), mask_target.sum())]
            shared_features_S = shared_features_S_set[indices]

            features = torch.cat((shared_features_S, shared_features_T), 0)
            # Classify
            bc_labels_t = torch.ones(shared_features_T.size(0), 1).to(shared_features_T.device)
            bc_labels_s = torch.zeros(shared_features_S.size(0), 1).to(shared_features_S.device)
            bc_labels = torch.cat((bc_labels_s, bc_labels_t))  # Shape: [256, 200]

            min_val = torch.zeros((features.shape[0], 1, 1), device=features.device)
            max_val = torch.zeros((features.shape[0], 1, 1), device=features.device)
            for i in range(features.shape[0]):
                # mask = image_tensor[i, 0, :] !=0
                min_val[i, 0, 0], _ = torch.min(features[i, 0, :], dim=0, keepdim=True)
                max_val[i, 0, 0], _ = torch.max(features[i, 0, :], dim=0, keepdim=True)
            
            normalized_features = (features - min_val) / (max_val - min_val)
            
            # Scale to range [-1, 1]
            normalized_features = normalized_features * 2 - 1
            if normalized_features.shape[0] > 1 :
                # a = normalized_features.permute(0, 2, 1).detach()
                classification_output = self.binary_classifier(normalized_features.permute(0, 2, 1).detach())
                loss = focal_loss(classification_output.squeeze(-1),  bc_labels.float())
            else:
                loss = 0


            # for name, param in self.binary_classifier.named_parameters():
            #     if 'weight' in name:  # You can filter for weights if needed
            #         print(f"Layer: {name}, Weights: {param.data[0:5, 0:5, 0]}")
            #         break
            shared_features_DA_pred, shared_features_DA_labels = image_guided_generation(shared_features_T, copy.deepcopy(self.binary_classifier.state_dict()))
            
            # for name, param in self.binary_classifier.named_parameters():
            #     if 'weight' in name:  # You can filter for weights if needed
            #         print(f"Layer: {name}, Weights: {param.data[0:5, 0:5, 0]}")
            #         break
            self.forward_ret_dict['shared_features_DA_labels'] = shared_features_DA_labels
            self.forward_ret_dict['shared_features_DA_pred'] = shared_features_DA_pred
            self.forward_ret_dict['bc_loss'] = loss

            # self.forward_ret_dict['loss_boundry'] = loss_boundry

        return batch_dict