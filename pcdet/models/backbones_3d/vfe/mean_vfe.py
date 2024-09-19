import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        # points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        # normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        # points_mean = points_mean / normalizer
        # batch_dict['voxel_features'] = points_mean.contiguous()

        voxel_feature, voxel_num_point = batch_dict['voxels'], batch_dict['voxel_num_points']
        if batch_dict.get('voxels_T', None) != None:
            voxel_features_T, voxel_num_points_t = batch_dict['voxels_T'], batch_dict['voxel_num_points_T']
            voxel_features, voxel_num_points = torch.concatenate([voxel_feature, voxel_features_T], axis=0), torch.concatenate([voxel_num_point, voxel_num_points_t], axis=0)
        else:
            voxel_features, voxel_num_points = voxel_feature, voxel_num_point  
        
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
