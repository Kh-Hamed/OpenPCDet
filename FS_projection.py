import sys
import numpy as np
import glob
import cv2, os
import matplotlib
import yaml
from PIL import Image 
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d
from pypcd4 import PointCloud
from scipy.spatial.transform import Rotation
import concurrent.futures as futures
import functools
import matplotlib.colors as mcolors
import moviepy.video.io.ImageSequenceClip

category_list = ['car', 'pedestrian', 'traffic_sign', 'traffic_light', 'bus', 'truck', 'bike', 'bike-cyclist', 'motorcycle', 'rider', 'animal']
name_list = ['car', 'ped', 'ts', 'tl', 'bus', 'truck', 'bike', 'bc', 'mc', 'rider', 'animal']
color_list = ['greenyellow', 'orange', 'red', 'purple', 'blue', 'pink', 'brown', 'olive', 'cyan', 'gray', 'orange']

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0    
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        num_colors = 100
        color_map = cm.get_cmap('viridis', num_colors)
        dis_list = (points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)**(1/2)
        dis_max = np.max(dis_list)
        colors_list = []
        for dis in dis_list:
            colors_list.append(color_map(dis/dis_max))
        colors_list = np.array(colors_list)[:,0:3]
        pts.colors = open3d.utility.Vector3dVector(colors_list)    
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        #vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        vis = draw_box(vis, ref_boxes, (0, 0, 1))

    vis.run()
    vis.destroy_window()


def lidarbox2image(K, r, t, lidar_file, box_file, image_file, file_out, d=None):
    boxes = get_boxes_from_yaml(box_file)

    pc = PointCloud.from_path(lidar_file)
    data = pc.numpy()     
    pointcloud = data[:,0:3]  
    if data.shape[1]>3:
        pointintensity = data[:,3]  
    else:
        pointintensity = (data[:,0]**2+data[:,1]**2+data[:,2]**2)**0.5
    pointintensity = pointintensity/np.max(pointintensity)

    img = cv2.imread(image_file)
    if d is not None:      
        h,w = img.shape[:2] 
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 1, (w,h))
        img = cv2.undistort(img, K, d, None, K_new)        
        K = K_new   

    R, _ = cv2.Rodrigues(r)
    M = np.concatenate([R,t], axis=1)
    P = np.matmul(K, M)

    X_lidar = np.concatenate([pointcloud, np.ones([pointcloud.shape[0],1])], axis=1)
    X_cam = np.matmul(M, X_lidar.transpose())
    X_cam = X_cam.transpose()
    idx = X_cam[:,2]>0
    X_cam = X_cam[idx]
    intensity = pointintensity[idx]
    x = np.matmul(K, X_cam.transpose())
    x = x.transpose()
    x[:,0] /= x[:,2]
    x[:,1] /= x[:,2]
    
    # draw points
    img_show = img
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    for i in range(x.shape[0]):
        px = int(x[i,0])
        py = int(x[i,1])
        dis = X_cam[i,2]
        rgba1 = cmap(dis/50)
        rgba2 = cmap(intensity[i])
        w = 0.5
        r = int((w*rgba1[0]+(1-w)*rgba2[0])*255)
        g = int((w*rgba1[1]+(1-w)*rgba2[1])*255)
        b = int((w*rgba1[2]+(1-w)*rgba2[2])*255)
        
        if px<5 or px>img_show.shape[1]-5 or py<5 or py>img_show.shape[0]-5:
            continue
        cv2.circle(img_show,(px, py), 4, (r, g, b), -1)            
    
    # draw boxes
    box_order = np.array([0,1,3,2,0,4,5,7,6,4,5,1,3,7,6,2])
    for i in range(boxes.shape[0]):
        category = int(boxes[i,7])
        obj_id = int(boxes[i,8])
        x, y, z, dx, dy, dz, angle = boxes[i,0:7]
        bbox_pts_x = np.array([[-dx/2,-dx/2,-dx/2,-dx/2,+dx/2,+dx/2,+dx/2,+dx/2]])
        bbox_pts_y = np.array([[-dy/2,-dy/2,+dy/2,+dy/2,-dy/2,-dy/2,+dy/2,+dy/2]])
        bbox_pts_z = np.array([[-dz/2,+dz/2,-dz/2,+dz/2,-dz/2,+dz/2,-dz/2,+dz/2]])
        bbox_pts = np.vstack([bbox_pts_x,bbox_pts_y,bbox_pts_z])

        Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        bbox_pts = np.matmul(Rz, bbox_pts)
        bbox_pts[0,:] += x
        bbox_pts[1,:] += y
        bbox_pts[2,:] += z    
        bbox_pts = bbox_pts.transpose()   

        X_lidar = np.concatenate([bbox_pts, np.ones([bbox_pts.shape[0],1])], axis=1)
        X_cam = np.matmul(M, X_lidar.transpose())
        X_cam = X_cam.transpose()
        x = np.matmul(K, X_cam.transpose())
        x = x.transpose()
        x[:,0] /= x[:,2]
        x[:,1] /= x[:,2]
        mask = (x[:,0]>0)*(x[:,0]<img_show.shape[1])*(x[:,1]>0)*(x[:,1]<img_show.shape[0])*(x[:,2]>0)
        if np.sum(mask)<2:
            continue

        # draw to image
        pxy = x[box_order]   
        x_mean = np.mean(pxy[:,0])
        y_mean = np.mean(pxy[:,1])
        css4 = mcolors.CSS4_COLORS[color_list[category]]
        rgba = mcolors.to_rgba_array([css4])[0]*255
        rgb = (int(rgba[2]), int(rgba[1]), int(rgba[0]))
        img_show = cv2.putText(img_show, name_list[category]+'_'+str(obj_id), (int(x_mean), int(y_mean)), 1, 3, rgb, 2, cv2.LINE_AA)
        for j in range(pxy.shape[0]-1):
            pts = (int(pxy[j,0]), int(pxy[j,1]))
            pte = (int(pxy[j+1,0]), int(pxy[j+1,1]))            
            img_show = cv2.line(img_show, pts, pte, rgb, 1, cv2.LINE_AA) 
        
    #
    print(os.path.basename(lidar_file))
    #file_out = outfolder+'/'+os.path.basename(lidar_file)[:-4]+'_lidar.png'
    cv2.imwrite(file_out, img_show)
    # cv2.imshow('',img_show)
    # cv2.waitKey(0)

    return K


def lidar2image(K, r, t, lidar_file, image_file, file_out, d=None):
    #pointcloud = np.loadtxt(lidar_file)
    pc = PointCloud.from_path(lidar_file)
    data = pc.numpy()     
    pointcloud = data[:,0:3]  
    if data.shape[1]>3:
        pointintensity = data[:,3]  
    else:
        pointintensity = (data[:,0]**2+data[:,1]**2+data[:,2]**2)**0.5
    pointintensity = pointintensity/np.max(pointintensity)

    img = cv2.imread(image_file)
    if d is not None:      
        h,w = img.shape[:2] 
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 1, (w,h))
        img = cv2.undistort(img, K, d, None, K_new)        
        K = K_new   

    R, _ = cv2.Rodrigues(r)
    M = np.concatenate([R,t], axis=1)
    P = np.matmul(K, M)

    X_lidar = np.concatenate([pointcloud, np.ones([pointcloud.shape[0],1])], axis=1)
    X_cam = np.matmul(M, X_lidar.transpose())
    X_cam = X_cam.transpose()
    idx = X_cam[:,2]>0
    X_cam = X_cam[idx]
    intensity = pointintensity[idx]
    x = np.matmul(K, X_cam.transpose())
    x = x.transpose()
    x[:,0] /= x[:,2]
    x[:,1] /= x[:,2]
    
    # draw
    img_show = img
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    for i in range(x.shape[0]):
        px = int(x[i,0])
        py = int(x[i,1])
        dis = X_cam[i,2]
        rgba1 = cmap(dis/50)
        rgba2 = cmap(intensity[i])
        w = 0.5
        r = int((w*rgba1[0]+(1-w)*rgba2[0])*255)
        g = int((w*rgba1[1]+(1-w)*rgba2[1])*255)
        b = int((w*rgba1[2]+(1-w)*rgba2[2])*255)
        
        if px<5 or px>img_show.shape[1]-5 or py<5 or py>img_show.shape[0]-5:
            continue
        cv2.circle(img_show,(px, py), 3, (r, g, b), -1)            
    
    #
    print(os.path.basename(lidar_file))
    #file_out = outfolder+'/'+os.path.basename(lidar_file)[:-4]+'_lidar.png'
    cv2.imwrite(file_out, img_show)
    # cv2.imshow('',img_show)
    # cv2.waitKey(0)

    return K


def get_boxes_from_yaml(box_file):    
    with open(box_file) as read_file:
        #data = json.load(read_file)
        data = yaml.safe_load(read_file)
    
    labels = []
    for label in data["labels"]:
        category = label['category']
        id = label['id']
        category = category_list.index(category)
        x,y,z = label["box3d"]["location"]["x"],label["box3d"]["location"]["y"],label["box3d"]["location"]["z"]
        l,w,h = label["box3d"]["dimension"]["length"],label["box3d"]["dimension"]["width"],label["box3d"]["dimension"]["height"]
        rz = label["box3d"]["orientation"]["z_rotation"]

        labels.append([x,y,z,l,w,h,rz, category, id])
    labels = np.array(labels)
    return labels

def box2image(K, r, t, box_file, image_file, file_out, d=None):
    """THIS FUNCTION ISN'T COMPLETE!
       Please implement get_boxes_from_yaml and draw_bbox
    """
    R, _ = cv2.Rodrigues(r)
    M = np.concatenate([R,t], axis=1)
    P = np.matmul(K, M)

    img = cv2.imread(image_file)
    if d is not None:      
        h,w = img.shape[:2] 
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 1, (w,h))
        img = cv2.undistort(img, K, d, None, K_new)        
        K = K_new 

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = Image.fromarray(img) 

    boxes = get_boxes_from_yaml(box_file)

    # 
    fig, ax = plt.subplots(figsize=(16, 12), dpi=80)
    ax.imshow(img)        

    box_order = np.array([0,1,3,2,0,4,5,7,6,4,5,1,3,7,6,2])
    for i in range(boxes.shape[0]):
        x, y, z, dx, dy, dz, angle = boxes[i,0:7]
        bbox_pts_x = np.array([[-dx/2,-dx/2,-dx/2,-dx/2,+dx/2,+dx/2,+dx/2,+dx/2]])
        bbox_pts_y = np.array([[-dy/2,-dy/2,+dy/2,+dy/2,-dy/2,-dy/2,+dy/2,+dy/2]])
        bbox_pts_z = np.array([[-dz/2,+dz/2,-dz/2,+dz/2,-dz/2,+dz/2,-dz/2,+dz/2]])
        bbox_pts = np.vstack([bbox_pts_x,bbox_pts_y,bbox_pts_z])

        Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        bbox_pts = np.matmul(Rz, bbox_pts)
        bbox_pts[0,:] += x
        bbox_pts[1,:] += y
        bbox_pts[2,:] += z    
        bbox_pts = bbox_pts.transpose()   

        X_lidar = np.concatenate([bbox_pts, np.ones([bbox_pts.shape[0],1])], axis=1)
        X_cam = np.matmul(M, X_lidar.transpose())
        X_cam = X_cam.transpose()
        x = np.matmul(K, X_cam.transpose())
        x = x.transpose()
        x[:,0] /= x[:,2]
        x[:,1] /= x[:,2]
        mask = (x[:,0]>0)*(x[:,0]<img.size[0])*(x[:,1]>0)*(x[:,1]<img.size[1])*(x[:,2]>0)
        if np.sum(mask)<3:
            continue

        # draw to image
        pxy = x[box_order]   
        plt.plot(pxy[:,0], pxy[:,1], color="greenyellow", linewidth=1)
    #plt.show()
    #file_out = outfolder+'/'+os.path.basename(lidar_file)[:-4]+'_label3d.png'
    plt.savefig(file_out)
    a = 0


def box2image2(K, r, t, box_file, image_file, file_out, d=None):
    """THIS FUNCTION ISN'T COMPLETE!
       Please implement get_boxes_from_yaml and draw_bbox
    """
    R, _ = cv2.Rodrigues(r)
    M = np.concatenate([R,t], axis=1)
    P = np.matmul(K, M)

    img = cv2.imread(image_file)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = Image.fromarray(img)     
    boxes = get_boxes_from_yaml(box_file)

    # 
    fig, ax = plt.subplots(figsize=(16, 12), dpi=80)
    ax.imshow(img)        

    box_order = np.array([0,1,3,2,0,4,5,7,6,4,5,1,3,7,6,2])
    for i in range(boxes.shape[0]):
        category = int(boxes[i,7])
        x, y, z, dx, dy, dz, angle = boxes[i,0:7]
        bbox_pts_x = np.array([[-dx/2,-dx/2,-dx/2,-dx/2,+dx/2,+dx/2,+dx/2,+dx/2]])
        bbox_pts_y = np.array([[-dy/2,-dy/2,+dy/2,+dy/2,-dy/2,-dy/2,+dy/2,+dy/2]])
        bbox_pts_z = np.array([[-dz/2,+dz/2,-dz/2,+dz/2,-dz/2,+dz/2,-dz/2,+dz/2]])
        bbox_pts = np.vstack([bbox_pts_x,bbox_pts_y,bbox_pts_z])

        Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        bbox_pts = np.matmul(Rz, bbox_pts)
        bbox_pts[0,:] += x
        bbox_pts[1,:] += y
        bbox_pts[2,:] += z    
        bbox_pts = bbox_pts.transpose()   

        X_lidar = np.concatenate([bbox_pts, np.ones([bbox_pts.shape[0],1])], axis=1)
        X_cam = np.matmul(M, X_lidar.transpose())
        X_cam = X_cam.transpose()
        x = np.matmul(K, X_cam.transpose())
        x = x.transpose()
        x[:,0] /= x[:,2]
        x[:,1] /= x[:,2]
        mask = (x[:,0]>0)*(x[:,0]<img.size[0])*(x[:,1]>0)*(x[:,1]<img.size[1])*(x[:,2]>0)
        #mask = (x[:,0]>0)*(x[:,0]<img.shape[1])*(x[:,1]>0)*(x[:,1]<img.shape[0])*(x[:,2]>0)
        if np.sum(mask)<3:
            continue

        # draw to image
        pxy = x[box_order]   
        x_mean = np.mean(pxy[:,0])
        y_mean = np.mean(pxy[:,1])
        plt.plot(pxy[:,0], pxy[:,1], color=color_list[category], linewidth=1)
        #plt.text(x_mean, y_mean, category_list[category], fontsize=12, bbox=dict(facecolor=color_list[category], alpha=0.5))
        plt.text(x_mean, y_mean, category_list[category], fontsize=12, color=color_list[category])
    #plt.show()
    #file_out = outfolder+'/'+os.path.basename(lidar_file)[:-4]+'_label3d.png'
    plt.savefig(file_out)
    a = 0


def box2lidar(box_file, lidar_file):
    pcd = open3d.io.read_point_cloud(lidar_file)
    pointcloud = np.asarray(pcd.points)  
    boxes = get_boxes_from_yaml(box_file)
    draw_scenes(pointcloud, boxes)


def test_all():
    """Project lidar to image.
    """    
    cam_name = 'mid'
    segmentid = '2022_rain'
    folder_data = '/egr/research-canvas/four_seasons/dataset/'+segmentid+'/'

    file_list = glob.glob(folder_data+'top_'+cam_name+'_raw/*_top_'+cam_name+'.jpg') 
    frameid_list = [file.split('/')[-1].split('_')[0] for file in file_list]
    frameid_list.sort()
    frameid = frameid_list[2000]

    # 
    #K = np.load('/home/lxh/Documents/Code/Annotation3D/Calibration/parameters_'+segmentid+'/K_'+cam_name+'.npy')    
    #r = np.load('/home/lxh/Downloads/done_calib/parameters_2023_late_summer/r_right.npy')
    #t = np.load('/home/lxh/Downloads/done_calib/parameters_2023_late_summer/t_right.npy')
    K = np.load('/home/lxh/Downloads/K'+cam_name+'.npy')    
    r = np.load('/home/lxh/Downloads/r'+cam_name+'.npy')
    t = np.load('/home/lxh/Downloads/t'+cam_name+'.npy')
    d = np.load('/home/lxh/Downloads/d'+cam_name+'.npy')

    lidar_file = folder_data+'oust/'+frameid+'_oust.pcd'
    image_file = folder_data+'top_'+cam_name+'_raw/'+frameid+'_top_'+cam_name+'.jpg'
    label3d_file = folder_data+'label3d/'+frameid+'_label3d.yaml'
    folder_out = '/home/lxh/Downloads'
    lidar2image(K, r, t, lidar_file, image_file, folder_out, d=d)
    #box2image(K, r, t, label3d_file, image_file, folder_out)


def read_misc(file_misc, cam_name):
    with open(file_misc) as read_file:
        #data = json.load(read_file)
        data = yaml.safe_load(read_file)
    
    key = '/sensors/camera/top_'+cam_name+'/image_color/compressed'
    K = data[key]['intrinsics']['K']
    K = np.array(K)

    d = np.array(data[key]['intrinsics']['d'])

    for k in data['transforms']['transforms']:
        if k['frame_id'] == 'os1_sensor' and k['child_frame_id'] == 'top_'+cam_name:
            t = np.array(k['params'][0:3]).reshape(3,1)
            quat = np.array(k['params'][3:])
            R_obj = Rotation.from_quat(quat)
            R = R_obj.as_matrix()
            r = cv2.Rodrigues(R)[0]
    
    return K, d, r, t

def process_single_frame(frameid, folder_data, folder_label, cam_name, segmentid):
    #frameid = '1649445222652077056'
    lidar_file = folder_data+'oust/'+frameid+'_oust.pcd'
    misc_file = folder_data+'/misc/'+frameid+'_misc.yaml'
    image_file = folder_data+'top_'+cam_name+'/'+frameid+'_top_'+cam_name+'.jpg'
    label3d_file = folder_label+'/'+folder+'/'+frameid+'_label3d.yaml'            
    
    #box2lidar(label3d_file, lidar_file)
    K, d, r, t = read_misc(misc_file, cam_name)
    # r = np.load('/egr/research-canvas/detection3d_datasets/four_seasons/'+segmentid+'/r_'+cam_name+'.npy')
    # t = np.load('/egr/research-canvas/detection3d_datasets/four_seasons/'+segmentid+'/t_'+cam_name+'.npy')

    file_out_lidar = folder_out+'/'+folder+'/'+frameid+'_top_'+cam_name+'_lidar.jpg'
    K_new = lidar2image(K, r, t, lidar_file, image_file, file_out_lidar, d=d)

    file_out_box = folder_out+'/'+folder+'/'+frameid+'_top_'+cam_name+'_label3d.jpg'
    box2image2(K_new, r, t, label3d_file, file_out_lidar, file_out_box, d=d)
    a = 0


def process_batch(frame_list, folder_data, folder_label, cam_name, segmentid):
    partial_process_single_frame = functools.partial(process_single_frame, 
                                         folder_data = folder_data, 
                                         folder_label = folder_label, 
                                         cam_name = cam_name, 
                                         segmentid = segmentid)

    num_workers = 16
    with futures.ThreadPoolExecutor(num_workers) as executor:
        executor.map(partial_process_single_frame, frame_list)


def process():
    """Project lidar to image.
    """    
    cam_name = 'mid'
    segmentid = '2023_late_summer'
    #segmentid = '2022_rain'
    folder_data = '/egr/research-canvas/four_seasons/dataset/'+segmentid+'/'
    folder_label = '/egr/research-canvas/detection3d_datasets/four_seasons/imerit_check/label3d/2023_later_summer'
    folder_out = '/egr/research-canvas/detection3d_datasets/four_seasons/imerit_check/projection/2023_later_summer_full'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    # get all folders
    folder_list = glob.glob(folder_label+"/*", recursive = True)
    folder_list = [f.split('/')[-1] for f in folder_list]
    folder_list.sort()

    #folder_list = ['Batch3', 'Batch8', 'Batch19', 'Batch23', 'Batch29', 'Batch36']
    #for k in range(0, 3):
    for k in range(0, len(folder_list)):
        folder = folder_list[k]
        print(folder)

        if not os.path.exists(folder_out+'/'+folder):
            os.mkdir(folder_out+'/'+folder)

        file_list = glob.glob(folder_label+'/'+folder+'/*.yaml') 
        frameid_list = [file.split('/')[-1].split('_')[0] for file in file_list]
        frameid_list.sort()

        for i in range(0, len(frameid_list)):
            id = i+1
            #if id==1 or id%5==0:
            if True:
                frameid = frameid_list[i]
                #frameid = '1649445222652077056'
                lidar_file = folder_data+'oust/'+frameid+'_oust.pcd'
                misc_file = folder_data+'/misc/'+frameid+'_misc.yaml'
                image_file = folder_data+'top_'+cam_name+'/'+frameid+'_top_'+cam_name+'.jpg'
                label3d_file = folder_label+'/'+folder+'/'+frameid+'_label3d.yaml'            
                
                #box2lidar(label3d_file, lidar_file)
                K, d, r, t = read_misc(misc_file, cam_name)
                #r = np.load('/egr/research-canvas/detection3d_datasets/four_seasons/parameters_v2/'+segmentid+'/r_'+cam_name+'.npy')
                #t = np.load('/egr/research-canvas/detection3d_datasets/four_seasons/parameters_v2/'+segmentid+'/t_'+cam_name+'.npy')

                #file_out_lidar = folder_out+'/'+folder+'/'+f'{id:04}'+'_'+frameid+'_top_'+cam_name+'.jpg'
                file_out_lidar = folder_out+'/'+folder+'/'+frameid+'.jpg'
                lidarbox2image(K, r, t, lidar_file, label3d_file, image_file, file_out_lidar, d=d)
                # file_out_lidar = folder_out+'/'+folder+'/'+frameid+'_top_'+cam_name+'_lidar.jpg'
                # K_new = lidar2image(K, r, t, lidar_file, image_file, file_out_lidar, d=d)

                # file_out_box = folder_out+'/'+folder+'/'+frameid+'_top_'+cam_name+'_label3d.jpg'
                # box2image2(K_new, r, t, label3d_file, file_out_lidar, file_out_box, d=d)
                # a = 0


def make_videos():
    cam_name = 'mid'
    #segmentid = '2023_late_summer'
    folder_out = '/egr/research-canvas/detection3d_datasets/four_seasons/imerit_check/projection/cvpr_video_keyframes'

    # get all folders
    folder_list = glob.glob(folder_out+"/*", recursive = True)
    folder_list = [f.split('/')[-1] for f in folder_list]
    folder_list.sort()

    file_list_all = []
    for k in range(0, len(folder_list)):
        folder = folder_list[k]
        print(folder)
        folder_imgs = folder_out+'/'+folder
        file_list = glob.glob(folder_imgs+'/*.jpg') 
        file_video = folder_out+'/'+folder+'.mp4'

        name_list = []
        for ff in file_list:
            name = ff.split('/')[-1][:4]
            name_list.append(float(name))
        order = np.argsort(np.array(name_list))
        file_list_new = []
        for idx in order:
            file_list_new.append(file_list[idx])

        file_list_all += file_list_new
        fps = 5
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_list_new, fps=fps)
        clip.write_videofile(file_video)
        clip.close()
        
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_list_all, fps=fps)
    clip.write_videofile(folder_out+'/all.mp4')
    clip.close()


def check_calib():
    segmentid = '2023_neighborhood_fall'
    cam_name = 'right'
    frameid = '1699890144956609024'
    
    lidar_file = '/egr/research-canvas/detection3d_datasets/four_seasons/oust/2023_neighborhood_fall/oust/'+frameid+'_oust.pcd'
    misc_file = '/egr/research-canvas/detection3d_datasets/four_seasons/misc/2023_neighborhood_fall/misc/'+frameid+'_misc.yaml'
    image_file = '/egr/research-canvas/detection3d_datasets/four_seasons/top_left/2023_neighborhood_fall/top_left/'+frameid+'_top_left.jpg'
    
    #K, d, r, t = read_misc(misc_file, cam_name)
    r = np.load('/egr/research-canvas/detection3d_datasets/four_seasons/parameters_v2/'+segmentid+'/r_'+cam_name+'.npy')
    t = np.load('/egr/research-canvas/detection3d_datasets/four_seasons/parameters_v2/'+segmentid+'/t_'+cam_name+'.npy')
    print(t.reshape(-1))

    if True:
        print(r)
        # convert opencv r to scipy quat
        R = cv2.Rodrigues(r)[0]
        R_obj = Rotation.from_matrix(R)
        quat = R_obj.as_quat()
        print(quat)

        # convert scipy quat to opencv r
        R_obj = Rotation.from_quat(quat)
        R = R_obj.as_matrix()
        r = cv2.Rodrigues(R)[0]
        print(r)
            
    file_out_lidar = '/egr/research-canvas/detection3d_datasets/four_seasons/'+frameid+'_lidar.jpg'
    K_new = lidar2image(K, r, t, lidar_file, image_file, file_out_lidar, d=d)

if __name__ == '__main__':
    #check_calib()
    process()
    #make_videos()
