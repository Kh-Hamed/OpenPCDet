import cv2
import glob
import numpy as np
from tqdm import tqdm
import random

def get_independent_frameids():
    set_name = 'trainval'
    cam_name = 'mid'
    segmentid = '2023_late_summer'
    folder_data_image = '/egr/research-canvas/four_seasons/dataset/'+segmentid+'/'
    folder_data_lidar = '/egr/research-canvas/detection3d_datasets/four_seasons/imerit_check/label3d/'+segmentid+'/'
    folder_out = '/space/userfiles/khatouna/OpenPCDet_FS/data/fourseason/ImageSets_independent/'

    #file_list = glob.glob(folder_data+'top_'+cam_name+'_raw/*_top_'+cam_name+'.jpg') 
    # file_list = glob.glob(folder_data+'top_'+cam_name+'_dd/*_top_'+cam_name+'_dd.png') 
    # file_list = glob.glob(folder_data+'top_'+cam_name+'/*_top_'+cam_name+'.jpg') 
    # file_list.sort()

    step = 4
    total_frames = 0
    frameids_list = []
    for  batch in tqdm(range(1, 19, 1)):
        file_list_lidar = glob.glob(folder_data_lidar+ f'Batch{batch}'+'/*_label3d.yaml') 
        file_list_lidar.sort()

        idx_last = -99999999
        down_rate = 4
        for i in tqdm(range(1, len(file_list_lidar) - step, step), leave=False):  
            total_frames = total_frames +1      
            frameid1 = file_list_lidar[i].split('/')[-1].split('_')[0]
            img_address1 = folder_data_image + 'top_' + cam_name + '/' + frameid1 + '_top_' + cam_name + '.jpg'
            frameid2 = file_list_lidar[i + step].split('/')[-1].split('_')[0]
            img_address2 = folder_data_image + 'top_' + cam_name + '/' + frameid2 + '_top_' + cam_name + '.jpg'
            img1 = cv2.imread(img_address1)
            img1 = cv2.resize(img1, (int(img1.shape[0]/down_rate), int(img1.shape[1]/down_rate)), interpolation = cv2.INTER_LINEAR)

            img2 = cv2.imread(img_address2)
            img2 = cv2.resize(img2, (int(img2.shape[0]/down_rate), int(img2.shape[1]/down_rate)), interpolation = cv2.INTER_LINEAR)

            dev = img1.astype(np.float64) - img2.astype(np.float64)
            dev = np.mean(abs(dev))
            if dev >= 10 and (i - idx_last) >= step :
                # print(frameid1)
                idx_last = i
                frameids_list.append(f'Batch{batch}/' + frameid1 + '_label3d.yaml')
            

            random.shuffle(frameids_list)

    with open(folder_out + segmentid + '/' + set_name + '.txt', 'w') as f:
        for tt in frameids_list:
            f.write(f"{tt}\n")
    f.close()

    print("Total frames are:")
    print(total_frames)
    


get_independent_frameids()
