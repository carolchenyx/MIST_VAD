import cv2
import h5py
import os
from tqdm import tqdm
import numpy as np

"""
For training speed, we translate the video datasets into a single h5py file for reducing the indexing time in Disk
By keeping the compressed type as JPG, we can reduce the memory space

Here, we give the example as translating UCF-Crime training set into a single h5py file, you can modify it for other dataset
OR
You can modify the datasets/dataset.py for directly using the video files for testing!

"""
def findfile(start, name):
    for relpath, dirs, files in os.walk(start):
        if name in dirs:
            return True

def Video2ImgH5(video_dir,video_dir_frame,h5_path,train_list,segment_len=16,max_vid_len=2000):
    # not multi-thread, may take time
    h=h5py.File(h5_path,'a')
    for path in tqdm(train_list):
        if 'Normal' in path:
            path = path.split('/')[1]
        # vc=cv2.VideoCapture(os.path.join(video_dir,path[:-2]+".avi"))
        if findfile(video_dir_frame,path):
            for relpath, dirs, files in os.walk(video_dir_frame+path):
                lenframe = len(files)
            for i in tqdm(range(int(lenframe // segment_len))):
                tmp_frames = []
                key = path + '-{0:06d}'.format(i)
                for j in range(segment_len):
                    img = cv2.imread(video_dir_frame+path+'/'+files[i*16+j])
                    cv2.imshow("test", img)
                    cv2.waitKey(1)
                    _, frame = cv2.imencode('.JPEG', img)
                    frame = np.array(frame).tostring()
                    tmp_frames.append(frame)

                h.create_dataset(key, data=tmp_frames, chunks=True)
        else:
            vc = cv2.VideoCapture(video_dir+path+".avi")
            vid_len=vc.get(cv2.CAP_PROP_FRAME_COUNT)
            for i in tqdm(range(int(vid_len//segment_len))):
                tmp_frames=[]
                key='Normal_'+path+'-{0:06d}'.format(i)
                for j in range(segment_len):
                    ret,frame=vc.read()
                    cv2.imshow("test",frame)
                    cv2.waitKey(1)
                    _,frame=cv2.imencode('.JPEG',frame)
                    frame=np.array(frame).tostring()
                    if ret:
                        tmp_frames.append(frame)
                    else:
                        print('Bug Reported!')
                        exit(-1)
                h.create_dataset(key,data=tmp_frames,chunks=True)
        print(path)

    print('finished!')

if __name__=='__main__':
    # video_dir='/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/UCF-crime/Anomaly-Videos/'
    # h5_file_path='../data/UCF-Frames-16.h5'
    # txt_path= '/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/UCF-crime/Anomaly_Train.txt'
    video_dir='/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/SHT/training/videos/'
    video_dir_frame = '/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/AbnormalDetection/DATASET/SHT/testing/frames/'
    h5_file_path='../data/SHT-Train-16.h5'
    txt_path= '../data/SH_Train_new.txt'
    train_list=[]
    with open(txt_path,'r')as f:
        paths=f.readlines()
        for path in paths:
            # ano_type=path.strip().split('/')[0]
            # if 'Normal' in ano_type:
            #     path='Normal/'+path.strip().split('/')[-1]
            ano_type = path.strip().split(',')[1]
            if '0' in path.strip().split(',')[1]:
                path = 'Normal/' + path.strip().split(',')[0]
                train_list.append(path.strip())
            else:
                train_list.append(path.strip().split(',')[0])
    print(train_list)
    Video2ImgH5(video_dir,video_dir_frame,h5_file_path,train_list,segment_len=16)
