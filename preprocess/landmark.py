import os
import argparse
import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
from skimage import transform
from facelandmarks.faceland import FaceLanndInference
from tqdm import tqdm
import multiprocessing
import subprocess
import scipy.signal
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file-list', required=True, help='filelist dir contains {filelist}-test.txt')
parser.add_argument('-i', '--input', required=True, help='mp4 data path')
parser.add_argument('-o', '--output', required=True, help='target mp4 data path')
parser.add_argument('-w', '--worker', default=4, type=int, help='size of multiprocessing pool')
parser.add_argument('-p', '--pth', default='./facelandmarks/faceland.pth', help='face landmark model path')
parser.add_argument('-t', '--type', default='face', choices=['face', 'lip'], help='crop face or lip')
parser.add_argument('--scale', default=1.1, type=float, help='croped area expand ratio')
parser.add_argument('--size', default=112, type=int, help='output mp4 file frame size')
parser.add_argument('--save-landmark', default=False, action='store_true', help='save landmark result or not')
"""
98 point landmarks (0~97)

lip region:
    76 ~ 95
    76: left corner
    82: right corner
    78,80 up
    85 down
face center:
    54
"""

# load lanmark extractor
face_landmark_batchsize = 128
model_path = os.path.join(os.path.dirname(__file__), 'facelandmarks/faceland.pth')
checkpoint = torch.load(model_path)
plfd_backbone = FaceLanndInference().cuda()
plfd_backbone.load_state_dict(checkpoint)
plfd_backbone.eval()
plfd_backbone = plfd_backbone.cuda()

# transform from mat to tensor
tf = transforms.Compose([transforms.ToTensor()])


# extract landmark
def extract_landmark(ori_v_fn):
    video = cv2.VideoCapture(ori_v_fn)
    frame_cnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
    landmark_list = []
    frame_list = []
    for i in range(int(frame_cnt)):
        success, ori_frame = video.read()
        if not success:
            return []
        frame = cv2.resize(ori_frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(tf(frame))
        if (i+1)%face_landmark_batchsize == 0 or i+1 == int(frame_cnt):
            frame_batch = torch.stack(frame_list).cuda()
            landmarks = plfd_backbone(frame_batch)
            # float landmark_list, value between 0~1
            landmark_list.extend([landmark.detach().cpu().numpy().reshape(-1, 2) for landmark in landmarks])
            frame_list = []
    return landmark_list


def get_face_center_r(landmark):
    x = [p[0] for p in landmark]
    y = [p[1] for p in landmark]
    center = landmark[54]
    x_distance = [abs(center[0] - px) for px in x]
    y_distance = [abs(center[1] - py) for py in y]
    max_r = max(max(x_distance), max(y_distance))
    roi = [center[0] - max_r, center[0] + max_r, center[1] - max_r, center[1] + max_r]
    return roi


def get_lip_center_r(landmark):
    x = [p[0] for p in landmark[76:96]]
    y = [p[1] for p in landmark[76:96]]
    center = [
        (landmark[76][0] + landmark[82][0]) / 2,
        ((landmark[78][1] + landmark[80][1]) / 2 + landmark[85][1]) / 2,
    ]
    x_distance = [abs(center[0] - px) for px in x]
    y_distance = [abs(center[1] - py) for py in y]
    max_r = max(max(x_distance), max(y_distance))
    roi = [center[0] - max_r, center[0] + max_r, center[1] - max_r, center[1] + max_r]
    return roi


def smooth_sequence(seq: list) -> list:
    win_len = min(len(seq), 21)
    if win_len % 2 == 0:
        win_len -= 1
    seq = scipy.signal.savgol_filter(seq, window_length=win_len, polyorder=1)
    return seq


def smooth_center(centers):
    x_seq = smooth_sequence([center[0] for center in centers])
    y_seq = smooth_sequence([center[1] for center in centers])
    return [[x_seq[i], y_seq[i]] for i in range(len(centers))]


def get_roi(landmarks, get_center_r_fn, scale):
    rois = [get_center_r_fn(landmark) for landmark in landmarks]
    centers = smooth_center([[(roi[0] + roi[1]) / 2, (roi[2] + roi[3]) / 2] for roi in rois])
    max_r = max([abs(roi[0] - roi[1]) / 2 for roi in rois]) * scale
    public_rois = [[center[0] - max_r, center[0] + max_r, center[1] - max_r, center[1] + max_r] for center in centers]
    return public_rois


def crop_face(ori_v_fn, dst_v_fn, tar_size=112, scale=1.1, face=True, save_landmark=False, landmark_fn=None):
    try:
        if landmark_fn is not None and os.path.exists(landmark_fn):
            landmark_list = np.load(landmark_fn)
        else:
            landmark_list = extract_landmark(ori_v_fn)

        if landmark_list == []:
            return
        video = cv2.VideoCapture(ori_v_fn)
        frame_cnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
        size_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        size_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if face:
            rois = get_roi(landmarks=landmark_list, get_center_r_fn=get_face_center_r, scale=scale)
        else:
            rois = get_roi(landmarks=landmark_list, get_center_r_fn=get_lip_center_r, scale=scale)
        float_rois = [[roi[0] * size_w, roi[1] * size_w, roi[2] * size_h, roi[3] * size_h] for roi in rois]
        int_rois = []
        for roi in float_rois:
            center_x = round((roi[0] + roi[1]) / 2)
            center_y = round((roi[2] + roi[3]) / 2)
            r = round(roi[1] - center_x)
            int_rois.append([center_x - r, center_x + r, center_y - r, center_y + r])
        # crop
        new_frames = []
        for i in range(int(frame_cnt)):
            success, ori_frame = video.read()
            if not success:
                return
            if min(int_rois[i]) < 0 or max(int_rois[i][2:]) > size_h or max(int_rois[i][:2]) > size_w:
                ori_frame = cv2.copyMakeBorder(ori_frame,
                                               int(size_h),
                                               int(size_h),
                                               int(size_w),
                                               int(size_w),
                                               borderType=cv2.BORDER_CONSTANT,
                                               value=0)
                crop_frame = ori_frame[int(size_h) + int_rois[i][2]:int(size_h) + int_rois[i][3],
                                       int(size_w) + int_rois[i][0]:int(size_w) + int_rois[i][1], :]
            else:
                crop_frame = ori_frame[int_rois[i][2]:int_rois[i][3], int_rois[i][0]:int_rois[i][1], :]
            # if crop_frame.empty():
            #     print(ori_v_fn)
            #     crop_frame = ori_frame
            try:
                crop_frame = cv2.resize(crop_frame, (tar_size, tar_size))
            except Exception as e:
                print(rois[i])
                print(int_rois[i])
                print(crop_frame)
                print(e)
            new_frames.append(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
        Path(os.path.dirname(dst_v_fn)).mkdir(exist_ok=True, parents=True)
        torchvision.io.write_video(dst_v_fn, video_array=torch.from_numpy(np.array(new_frames)), fps=25)
        if save_landmark:
            Path(os.path.dirname(landmark_fn)).mkdir(exist_ok=True, parents=True)
            np.save(landmark_fn, landmark_list)
    except Exception as e:
        print(ori_v_fn)
        print(e)
        exit(-1)


def crop_face_main(fn_list, src_dir, dst_dir, scale=1.1, face=True, framesize=112, save_landmark=False, worker=4):
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Pool(processes=worker)
    pbar = tqdm(total=len(fn_list), desc='crop video')
    update = lambda *args: pbar.update()
    for fn in fn_list:
        src_v_fn = os.path.join(src_dir, fn + '.mp4')
        dst_v_fn = os.path.join(dst_dir, fn + '.mp4')
        landmark_fn = os.path.join(src_dir, fn + '_landmark.npy')
        p.apply_async(crop_face, (src_v_fn, dst_v_fn, framesize, scale, face, save_landmark, landmark_fn), callback=update)
    p.close()
    p.join()


if __name__ == '__main__':
    args = parser.parse_args()
    filelist = args.file_list
    src_dir = args.input
    dst_dir = args.output
    worker = args.worker
    save_landmark = args.save_landmark
    face = True if args.type == 'face' else False
    scale = args.scale
    framesize = args.size
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    fn_list = []
    with open(filelist + '-train.txt') as fp:
        fn_list.extend([line.strip() for line in fp.readlines()])
    with open(filelist + '-val.txt') as fp:
        fn_list.extend([line.strip() for line in fp.readlines()])
    with open(filelist + '-test.txt') as fp:
        fn_list.extend([line.strip() for line in fp.readlines()])
    fn_list = sorted(fn_list)
    crop_face_main(fn_list, src_dir, dst_dir, scale, face, framesize, save_landmark, worker)