import argparse
import multiprocessing
import os
import dlib
import cv2
from tqdm import tqdm
import scipy.signal
from pathlib import Path
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file-list', required=True, help='filelist dir contains {filelist}-test.txt')
parser.add_argument('-i', '--input', required=True, help='mp4 data path')
parser.add_argument('-o', '--output', required=True, help='target mp4 data path')
parser.add_argument('-w', '--worker', default=4, type=int, help='size of multiprocessing pool')
parser.add_argument('--size', default=224, type=int, help='output mp4 file frame size')
parser.add_argument('--save-rois', default=False, action='store_true', help='save tracker result or not')

def smooth_sequence(seq: list) -> list:
    win_len = min(len(seq), 21)
    if win_len % 2 == 0:
        win_len -= 1
    seq = scipy.signal.savgol_filter(seq, window_length=win_len, polyorder=1)
    # step_len = 1
    # tolerence = 1
    # last = 0
    # remove jitter
    # for i in range(len(seq)):
    #     if abs(seq[i] - seq[last]) > tolerence:
    #         last = i - step_len if last < i - step_len else last
    #         step = (seq[i] - seq[last]) / (i - last)
    #         start = seq[last]
    #         for j in range(last, i):
    #             seq[j] = round(start + (j - last) * step)
    #         last = i
    return seq

def get_smoothed_head_roi_from_tracker_res(tracker_res: list) -> list:
    '''
    description: get a smoothed tracker positions
    param {list} tracker_res  list of tuples (l,r,t,b)
    return {list} [(l,r,t,b) of per frame]
    '''
    x_size = max([pos[1] for pos in tracker_res]) - min([pos[0] for pos in tracker_res])
    y_size = max([pos[3] for pos in tracker_res]) - min([pos[2] for pos in tracker_res])
    ls = [pos[0] for pos in tracker_res]
    rs = [pos[1] for pos in tracker_res]
    ts = [pos[2] for pos in tracker_res]
    bs = [pos[3] for pos in tracker_res]
    cxs = [round((ls[i] + rs[i]) / 2) for i in range(len(tracker_res))]
    cys = [round((ts[i] + bs[i]) / 2) for i in range(len(tracker_res))]
    cxs = smooth_sequence(cxs)
    cys = smooth_sequence(cys)
    centers = [(cxs[i], cys[i]) for i in range(len(cxs))]
    roi = [(
        int(center[0] - x_size / 2),
        int(center[0] + x_size / 2),
        int(center[1] - y_size / 2),
        int(center[1] + y_size / 2),
    ) for center in centers]
    return roi



def adjust_roi_position(roi: tuple, frameSize: tuple) -> tuple:
    '''
    description: shift roi to ensure roi is totally contained by frame
            if roi is too large, reset roi to size of frame
    param {tuple}  roi
    param {tuple}  frameSize
    return {tuple} adjusted roi
    '''
    l, r, t, b = roi
    if frameSize[0] < r - l or frameSize[1] < b - t:
        return (0, frameSize[0], 0, frameSize[1])
    if l < 0:
        r = r - l
        l = 0
    if r > frameSize[0]:
        l = l - (r - frameSize[0])
        r = frameSize[0]
    if t < 0:
        b = b - t
        t = 0
    if b > frameSize[1]:
        t = t - (b - frameSize[1])
        b = frameSize[1]
    return (l, r, t, b)

def adjust_head_roi_to_fit_frame(head_roi: tuple, frameSize: tuple) -> tuple:
    '''
    description: select a square area in frame which contains the whole face 
        1. calculate head roi center point
        2. get max edge length of head roi
        3. select a square area which contains the whole face
    param {tuple}  head_roi  (l,r,t,b)
    param {tuple}  frameSize (width, height) from cv2.CAP_PROP_FRAME_WIDTH / HEIGHT
    return {tuple} adjusted head roi(l,r,t,b)
    '''
    # 1. calculate head roi center point
    head_center = (
        int((head_roi[0] + head_roi[1]) / 2),
        int((head_roi[2] + head_roi[3]) / 2),
    )
    # 2. get max edge length of head roi
    head_size = (
        int(head_roi[1] - head_roi[0]),
        int(head_roi[3] - head_roi[2]),
    )
    max_edge = max([head_size[0], head_size[1]])
    min_edge = min([head_size[0], head_size[1]])
    # 3. select a square area which contains the whole face
    rectan = 1.2 * max_edge
    if rectan > frameSize[0] or rectan > frameSize[1]:
        rectan = min([frameSize[0], frameSize[1]])
    l = int(head_center[0] - rectan / 2)
    r = int(l + rectan)
    t = int(head_center[1] - rectan / 2)
    b = int(t + rectan)
    l, r, t, b = adjust_roi_position((l, r, t, b), frameSize)
    return (l, r, t, b)

def get_smoothed_head_roi_frames(video, head_roi: list, frame_boundary: tuple, target_size: tuple) -> list:
    '''
    description: get crops of same size (each contain head)
        1. calculate head roi center point
        2. get max edge length of head roi
        3. select a square area which contains the whole face
        4. get crop images from video, resize the square area to target size
    param {*}     video          cv2.VideoCapture() instance
    param {list}  head_roi       [(l, r, t, b)] list of rectangle contains head area
    param {tuple} frame_boundary (start frame index, end frame index) clip time boundaries in video
    param {tuple} target_size    (weight, height) [is square] when capture head roi, resize to this size
    return {list} list of crop images
    '''
    assert target_size[0] == target_size[1], f'shape is not square {target_size[0]}, {target_size[1]}'
    target_size = target_size[0]
    frameSize = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    imgs = []
    rois = []
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    for idx in range(frame_boundary[0], frame_boundary[1]):
        l, r, t, b = adjust_head_roi_to_fit_frame(head_roi[idx - frame_boundary[0]], frameSize)
        success, frame = video.read()
        if success:
            head = frame[t:b, l:r, :]
            rois.append((l,r,t,b))
            head = cv2.resize(head, (target_size, target_size))
            imgs.append(head)
        else:
            return [], []
    return imgs, rois

def write_video(output_file: str, frames: list, frameSize: tuple) -> None:
    '''
    description: write frames to mp4 file 
    param {str}   output_file target output mp4 file
    param {list}  frames      list of frame images, same size with frameSize 
    param {tuple} frameSize   (weight, height)
    return {*}
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_file,
        fourcc=fourcc,
        fps=25,
        frameSize=(int(frameSize[0]), int(frameSize[1])),
        isColor=True,
    )
    for frame in frames:
        writer.write(frame)
    writer.release()

def face_tracker(detector, src_v_fn, dst_v_fn, framesize, save_rois, rois_fn):
    res = []
    video = cv2.VideoCapture(src_v_fn)
    max_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = dlib.correlation_tracker()
    success, first_frame = video.read()
    if not success:
        print(f'can not read {src_v_fn}')
        return
    first_dets = detector(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), 1)
    if len(first_dets) < 1:
        print(f'can not detect face in {src_v_fn}')
    tracker.start_track(first_frame, first_dets[0])
    position = tracker.get_position()
    pos = (position.left(), position.right(), position.top(), position.bottom())
    res.append(pos)
    for i in range(1, max_frame):
        success, frame = video.read()
        if success:
            tracker.update(frame)
            position = tracker.get_position()
            pos = (position.left(), position.right(), position.top(), position.bottom())
        else:
            # print('read error')
            pass
        res.append(pos)
    head_roi = get_smoothed_head_roi_from_tracker_res(res)
    images, rois = get_smoothed_head_roi_frames(video, head_roi, (0, max_frame), target_size=(framesize,framesize))
    if save_rois:
        Path(os.path.dirname(rois_fn)).mkdir(exist_ok=True, parents=True)
        np.save(rois_fn, rois)
    # print(dst_v_fn)
    Path(os.path.dirname(dst_v_fn)).mkdir(exist_ok=True, parents=True)
    write_video(dst_v_fn, images, frameSize=(framesize, framesize))

def detection_main(fn_list, src_dir, dst_dir, framesize, save_rois, worker):
    detector = dlib.get_frontal_face_detector()
    p = multiprocessing.Pool(processes=worker)
    pbar = tqdm(total=len(fn_list), desc='track face')
    update = lambda *args: pbar.update()
    for fn in fn_list:
        src_v_fn = os.path.join(src_dir, fn + '.mp4')
        dst_v_fn = os.path.join(dst_dir, fn + '.mp4')
        rois_fn = os.path.join(src_dir.replace('video','rois'), fn + '_rois.npy')
        p.apply_async(face_tracker, (detector, src_v_fn, dst_v_fn, framesize, save_rois, rois_fn), callback=update)
        # face_tracker(detector, src_v_fn, dst_v_fn, framesize, save_rois, rois_fn)
    p.close()
    p.join()

    pass

if __name__ == '__main__':
    args = parser.parse_args()
    filelist = args.file_list
    src_dir = args.input
    dst_dir = args.output
    worker = args.worker
    save_rois = args.save_rois
    framesize = args.size
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    fn_list = []
    with open(filelist + '-train.txt') as fp:
        fn_list.extend([line.strip() for line in fp.readlines()])
    with open(filelist + '-val.txt') as fp:
        fn_list.extend([line.strip() for line in fp.readlines()])
    with open(filelist + '-test.txt') as fp:
        fn_list.extend([line.strip() for line in fp.readlines()])
    # new_fn = []
    # for fn in fn_list:
    #     if not os.path.exists(os.path.join(dst_dir, fn + '.mp4')):
    #         # print(fn)
    #         new_fn.append(fn)
    # fn_list = new_fn
    fn_list = sorted(fn_list)
    # print(fn_list)
    detection_main(fn_list, src_dir, dst_dir, framesize, save_rois, worker)