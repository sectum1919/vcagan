'''
Author: chenchen2121 c-c14@tsinghua.org.cn
Date: 2022-09-07 15:52:17
LastEditors: chenchen c-c14@tsinghua.org.cn
LastEditTime: 2022-11-03 05:06:32
Description: 

'''
import os
import json
import argparse
import logging
import time
from pathlib import Path
import multiprocessing
import math
import subprocess
import cv2
import soundfile

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', required=True, help='download & transcode files in this path')
parser.add_argument('-j', '--metadata', required=True, help='given json file')
parser.add_argument('-r', '--roi_dir', required=True, help='given roi json file folder')
parser.add_argument('-f', '--frame_width', default=224, type=int, help='size of output video')
parser.add_argument('-w', '--worker', default=2, type=int, help='size of multiprocessing pool')
parser.add_argument('-l', '--loglevel', default=0, type=int, choices=[0, 1, 2, 3, 4], help='logger output level')
parser.add_argument('--save_origin_video',
                    default=False,
                    action='store_true',
                    help='save or remove origin video file after transcoding')
parser.add_argument('--save_transcoded_video',
                    default=False,
                    action='store_true',
                    help='save or remove transcoded video file after processing')

loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]


def write_video(output_file: str, frames: list, frame_size: tuple) -> None:
    '''
    description: write frames to mp4 file 
    param {str}   output_file target output mp4 file
    param {list}  frames      list of frame images, same size with frame_size 
    param {tuple} frame_size  (weight, height)
    return {*}
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_file,
        fourcc=fourcc,
        fps=25,
        frameSize=(int(frame_size[0]), int(frame_size[1])),
        isColor=True,
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def split_audio(src: str, ss_t_list: list, dst_list: list) -> None:
    audio, sr = soundfile.read(src)
    for ss_t, dst in ss_t_list, dst_list:
        start_idx = int(ss_t[0] * sr)
        end_idx = int(ss_t[0] * sr + ss_t[1] * sr)
        soundfile.write(dst, audio[start_idx:end_idx], sr)

def write_audio(audio, sr, ss: float, t: float, dst) -> None:
    start_idx = int(ss * sr)
    end_idx = int(ss * sr + t * sr)
    soundfile.write(dst, audio[start_idx:end_idx], sr)

def read_video_inner(video, frame_boundary: tuple, rois: list, frame_size: tuple = (224, 224)) -> list:
    imgs = []
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    for idx in range(frame_boundary[0], frame_boundary[1]):
        l, r, t, b = rois[idx - frame_boundary[0]]
        success, frame = video.read()
        if success:
            head = frame[t:b, l:r, :]
            head = cv2.resize(head, frame_size)
            imgs.append(head)
        else:
            return []
    return imgs

def read_video(src: str, frame_boundary: tuple, rois: list, frame_size: tuple = (224, 224)) -> list:
    imgs = []
    video = cv2.VideoCapture(src)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    for idx in range(frame_boundary[0], frame_boundary[1]):
        l, r, t, b = rois[idx - frame_boundary[0]]
        success, frame = video.read()
        if success:
            head = frame[t:b, l:r, :]
            head = cv2.resize(head, frame_size)
            imgs.append(head)
        else:
            return []
    return imgs


def trancode_worker(meta, src_dir, dst_dir, save_ori):
    print('transcode')
    src = os.path.join(src_dir, f'{meta["spkid"]}_{meta["videoid"]}.mp4')
    dst = os.path.join(dst_dir, f'{meta["spkid"]}_{meta["videoid"]}.mp4')
    if os.path.exists(dst):
        logger.debug(f'{dst} exists')
        return 0
    else:
        get_fps = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "
        cmd = f"{get_fps} {src}"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if str(res.stdout[:-1], 'utf-8').split('/') == ['25', '1']:
            cmd = f"cp {src} {dst}"
            res = subprocess.call(cmd, shell=True)
        else:
            cmd = f'ffmpeg -v error -i {src} -qscale 0 -r 25 -y {dst}'
            logging.debug(cmd)
            res = subprocess.call(cmd, shell=True)
        if res != 0:
            logging.warning(f'error occur when transcoding {src}')
            if os.path.exists(dst):
                os.remove(dst)
        else:
            logging.debug(f'finish transcoding {dst}')
    if not save_ori and os.path.exists(src):
        os.remove(src)
    return res


def process_worker(meta, roi_dir, src_dir, dst_dir, frame_size, save_ori):
    print('process')
    try:
        src = os.path.join(src_dir, f'{meta["spkid"]}_{meta["videoid"]}.mp4')
        src_audio = os.path.join(src_dir, f'{meta["spkid"]}_{meta["videoid"]}.wav')
        cmd = f'ffmpeg -v error -y -accurate_seek -i {src} -avoid_negative_ts 1 -b:a 256k -ar 16000 -ac 1 -acodec pcm_s16le -strict -2 {src_audio}'
        logger.debug(cmd)
        subprocess.call(cmd, shell=True)
        audio, sr = soundfile.read(src_audio)
        video = cv2.VideoCapture(src)
        for _, utt_info in enumerate(meta['uttlist']):
            uid = utt_info['uttid']
            af = os.path.join(dst_dir, f'{meta["spkid"]}_{meta["videoid"]}_{uid}.wav')
            vf = os.path.join(dst_dir, f'{meta["spkid"]}_{meta["videoid"]}_{uid}.mp4')
            # extract audio
            ss = float(utt_info['ss'])
            t = float(utt_info['t'])
            write_audio(audio, sr, ss, t, af)
            # extract silent video
            rois = json.load(open(os.path.join(roi_dir, f'{meta["spkid"]}_{meta["videoid"]}_{uid}.json')))
            frames = read_video_inner(video, (utt_info['video_frame_ss'], utt_info['video_frame_ed']), rois, frame_size=frame_size)
            write_video(vf, frames, frame_size=frame_size)
            logger.debug(f'write_video({vf}, frames, frame_size={frame_size})')
        if not save_ori and os.path.exists(src):
            os.remove(src)
            os.remove(src_audio)
    except Exception as e:
        logger.exception(e)


def download_worker(meta, dst_dir):
    print('download')
    url = meta['videourl']
    save_name = f'{meta["spkid"]}_{meta["videoid"]}'
    try:
        if not os.path.exists(os.path.join(dst_dir, f"{save_name}.mp4")):
            print('downloading' + os.path.join(dst_dir, f"{save_name}.mp4"))
            cmd = f'you-get "{url}" -o "{dst_dir}" -O "{save_name}"'
            res = subprocess.call(cmd, shell=True)
        else:
            print(os.path.join(dst_dir, f"{save_name}.mp4") + 'exist')
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def pipeline(meta, rpath, dpath, tpath, fpath, frame_size, save_download, save_transcoded):
    res = download_worker(meta, dpath)
    if res < 0:
        return res
    res = trancode_worker(meta, dpath, tpath, save_download)
    if res < 0:
        return res
    process_worker(meta, rpath,  tpath, fpath, frame_size, save_transcoded)


def main(metadata, rpath, dpath, tpath, fpath, frame_size, worker, save_origin_video, save_transcoded_video):

    pool = multiprocessing.Pool(processes=worker)
    for meta in metadata:
        # pipeline(meta, dpath, tpath, fpath, save_origin_video, save_transcoded_video)
        pool.apply_async(pipeline, (meta, rpath, dpath, tpath, fpath, frame_size, save_origin_video, save_transcoded_video))

    pool.close()
    pool.join()


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    worker = args.worker
    jsonfile = args.metadata
    roi_dir = args.roi_dir
    frame_width = args.frame_width
    frame_size = (frame_width, frame_width)
    try:
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        logging.basicConfig(
            level=loglevel_list[args.loglevel],
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s %(message)s',
            filename=os.path.join(data_dir, f'logs_{cur_time}.log'),
            filemode='a',
        )
        logger = logging.getLogger(__file__)
    except Exception as e:
        print(e)
        print("can't init logger")
    try:
        # load metadata from json file
        if not os.path.exists(jsonfile):
            logger.critical('metadata file not exist')
            exit(-1)
        metadata = json.load(open(jsonfile, 'r'))
        # build folder
        download_data_dir = os.path.join(data_dir, 'download')
        transcoded_data_dir = os.path.join(data_dir, 'transcoded')
        final_data_dir = os.path.join(data_dir, 'final')
        Path(download_data_dir).mkdir(exist_ok=True, parents=True)
        Path(transcoded_data_dir).mkdir(exist_ok=True, parents=True)
        Path(final_data_dir).mkdir(exist_ok=True, parents=True)
        # do download and transcode and process
        main(
            metadata,
            roi_dir,
            download_data_dir,
            transcoded_data_dir,
            final_data_dir,
            frame_size,
            worker,
            args.save_origin_video,
            args.save_transcoded_video,
        )

    except Exception as e:
        logger.exception(e)
