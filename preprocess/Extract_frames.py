import os, glob, subprocess
import argparse
from tqdm import tqdm
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Grid_dir', type=str, default="Data dir to GRID_corpus")
    parser.add_argument("--Output_dir", type=str, default='Output dir Ex) ./GRID_imgs_aud')
    args = parser.parse_args()
    return args

def process_worker(v, nouse=None):
    f_name = os.path.basename(v)
    t, _ = os.path.split(v)
    _, sub_name = os.path.split(t)
    out_im = os.path.join(args.Output_dir, sub_name, 'video', f_name[:-4])
    if len(glob.glob(os.path.join(out_im, '*.png'))) < 75:  # Can resume after killed
        if not os.path.exists(out_im):
            os.makedirs(out_im)
        out_aud = os.path.join(args.Output_dir, sub_name, 'audio')
        if not os.path.exists(out_aud):
            os.makedirs(out_aud)
        subprocess.call(f'ffmpeg -v error -y -i {v} -qscale:v 2 -r 25 {out_im}/%02d.png', shell=True)
        subprocess.call(
            f'ffmpeg -v error -y -i {v} -ac 1 -acodec pcm_s16le -ar 16000 {os.path.join(out_aud, f_name[:-4] + ".wav")}',
            shell=True)

if __name__ == "__main__":

    args = parse_args()
    vid_files = sorted(glob.glob(os.path.join(args.Grid_dir, 'video', '*',
                                            '*.mpg')))  #suppose the directory: Data_dir/subject/video/mpg files
    pbar = tqdm(total=len(vid_files), desc="Extract Frames")
    update = lambda *args: pbar.update()
    p = multiprocessing.Pool(processes=2)
    for k, v in enumerate(vid_files):
        p.apply_async(process_worker, (v, 1), callback=update)
        # process_worker(v)
        # pbar.update()
    p.close()
    p.join()