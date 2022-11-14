# first, download cmlr dataset to data/cmlr/ori_data

# face detect on origin video
python preprocess/face_detection.py -f data/cmlr/processed/filelists/s1 -i data/cmlr/ori_data/video/ -o data/cmlr/processed/face/ --save-rois -w 10
# extract face landrmark to extract lip roi
python preprocess/landmark.py -f data/cmlr/processed/filelists/s1 -i data/cmlr/processed/face/ -o data/cmlr/processed/video/ --save-landmark -t lip -w 2
# cp audio files
cp -r data/cmlr/ori_data/audio/* data/cmlr/processed/audio/
# train
python train_cmlr.py --gpu 0 --dataset cmlr --data_dir data/cmlr/processed/ --checkpoint_dir output/checkpoint/cmlr/demo/ --batch_size 4 --epochs 1 --subject s1 --eval_step 100
# test
python test.py --gpu 0 --dataset cmlr --data_dir data/cmlr/processed/ --checkpoint_dir output/checkpoint/cmlr/demo/ --checkpoint output/checkpoint/cmlr/demo/cmlr_s1_16000_640_160/Best.ckpt --batch_size 8 --subject s1 --output_dir output/predictions/cmlr/demo/ --save_mel --save_wav