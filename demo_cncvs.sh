# download data
python data/download_and_process.py -d data/download_temp_dir -j data/s00002/s00002.json -r data/s00002/roi/
mkdir -p data/demo/audio/s00002
mkdir -p data/demo/filelists
mkdir -p data/demo/video/s00002
mv data/download_temp_dir/final/*.wav data/demo/audio/s00002
cp data/s00002/filelists/* data/demo/filelists
# preprocess
python preprocess/landmark.py -f data/demo/filelists/s00002 -i data/download_temp_dir/final/ -o data/demo/video/s00002/ -p preprocess/facelandmarks/faceland.pth --save-landmark
# train
python train_cncvs.py --gpu 0 --dataset cncvs --data_dir data/demo/ --checkpoint_dir output/checkpoint/cncvs/demo/ --batch_size 4 --epochs 1 --subject s00002 --eval_step 100
# test
python test.py --gpu 0 --dataset cncvs --data_dir data/demo/ --checkpoint_dir output/checkpoint/cncvs/demo/ --checkpoint output/checkpoint/cncvs/demo/cncvs_s00002_16000_640_160/Best.ckpt --batch_size 8 --subject s00002 --output_dir output/predictions/cncvs/demo/ --save_mel --save_wav