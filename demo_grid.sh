# download GRID to data/grid/ori_data/
# download GRID_LM.zip to /work102/cchen/VTS/vcagan/data/grid/grid_lm/GRID_LM.zip
# unzip GRID_LM.zip

# Extract
python preprocess/Extract_frames.py --Grid_dir data/grid/ori_data/ --Output_dir data/grid/extracted/data/
stty echo
# Preprocess
TMPDIR=/work102/cchen/tmp/ python preprocess/Preprocess.py --Data_dir data/grid/extracted/data/ --Landmark data/grid/grid_lm/ --FPS 25 --reference preprocess/Ref_face.txt --Output_dir data/grid/processed/data/
stty echo
# train
python train_grid.py --gpu 0 --grid data/grid/processed/ --checkpoint_dir output/checkpoint/grid/demo/ --batch_size 4 --epochs 1 --subject s1 --eval_step 100
stty echo
# test
python test.py --gpu 0 --dataset grid --data_dir data/grid/processed/ --checkpoint_dir output/checkpoint/grid/demo/ --checkpoint output/checkpoint/grid/demo/grid_s1_16000_640_160/Best.ckpt --batch_size 8 --subject s1 --output_dir output/predictions/grid/demo/ --save_mel --save_wav
stty echo