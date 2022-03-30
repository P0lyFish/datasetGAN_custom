#### Human Face ####

# # Make training data 
# CUDA_VISIBLE_DEVICES=0 python make_training_data_stylegan2.py \
# --exp configs/stylegan2/sg2_face_2.json \
# --sv_path data/stylegan2/annotation/face_2 \
# --num_sample 20000

# ## Manually few images 
# 
# # Train interpreter 
CUDA_VISIBLE_DEVICES=1 python train_interpreter_stylegan2.py  \
--exp "configs/stylegan2/sg2_face_2.json"
# 
# # Generate labeled data 
# CUDA_VISIBLE_DEVICES=2 python train_interpreter_stylegan2.py  \
# --exp "configs/stylegan2/sg2_face_2.json" \
# --resume "model_dir/sg2_face_2" \
# --num_sample 20 \
# --save_vis True  \
# --start_step 0 \
# --generate_data True 

