
export CUDA_VISIBLE_DEVICES=1

python clip_feature.py \
    --image_root ../datasets/amazon18/Images \
    --save_root ../datasets \
    --model_cache_dir ../cache_models/clip \
    --dataset Beauty


