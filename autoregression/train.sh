export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONPATH=/data/jwna230/peiyu/GenRec/MM-RQVAE/autoregression:$PYTHONPATH

python -m src.train experiment=tiger_train_flat \
    data_dir=../datasets/Beauty \
    semantic_id_path=../datasets/Beauty/codebook.pt \
    num_hierarchies=4