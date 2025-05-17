CUDA_VISIBLE_DEVICES=0 python run_wise_editing.py \
  --editing_method=WISE \
  --hparams_dir=../hparams/WISE/llama-3-8b.yaml \
  --data_dir=../data/wise \
  --ds_size=100 \
  --data_type=ZsRE \
  --sequential_edit


#CUDA_VISIBLE_DEVICES=0 python run_wise_editing.py \
#   --editing_method=WISE \
#   --hparams_dir=../hparams/WISE/llama-7b \
#   --data_dir=../data/wise \
#   --ds_size=3 \
#   --data_type=temporal \
#   --sequential_edit

WISE-Retrieve
#python run_wise_editing.py \
#  --editing_method=WISE \
#  --hparams_dir=../hparams/WISE/llama-7b \
#  --data_dir=../data/wise \
#  --ds_size=3 \
#  --data_type=hallucination
##  --sequential_edit