#!/bin/bash
#SBATCH --job-name=hptuning2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/Advanced_NLP_course_HW2
# Verify working directory
echo $(pwd)

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate cpu

python3 train-torch.py \
    -u 128,64 \
    -l 8e-2 \
    -f 200 \
    -b 64 \
    -e 10 \
    -E glove.6B.50d.txt \
    -i datasets/custom_dataset/products.train.train.txt \
    -o torch.products.model \
    --dev_text datasets/custom_dataset/products.train.dev.txt \
    --dev_labels datasets/custom_dataset/products.train.dev_labels.txt \
    --test_text datasets/custom_dataset/products.train.test.txt \
    --test_labels datasets/custom_dataset/products.train.test_labels.txt \
    --wandb_comment "torch_products"


python3 train-torch.py \
    -u 64,32 \
    -l 1e-5 \
    -f 400 \
    -b 16 \
    -e 50 \
    -E glove.6B.50d.txt \
    -i datasets/custom_dataset/4dim.train.train.txt \
    -o torch.4dim.model \
    --dev_text datasets/custom_dataset/4dim.train.dev.txt \
    --dev_labels datasets/custom_dataset/4dim.train.dev_labels.txt \
    --test_text datasets/custom_dataset/4dim.train.test.txt \
    --test_labels datasets/custom_dataset/4dim.train.test_labels.txt \
    --wandb_comment "torch_4dim"



python3 train-torch.py \
    -u 128,64 \
    -l 8e-2 \
    -f 25 \
    -b 32 \
    -e 20 \
    -E ufvytar.100d.txt \
    -i datasets/custom_dataset/questions.train.train.txt \
    -o torch.questions.model \
    --dev_text datasets/custom_dataset/questions.train.dev.txt \
    --dev_labels datasets/custom_dataset/questions.train.dev_labels.txt \
    --test_text datasets/custom_dataset/questions.train.test.txt \
    --test_labels datasets/custom_dataset/questions.train.test_labels.txt \
    --wandb_comment "torch_questions"



python3 train-torch.py \
    -u 128,64 \
    -l 5e-2 \
    -f 20 \
    -b 32 \
    -e 10 \
    -E fasttext.wiki.300d.vec \
    -i datasets/custom_dataset/odia.train.train.txt \
    -o torch.odia.model \
    --dev_text datasets/custom_dataset/odia.train.dev.txt \
    --dev_labels datasets/custom_dataset/odia.train.dev_labels.txt \
    --test_text datasets/custom_dataset/odia.train.test.txt \
    --test_labels datasets/custom_dataset/odia.train.test_labels.txt \
    --wandb_comment "torch_odia"


conda deactivate

