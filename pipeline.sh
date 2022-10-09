# python3 train.py \
#     -u 128 \
#     -s 64 \
#     -l 8e-2 \
#     -f 200 \
#     -b 64 \
#     -e 10 \
#     -E glove.6B.50d.txt \
#     -i datasets/custom_dataset/products.train.train.txt \
#     -o products.model \
#     --dev_text datasets/custom_dataset/products.train.dev.txt \
#     --dev_labels datasets/custom_dataset/products.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/products.train.test.txt \
#     --test_labels datasets/custom_dataset/products.train.test_labels.txt \
#     --wandb_comment products


# python3 train.py \
#     -u 128 \
#     -s 64 \
#     -l 1e-3 \
#     -f 300 \
#     -b 64 \
#     -e 20 \
#     -E glove.6B.50d.txt \
#     -i datasets/custom_dataset/4dim.train.train.txt \
#     -o 4dim.model \
#     --dev_text datasets/custom_dataset/4dim.train.dev.txt \
#     --dev_labels datasets/custom_dataset/4dim.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/4dim.train.test.txt \
#     --test_labels datasets/custom_dataset/4dim.train.test_labels.txt \
#     --wandb_comment 4dim



python3 train.py \
    -u 128 \
    -s 64 \
    -l 8e-2 \
    -f 25 \
    -b 32 \
    -e 10 \
    -E ufvytar.100d.txt \
    -i datasets/custom_dataset/questions.train.train.txt \
    -o questions.model \
    --dev_text datasets/custom_dataset/questions.train.dev.txt \
    --dev_labels datasets/custom_dataset/questions.train.dev_labels.txt \
    --test_text datasets/custom_dataset/questions.train.test.txt \
    --test_labels datasets/custom_dataset/questions.train.test_labels.txt \
    --wandb_comment questions



# python3 train.py \
#     -u 128 \
#     -s 64 \
#     -l 5e-2 \
#     -f 20 \
#     -b 32 \
#     -e 10 \
#     -E fasttext.wiki.300d.vec \
#     -i datasets/custom_dataset/odia.train.train.txt \
#     -o odia.model \
#     --dev_text datasets/custom_dataset/odia.train.dev.txt \
#     --dev_labels datasets/custom_dataset/odia.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/odia.train.test.txt \
#     --test_labels datasets/custom_dataset/odia.train.test_labels.txt \
#     --wandb_comment odia
