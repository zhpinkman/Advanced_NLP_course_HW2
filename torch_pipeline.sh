# python3 train-torch.py \
#     -u 128 \
#     -l 1e-4 \
#     -f 128 \
#     -b 64 \
#     -e 10 \
#     -E glove.6B.50d.txt \
#     -i datasets/custom_dataset/products.train.train.txt \
#     -o torch.products.model \
#     --dev_text datasets/custom_dataset/products.train.dev.txt \
#     --dev_labels datasets/custom_dataset/products.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/products.train.test.txt \
#     --test_labels datasets/custom_dataset/products.train.test_labels.txt \
#     --wandb_comment "torch products"

# Works TODO: but have to improve the results. They're pretty poor 
python3 train-torch.py \
    -u 256 \
    -l 5e-4 \
    -f 300 \
    -b 32 \
    -e 40 \
    -E glove.6B.50d.txt \
    -i datasets/custom_dataset/4dim.train.train.txt \
    -o torch.4dim.model \
    --dev_text datasets/custom_dataset/4dim.train.dev.txt \
    --dev_labels datasets/custom_dataset/4dim.train.dev_labels.txt \
    --test_text datasets/custom_dataset/4dim.train.test.txt \
    --test_labels datasets/custom_dataset/4dim.train.test_labels.txt \
    --wandb_comment "torch 4dim"

# Doesn't work, because of the embedding file, FIXME: so have to fix the reader

# python3 train-torch.py \
#     -u 128 \
#     -l 1e-4 \
#     -f 128 \
#     -b 64 \
#     -e 10 \
#     -E ufvytar.100d.txt \
#     -i datasets/custom_dataset/questions.train.train.txt \
#     -o torch.questions.model \
#     --dev_text datasets/custom_dataset/questions.train.dev.txt \
#     --dev_labels datasets/custom_dataset/questions.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/questions.train.test.txt \
#     --test_labels datasets/custom_dataset/questions.train.test_labels.txt \
#     --wandb_comment "torch questions"



# Works but have to make it better a bit. 

# python3 train-torch.py \
#     -u 256 \
#     -l 1e-4 \
#     -f 30 \
#     -b 32 \
#     -e 10 \
#     -E fasttext.wiki.300d.vec \
#     -i datasets/custom_dataset/odia.train.train.txt \
#     -o torch.odia.model \
#     --dev_text datasets/custom_dataset/odia.train.dev.txt \
#     --dev_labels datasets/custom_dataset/odia.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/odia.train.test.txt \
#     --test_labels datasets/custom_dataset/odia.train.test_labels.txt \
#     --wandb_comment "torch odia"
