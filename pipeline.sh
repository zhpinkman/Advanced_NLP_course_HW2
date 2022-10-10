
# echo "Training simple Neural Network on Products dataset"
# python3 train.py \
#     -u 128,64 \
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


# echo "Training simple Neural Network on 4dim dataset"
# python3 train.py \
#     -u 256,128,64 \
#     -l 1e-1 \
#     -f 300 \
#     -b 16 \
#     -e 70 \
#     -E glove.6B.50d.txt \
#     -i datasets/custom_dataset/4dim.train.train.txt \
#     -o 4dim.model \
#     --dev_text datasets/custom_dataset/4dim.train.dev.txt \
#     --dev_labels datasets/custom_dataset/4dim.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/4dim.train.test.txt \
#     --test_labels datasets/custom_dataset/4dim.train.test_labels.txt \
#     --wandb_comment 4dim


# echo "Training simple Neural Network on questions dataset"
# python3 train.py \
#     -u 128,64 \
#     -l 8e-2 \
#     -f 25 \
#     -b 32 \
#     -e 10 \
#     -E ufvytar.100d.txt \
#     -i datasets/custom_dataset/questions.train.train.txt \
#     -o questions.model \
#     --dev_text datasets/custom_dataset/questions.train.dev.txt \
#     --dev_labels datasets/custom_dataset/questions.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/questions.train.test.txt \
#     --test_labels datasets/custom_dataset/questions.train.test_labels.txt \
#     --wandb_comment questions


# echo "Training simple Neural Network on odia dataset"
# python3 train.py \
#     -u 128,64 \
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


echo "Evaluation of simple Neural Network on Products dataset"
python3 classify.py \
    -m products.model \
    -i datasets/custom_dataset/products.train.test.txt \
    -o tmp/products.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/products.train.test_labels.txt \
    --predictions tmp/products.predictions \
    --output_file tmp/products.results

echo "Evaluation of simple Neural Network on 4dim dataset"
python3 classify.py \
    -m 4dim.model \
    -i datasets/custom_dataset/4dim.train.test.txt \
    -o tmp/4dim.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/4dim.train.test_labels.txt \
    --predictions tmp/4dim.predictions \
    --output_file tmp/4dim.results

echo "Evaluation of simple Neural Network on questions dataset"
python3 classify.py \
    -m questions.model \
    -i datasets/custom_dataset/questions.train.test.txt \
    -o tmp/questions.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/questions.train.test_labels.txt \
    --predictions tmp/questions.predictions \
    --output_file tmp/questions.results

echo "Evaluation of simple Neural Network on odia dataset"
python3 classify.py \
    -m odia.model \
    -i datasets/custom_dataset/odia.train.test.txt \
    -o tmp/odia.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/odia.train.test_labels.txt \
    --predictions tmp/odia.predictions \
    --output_file tmp/odia.results