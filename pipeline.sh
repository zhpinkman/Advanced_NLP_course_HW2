
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
#     -u 64,32 \
#     -l 1e-2 \
#     -f 200 \
#     -w 0.001 \
#     -b 16 \
#     -e 60 \
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


# echo "Training simple Neural Network on odiya dataset"
# python3 train.py \
#     -u 128,64 \
#     -l 5e-2 \
#     -f 20 \
#     -b 32 \
#     -e 10 \
#     -E fasttext.wiki.300d.vec \
#     -i datasets/custom_dataset/odiya.train.train.txt \
#     -o odiya.model \
#     --dev_text datasets/custom_dataset/odiya.train.dev.txt \
#     --dev_labels datasets/custom_dataset/odiya.train.dev_labels.txt \
#     --test_text datasets/custom_dataset/odiya.train.test.txt \
#     --test_labels datasets/custom_dataset/odiya.train.test_labels.txt \
#     --wandb_comment odiya

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

cat tmp/products.results

echo "Evaluation of simple Neural Network on 4dim dataset"
python3 classify.py \
    -m 4dim.model \
    -i datasets/custom_dataset/4dim.train.test.txt \
    -o tmp/4dim.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/4dim.train.test_labels.txt \
    --predictions tmp/4dim.predictions \
    --output_file tmp/4dim.results

cat tmp/4dim.results

echo "Evaluation of simple Neural Network on questions dataset"
python3 classify.py \
    -m questions.model \
    -i datasets/custom_dataset/questions.train.test.txt \
    -o tmp/questions.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/questions.train.test_labels.txt \
    --predictions tmp/questions.predictions \
    --output_file tmp/questions.results

cat tmp/questions.results

echo "Evaluation of simple Neural Network on odiya dataset"
python3 classify.py \
    -m odiya.model \
    -i datasets/custom_dataset/odiya.train.test.txt \
    -o tmp/odiya.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/odiya.train.test_labels.txt \
    --predictions tmp/odiya.predictions \
    --output_file tmp/odiya.results

cat tmp/odiya.results