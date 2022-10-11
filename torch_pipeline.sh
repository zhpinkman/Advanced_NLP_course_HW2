echo "Training torch Neural Network on Products dataset"
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

echo "Training torch Neural Network on 4dim dataset"
python3 train-torch.py \
    -u 64,32 \
    -l 2e-2 \
    -f 200 \
    -b 16 \
    -e 100 \
    -E glove.6B.50d.txt \
    -i datasets/custom_dataset/4dim.train.train.txt \
    -o torch.4dim.model \
    --dev_text datasets/custom_dataset/4dim.train.dev.txt \
    --dev_labels datasets/custom_dataset/4dim.train.dev_labels.txt \
    --test_text datasets/custom_dataset/4dim.train.test.txt \
    --test_labels datasets/custom_dataset/4dim.train.test_labels.txt \
    --wandb_comment "torch_4dim"


echo "Training torch Neural Network on questions dataset"
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


echo "Training torch Neural Network on odia dataset"
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


echo "Evaluation of torch Neural Network on Products dataset"
python3 classify-torch.py \
    -m torch.products.model \
    -i datasets/custom_dataset/products.train.test.txt \
    -o tmp/torch.products.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/products.train.test_labels.txt \
    --predictions tmp/torch.products.predictions \
    --output_file tmp/torch.products.results

echo "Evaluation of torch Neural Network on 4dim dataset"
python3 classify-torch.py \
    -m torch.4dim.model \
    -i datasets/custom_dataset/4dim.train.test.txt \
    -o tmp/torch.4dim.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/4dim.train.test_labels.txt \
    --predictions tmp/torch.4dim.predictions \
    --output_file tmp/torch.4dim.results

echo "Evaluation of torch Neural Network on questions dataset"
python3 classify-torch.py \
    -m torch.questions.model \
    -i datasets/custom_dataset/questions.train.test.txt \
    -o tmp/torch.questions.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/questions.train.test_labels.txt \
    --predictions tmp/torch.questions.predictions \
    --output_file tmp/torch.questions.results

echo "Evaluation of torch Neural Network on odia dataset"
python3 classify-torch.py \
    -m torch.odia.model \
    -i datasets/custom_dataset/odia.train.test.txt \
    -o tmp/torch.odia.predictions

python3 evaluate.py \
    --labels datasets/custom_dataset/odia.train.test_labels.txt \
    --predictions tmp/torch.odia.predictions \
    --output_file tmp/torch.odia.results