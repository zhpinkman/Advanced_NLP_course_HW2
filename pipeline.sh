

python3 train.py \
    -u 256 \
    -l 0.001 \
    -f 128 \
    -b 32 \
    -e 10 \
    -E glove.6B.50d.txt \
    -i datasets/custom_dataset/products.train.train.txt \
    -o products.model






# python3 train.py \
#     -u 256 \
#     -l 0.0001 \
#     -f 30 \
#     -b 16 \
#     -e 10 \
#     -E fasttext.wiki.300d.vec \
#     -i datasets/custom_dataset/odia.train.train.txt \
#     -o odiya.model