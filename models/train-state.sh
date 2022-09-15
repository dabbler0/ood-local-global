
CUDA_VISIBLE_DEVICES=6,7 python train.py --data /raid/lingo/abau/lm/training-monolingual/news.2007.en.1m.tokenized --vocab /raid/lingo/abau/lm/vocab.en.1m.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/en.1m.2layer.lr1e6.state_dropout0.3 --layers 2 --lr 1e-6 --state_dropout 0.3 &
