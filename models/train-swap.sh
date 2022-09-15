for n in 0.02 0.04 0.06 0.08
do
  CUDA_VISIBLE_DEVICES=12 python train.py --data /raid/lingo/abau/lm/training-monolingual/news.2007.en.1m.tokenized --vocab /raid/lingo/abau/lm/vocab.en.1m.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/en.1m.2layer.lr1e6.focus.subst_prob$n --layers 2 --lr 1e-4 --subst_prob $n --epochs 100 &
done

for n in 0.1 0.12 0.14 0.16
do
  CUDA_VISIBLE_DEVICES=13 python train.py --data /raid/lingo/abau/lm/training-monolingual/news.2007.en.1m.tokenized --vocab /raid/lingo/abau/lm/vocab.en.1m.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/en.1m.2layer.lr1e6.focus.subst_prob$n --layers 2 --lr 1e-4 --subst_prob $n --epochs 100 &
done
