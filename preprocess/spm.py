import sentencepiece as spm

# Finnish
spm.SentencePieceTrainer.train(
        input='/raid/lingo/abau/lm/ud-treebanks-v2.7/UD_Finnish-TDT/fi_tdt-ud-train.txt',
        model_prefix='finnish-spm',
        vocab_size=2 ** 14
)

# English
spm.SentencePieceTrainer.train(
        input='/raid/lingo/abau/lm/training-monolingual/news.2007.en.1m.shuffled',
        model_prefix='english-spm',
        vocab_size=2 ** 14
)
