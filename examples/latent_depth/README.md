# Deep Transformers with Latent Depth (Li et al., 2020)

[https://arxiv.org/abs/2009.13102](https://arxiv.org/abs/2009.13102).

## Introduction

We present a probabilistic framework to automatically learn which layer(s) to use by learning the posterior
distributions of layer selection. As an extension of this framework, we propose a novel method to train one shared
Transformer network for multilingual machine translation with different layer selection posteriors for each language
pair.

## Training a multilingual model with latent depth

Below is an example of training with latent depth in decoder for one-to-many (O2M) related languages. We use the same
preprocessed (numberized and binarized) TED8 dataset as
in [Balancing Training for Multilingual Neural Machine Translation (Wang et al., 2020)](https://github.com/cindyxinyiwang/multiDDS)
, which could be generated
by [the script](https://github.com/cindyxinyiwang/multiDDS/blob/multiDDS/util_scripts/prepare_multilingual_data.sh) the
author provided.

```bash
lang_pairs_str="eng-aze,eng-bel,eng-ces,eng-glg,eng-por,eng-rus,eng-slk,eng-tur"
databin_dir=<path to binarized data>

fairseq-train ${databin_dir} \
  --user-dir examples/latent_depth/latent_depth_src \
  --lang-pairs "${lang_pairs_str}" \
  --arch multilingual_transformer_iwslt_de_en \
  --task multilingual_translation_latent_depth \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --share-encoders \
  --share-decoders \
  --decoder-langtok \
  --share-decoder-input-output-embed \
  --dropout 0.3 --attention-dropout 0.3 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --max-tokens 4096 --update-freq 1  \
  --lr 0.0015 \
  --clip-norm 1.0 \
  --seed 2 \
  --ddp-backend=legacy_ddp \
  --encoder-layers 12 \
  --decoder-layers 24 \
  --decoder-latent-layer \
  --sparsity-weight 0.1 \
  --anneal-updates 5000 \
  --soft-update 500  \
  --target-layers 12 \
  --share-weight 0.1
```

## Inference command

```bash
lang_pairs_str="eng-aze,eng-bel,eng-ces,eng-glg,eng-por,eng-rus,eng-slk,eng-tur"
databin_dir=<path to binarized data>
model_path=<path to checkpoint>
src_lang=<source language to translate from>
tgt_lang=<target language to translate to>
gen_data=<name of data split, e.g. valid, test, etc>

fairseq-generate ${databin_dir} \
  --path ${model_path} \
  --task multilingual_translation_latent_depth \
  --decoder-latent-layer \
  --lang-pairs "${lang_pairs_str}" \
  -s ${src_lang} -t ${tgt_lang} \
  --gen-subset $gen_data \
  --scoring sacrebleu \
  --remove-bpe 'sentencepiece' \
  --lenpen 1.0 \
  --beam 5  \
  --decoder-langtok \
  --max-tokens 4096
```

## Citation

```bibtex
@article{li2020deep,
  title={Deep Transformers with Latent Depth},
  author={Li, Xian and Stickland, Asa Cooper and Tang, Yuqing and Kong, Xiang},
  journal={arXiv preprint arXiv:2009.13102},
  year={2020}
}
```
