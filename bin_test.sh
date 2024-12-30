fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy6w_split1/test_1 --validpref data_oeis/easy6w_split1/test_1 --testpref data_oeis/easy6w_split1/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy6w_vocab4

# easy1w分割成4份分别转换为测试集
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy1w_split4/test_1 --validpref data_oeis/easy1w_split4/test_1 --testpref data_oeis/easy1w_split4/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy1w_split4_vocab4/easysplit1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy1w_split4/test_2 --validpref data_oeis/easy1w_split4/test_2 --testpref data_oeis/easy1w_split4/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy1w_split4_vocab4/easysplit2

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy1w_split4/test_3 --validpref data_oeis/easy1w_split4/test_3 --testpref data_oeis/easy1w_split4/test_3 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy1w_split4_vocab4/easysplit3

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy1w_split4/test_4 --validpref data_oeis/easy1w_split4/test_4 --testpref data_oeis/easy1w_split4/test_4 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy1w_split4_vocab4/easysplit4


# easy6w分割成4份分别转换为测试集
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy6w_split4/test_1 --validpref data_oeis/easy6w_split4/test_1 --testpref data_oeis/easy6w_split4/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy6w_split4_vocab4/easysplit1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy6w_split4/test_2 --validpref data_oeis/easy6w_split4/test_2 --testpref data_oeis/easy6w_split4/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy6w_split4_vocab4/easysplit2

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy6w_split4/test_3 --validpref data_oeis/easy6w_split4/test_3 --testpref data_oeis/easy6w_split4/test_3 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy6w_split4_vocab4/easysplit3

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy6w_split4/test_4 --validpref data_oeis/easy6w_split4/test_4 --testpref data_oeis/easy6w_split4/test_4 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy6w_split4_vocab4/easysplit4

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy1w_25_split1/test_1 --validpref data_oeis/easy1w_25_split1/test_1 --testpref data_oeis/easy1w_25_split1/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy1w_25_vocab4


fairseq-preprocess --source-lang src --target-lang tgt  \
        --trainpref /home/skm21/fairseq-0.12.0/checkpoints/Easy_iter/easy_25/train --validpref /home/skm21/fairseq-0.12.0/checkpoints/Easy_iter/easy_25/valid --testpref /home/skm21/fairseq-0.12.0/checkpoints/Easy_iter/easy_25/valid  \
        --srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
        --destdir data-bin/easy1w_25/iter50.oeis



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train data-bin/easy1w_25/iter50.oeis \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
        --warmup-updates 200 --warmup-init-lr '1e-07' \
        --batch-size 256 --max-epoch 100 \
        --arch transformer --save-dir checkpoints/Easy_iter/easy_25  --keep-best-checkpoints 5 --no-epoch-checkpoints  \
        --log-interval 96 --fp16  \
        --log-file checkpoints/Easy_iter/easy_25/file.log


# 500w_d12_oeisInitTerm 500w
CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train data-bin/500w_d12_oeisInitTerm \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 4000 --warmup-init-lr '1e-07' \
         --batch-size 256 --max-epoch 10 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/500w_d12_oeisInitTerm  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/500w_d12_oeisInitTerm/file.log



CUDA_VISIBLE_DEVICES=3,4,5 fairseq-train data-bin/500w_d6_oeisInitTerm \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 4000 --warmup-init-lr '1e-07' \
         --batch-size 256 --max-epoch 10 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/500w_d6_oeisInitTerm  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/500w_d6_oeisInitTerm/file.log