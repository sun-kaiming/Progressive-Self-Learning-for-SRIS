数据预处理
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/train --validpref data_oeis/val --testpref data_oeis/test  \
--destdir data-bin/iter1.2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/train --validpref data_oeis/val --testpref data_oeis/reverse_test/test  \
--destdir data-bin/test_reverse.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/train --validpref data_oeis/val --testpref data_oeis/reverse_test/test2  \
--destdir data-bin/test_reverse2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref checkpoints/model0/iter2.2/result/train --validpref checkpoints/model0/iter2.2/valid --testpref checkpoints/model0/iter2.2/test  \
--destdir data-bin/iter2.2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref checkpoints/model0/iter2.2/result/train --validpref checkpoints/model0/iter2.2/valid --testpref checkpoints/model0/iter2.2/test  \
--destdir data-bin/iter2.2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref checkpoints/model0/iter2/result/train --validpref checkpoints/model0/iter2/result/valid --testpref checkpoints/model0/iter2/result/valid  \
--destdir data-bin/iter2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref checkpoints/model0/iter2/result/train --validpref checkpoints/model0/iter2/result/valid --testpref data_oeis/test \
--destdir data-bin/iter2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/test_3 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/test3.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/test_4 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/test4.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/random_data/test_1 --validpref data_oeis/random_data/test_1 --testpref data_oeis/random_data/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/random1w_test_data_new
####################test集长度在6-25之间 预留了10个数，用来预测检查##########################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25/test_3 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25/test3.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25/test_4 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25/test4.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25/test_4 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/oeis6-25/test4.2.oeis

###############################把测试集分成三份 转化为bin文件############################################################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25_3/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25_3/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25_3/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25_3/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25_3/test_3 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25_3/test3.oeis


###############################逆向seq 把测试集分成2份 转化为bin文件############################################################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_reverse_2/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_reverse_2/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_reverse_2/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_reverse_2/test2.oeis

############################### 把测试集分成2份 转化为bin文件############################################################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_2/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_2/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_2/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_2/test2.oeis

############################构建tgt_vocab超过240的大小，使得束搜索宽度能够等于240########################################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/train --validpref data_oeis/val --testpref data_oeis/val \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/test_beam240.oeis



####################test集长度在6-15之间 预留了10个数，用来预测检查##########################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-15/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/oeis6-15/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-15/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/oeis6-15/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-15/test_3 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/oeis6-15/test3.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-15/test_4 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt.txt \
--destdir data-bin/oeis6-15/test4.oeis


####################test集长度在6-25之间 预留了1-all个数，用来预测检查，尽可能保留多的测试数据，就有可能更多的初始值，进而产生更多的新公式##########################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_4/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_4/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_4/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_4/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_4/test_3 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_4/test3.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_4/test_4 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_4/test4.oeis

#################### 分成三份数据 test集长度在6-25之间 预留了1-all个数，用来预测检查，尽可能保留多的测试数据，就有可能更多的初始值，进而产生更多的新公式##########################
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_3/test_1 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_3/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_3/test_2 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_3/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis6-25pre1_all_3/test_3 \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/oeis6-25pre1_all_3/test3.oeis

# 处理新的处理为二进制文件（含有a(n-7),,,a(n-12)）
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/checkpoints/model0_an_12/iter101/result/train --validpref /home/skm21/fairseq-0.12.0_213/checkpoints/model0_an_12/iter101/result/valid --testpref /home/skm21/fairseq-0.12.0_213/checkpoints/model0_an_12/iter101/result/valid \
--destdir data-bin/new_vocab_an7-12/


##处理新的测试集为二进制文件，新的测试集最小输入长度为12项
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_1 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_2 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_3 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test3.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_3 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test4.oeis


##处理新的测试集为二进制文件，新的测试集最小输入长度为12项
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_1 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test1.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_2 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test2.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_3 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test3.oeis

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/val --validpref data_oeis/val --testpref data_oeis/oeis13-25pre1_all_4_small/test_3 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis13-25pre1_all_4_small/test4.oeis


###把新生成的数据（包含an7-12）的转化为二进制用来训练
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/recur-main/data/10M/an7-12dataset/train --validpref /home/skm21/recur-main/data/10M/an7-12dataset/valid --testpref /home/skm21/recur-main/data/10M/an7-12dataset/valid \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/oeis_train_baseline_an7_12


###把初始项为从oeis中选择的方法生成2500万数据转化为二进制
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/recur-main/data/change_init_oeis/train --validpref /home/skm21/recur-main/data/change_init_oeis/val --testpref /home/skm21/recur-main/data/change_init_oeis/val \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/init_term_select_oeis

###将构造的500w线性递推转化为二进制
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/recur-main/data/lineRecur/train --validpref /home/skm21/recur-main/data/lineRecur/val --testpref /home/skm21/recur-main/data/lineRecur/val \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/lineRecur_500w

###将1w测试集转化为二进制
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/base --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/base --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/base \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/base1w

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sign --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sign --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sign \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/sign1w

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/easy1w

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy_25 --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy_25 --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy_25 \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/easy25_1w

###将3w测试集转化为二进制:合并easy、base、sign三个测测试集
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sum_test --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sum_test --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sum_test \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/sum_test


###将构造的4000w数据转化为二进制：包括2500w的初始项为oeis的数据+1000w的简单线性递推（max_op=2）+250w的简单递推（max_op=12）+250w的简单递推（max_op=12，含有mul,以及常数）
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/recur-main/data/merge_data_4000w/train --validpref /home/skm21/recur-main/data/merge_data_4000w/val --testpref /home/skm21/recur-main/data/merge_data_4000w/val \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/merge_data_4000w

# 将测试集装化为二进制，其中词表选用原来梯度度为0-6的词表
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/base --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/base --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/base \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/base1w_0-6recur

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sign --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sign --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/sign \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/sign1w_0-6recur

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy --validpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy --testpref /home/skm21/fairseq-0.12.0_213/data_oeis/src_tgt_testSet/easy \
--srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
--destdir data-bin/easy1w_0-6recur


#sum_data_sl_iter11 bin文件生成
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref checkpoints/sum_data_sl/sum_data_sl_iter11/data/train --validpref checkpoints/sum_data_sl/sum_data_sl_iter11/data/valid --testpref checkpoints/sum_data_sl/sum_data_sl_iter11/data/valid \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/sum_data_sl/iter11.oeis


#lineRecur_deep_sum_new bin文件生成 新数据 将简单的线性递推a(n)=a(n-1)+a(n-2)更改为a(n)=b1*a(n-1)+b2*a(n-2)
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/recur-main/data/lineRecur_deep_sum_new/train --validpref /home/skm21/recur-main/data/lineRecur_deep_sum_new/val --testpref /home/skm21/recur-main/data/lineRecur_deep_sum_new/val \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/lineRecur_deep_sum_new


###将构造的4500w数据转化为二进制：包括2500w的初始项为oeis的数据+1000w的简单线性递推（max_op=2）+250w的简单递推（max_op=12）+250w的简单递推（max_op=12，含有mul,以及常数）+500转化的常系数递推公式
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref /home/skm21/recur-main/data/merge_data_4500w/train --validpref /home/skm21/recur-main/data/merge_data_4500w/val --testpref /home/skm21/recur-main/data/merge_data_4500w/val \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/merge_data_4500w



fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/sign_split4/test_1 --validpref data_oeis/sign_split4/test_1 --testpref data_oeis/sign_split4/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/4500w_iter_sign/test1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/sign_split4/test_2 --validpref data_oeis/sign_split4/test_2 --testpref data_oeis/sign_split4/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/4500w_iter_sign/test2

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/sign_split4/test_3 --validpref data_oeis/sign_split4/test_3 --testpref data_oeis/sign_split4/test_3 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/4500w_iter_sign/test3

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/sign_split4/test_4 --validpref data_oeis/sign_split4/test_4 --testpref data_oeis/sign_split4/test_4 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/4500w_iter_sign/test4



fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy9w_35_split3/test_1 --validpref data_oeis/easy9w_35_split3/test_1 --testpref data_oeis/easy9w_35_split3/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy9w_35_split3/test1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy9w_35_split3/test_2 --validpref data_oeis/easy9w_35_split3/test_2 --testpref data_oeis/easy9w_35_split3/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy9w_35_split3/test2

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy9w_35_split3/test_3 --validpref data_oeis/easy9w_35_split3/test_3 --testpref data_oeis/easy9w_35_split3/test_3 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy9w_35_split3/test3




fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split4/test_1 --validpref data_oeis/easy36w_35_split4/test_1 --testpref data_oeis/easy36w_35_split4/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split4/test1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split4/test_2 --validpref data_oeis/easy36w_35_split4/test_2 --testpref data_oeis/easy36w_35_split4/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split4/test2

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split4/test_3 --validpref data_oeis/easy36w_35_split4/test_3 --testpref data_oeis/easy36w_35_split4/test_3 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split4/test3

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split4/test_4 --validpref data_oeis/easy36w_35_split4/test_4 --testpref data_oeis/easy36w_35_split4/test_4 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split4/test4


fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split3/test_1 --validpref data_oeis/easy36w_35_split3/test_1 --testpref data_oeis/easy36w_35_split3/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split3/test1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split3/test_2 --validpref data_oeis/easy36w_35_split3/test_2 --testpref data_oeis/easy36w_35_split3/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split3/test2

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split3/test_3 --validpref data_oeis/easy36w_35_split3/test_3 --testpref data_oeis/easy36w_35_split3/test_3 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split3/test3

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split2/test_1 --validpref data_oeis/easy36w_35_split2/test_1 --testpref data_oeis/easy36w_35_split2/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split2/test1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split2/test_2 --validpref data_oeis/easy36w_35_split2/test_2 --testpref data_oeis/easy36w_35_split2/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split2/test2







fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split6/test_1 --validpref data_oeis/easy36w_35_split6/test_1 --testpref data_oeis/easy36w_35_split6/test_1 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split6/test1

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split6/test_2 --validpref data_oeis/easy36w_35_split6/test_2 --testpref data_oeis/easy36w_35_split6/test_2 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split6/test2

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split6/test_3 --validpref data_oeis/easy36w_35_split6/test_3 --testpref data_oeis/easy36w_35_split6/test_3 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split6/test3

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split6/test_4 --validpref data_oeis/easy36w_35_split6/test_4 --testpref data_oeis/easy36w_35_split6/test_4 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split6/test4

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split6/test_5 --validpref data_oeis/easy36w_35_split6/test_5 --testpref data_oeis/easy36w_35_split6/test_5 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split6/test5

fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref data_oeis/easy36w_35_split6/test_6 --validpref data_oeis/easy36w_35_split6/test_6 --testpref data_oeis/easy36w_35_split6/test_6 \
--srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
--destdir data-bin/easy36w_35_split6/test6



#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
训练命令
CUDA_VISIBLE_DEVICES=2 fairseq-train data-bin/iter2.2.oeis \
--optimizer adam --lr-scheduler inverse_sqrt --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --batch-size 128 --max-epoch 50 \
--arch transformer --save-dir checkpoints/model0/iter2.8 --keep-last-epochs 10 \
--log-interval 96 --no-progress-bar --log-file checkpoints/model0/iter2.8/file.log

CUDA_VISIBLE_DEVICES=3 fairseq-train data-bin/iter2.3.oeis \
--optimizer adam --lr-scheduler inverse_sqrt --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --batch-size 256 --max-epoch 400 \
--arch transformer --save-dir checkpoints/model0/iter2.11  \
--log-interval 96 --no-progress-bar --log-file checkpoints/model0/iter2.11/file.log

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/data.rg \
--optimizer adam --lr-scheduler inverse_sqrt --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --batch-size 128 --max-epoch 50 \
--arch transformer --save-dir checkpoints/rg/train1 --keep-last-epochs 10 \
--log-interval 96 --no-progress-bar --log-file checkpoints/rg/train1/file.log

CUDA_VISIBLE_DEVICES=3 fairseq-train data-bin/test_beam240.oeis \
--optimizer adam --lr-scheduler inverse_sqrt --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --batch-size 128 --max-epoch 5 \
--arch transformer --save-dir checkpoints/model0/test_beam240 --keep-last-epochs 10 \
--fp16 \
--log-interval 96  --log-file checkpoints/model0/test_beam240/file.log


CUDA_VISIBLE_DEVICES=3,4 fairseq-train data-bin/iter2.oeis \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0005 --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.1 --dropout 0.2 --batch-size 128 --max-epoch 50 \
        --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
        --warmup-updates 2000 --warmup-init-lr '1e-07' --num-workers 8 \
        --arch transformer --save-dir checkpoints/model0/iter2.4 --keep-last-epochs 10 \
        --log-interval 96 --no-progress-bar --log-file checkpoints/model0/iter2.4/file.log

CUDA_VISIBLE_DEVICES=1,2,3,4,5 fairseq-train /home/skm21/fairseq-0.12.0_213/data-bin/oeis_train_baseline_an7_12 \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 8000 --warmup-init-lr '1e-07' \
         --batch-size 128 --max-epoch 10 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/baseline_an7_12  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/baseline_an7_12/file.log

CUDA_VISIBLE_DEVICES=1,2,3,4,5 fairseq-train /home/skm21/fairseq-0.12.0_213/data-bin/init_term_select_oeis \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 8000 --warmup-init-lr '1e-07' \
         --batch-size 128 --max-epoch 10 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/init_term_select_oeis  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/init_term_select_oeis/file.log \

CUDA_VISIBLE_DEVICES=1,2,3,4,5 fairseq-train /home/skm21/fairseq-0.12.0_213/data-bin/lineRecur_500w \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 8000 --warmup-init-lr '1e-07' \
         --batch-size 128 --max-epoch 20 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/lineRecur_500w  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/lineRecur_500w/file.log \
         --restore-file checkpoints/lineRecur_500w/checkpoint_best.pt

#训练4000w数据： 包括2500w的初始项为oeis的数据+1000w的简单线性递推（max_op=2）+250w的简单递推（max_op=12）+250w的简单递推（max_op=12，含有mul,以及常数）
CUDA_VISIBLE_DEVICES=1,2,3,4,5 fairseq-train /home/skm21/fairseq-0.12.0_213/data-bin/merge_data_4000w \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 8000 --warmup-init-lr '1e-07' \
         --batch-size 128 --max-epoch 20 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/merge_data_4000w  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/merge_data_4000w/file.log \
         --restore-file checkpoints/init_term_select_oeis/checkpoint_best.pt

# 训练sum_data_sl_iter11  该为fp32训练。提升训练精度
CUDA_VISIBLE_DEVICES=2,3,4,5 fairseq-train data-bin/sum_data_sl/iter11.oeis \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 1000 --warmup-init-lr '1e-07' \
         --batch-size 256 --max-epoch 200 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/sum_data_sl/sum_data_sl_iter11_dropout0.2 \
         --log-interval 500   \
         --log-file checkpoints/sum_data_sl/sum_data_sl_iter11_dropout0.2/file.log \


CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-train data-bin/lineRecur_deep_sum_new \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 8000 --warmup-init-lr '1e-07' \
         --batch-size  128  --max-epoch 15 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/lineRecur_deep_sum_new \
         --log-interval 500  --fp16 \
         --log-file checkpoints/lineRecur_deep_sum_new/file.log \
         --restore-file /home/skm21/fairseq-0.12.0_213/checkpoints/init_term_select_oeis/checkpoint_best.pt

#训练4500w数据： 包括2500w的初始项为oeis的数据+1000w的简单线性递推（max_op=2）+250w的简单递推（max_op=12）+250w的简单递推（max_op=12，含有mul,以及常数）+500w常系数线性递推
CUDA_VISIBLE_DEVICES=2,3,4,5 fairseq-train data-bin/merge_data_4500w \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 8000 --warmup-init-lr '1e-07' \
         --batch-size 128 --max-epoch 10 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/merge_data_4500w  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/merge_data_4500w/file.log \



生成结果
CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/skm21/fairseq-0.12.0_213/data-bin/oeis_train_baseline_an7_12 \
--path /home/skm21/fairseq-0.12.0_213/checkpoints/baseline_an7_12/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 32 --nbest 32 --results-path checkpoints/baseline_an7_12/result \

CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/oeis6-25/test4.2.oeis \
        --path checkpoints/model0/iter51/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 64 --nbest 32 --results-path checkpoints/model0/iter51/result4

CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/rg/rg.bin \
--path checkpoints/rg/train1/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/rg/train1/result \

CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/oeis6-15/test1.oeis \
--path checkpoints/model0/iter51/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/model0/iter51/result1 \
& \
CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/oeis6-15/test2.oeis \
--path checkpoints/model0/iter51/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/model0/iter51/result2 \
& \
CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/oeis6-15/test3.oeis \
--path checkpoints/model0/iter51/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/model0/iter51/result3 \
& \
CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/oeis6-15/test4.oeis \
--path checkpoints/model0/iter51/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/model0/iter51/result4 \


CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/test1.oeis \
--path checkpoints/model0/iter17/checkpoint_best.pt \
--batch-size 512 --beam 5 --nbest 1 --results-path checkpoints/model0/iter17/result2


CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/sum_test \
--path /home/skm21/fairseq-0.12.0_213/checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
--max-tokens 4096 --batch-size  512 --beam 32 --nbest 32 --results-path checkpoints/merge_data_4000w/all_test/result1 \


CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/sum_test \
--path checkpoints/lineRecur_500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  512 --beam 32 --nbest 32 --results-path checkpoints/lineRecur_500w/base_test_res/result1 \

#/home/skm21/fairseq-0.12.0_213/checkpoints/merge_data_4000w/checkpoint_19_950000.pt
CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/oeis13-25pre1_all_4/test1.oeis \
--path checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/merge_data_4000w/all_test/result1 \
& \
CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/oeis13-25pre1_all_4/test2.oeis \
--path checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/merge_data_4000w/all_test/result2 \
& \
CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/oeis13-25pre1_all_4/test3.oeis \
--path checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/merge_data_4000w/all_test/result3 \
& \
CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/oeis13-25pre1_all_4/test4.oeis \
--path checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/merge_data_4000w/all_test/result4

# /home/skm21/fairseq-0.12.0_213/checkpoints/lineRecur_500w/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/oeis13-25pre1_all_4/test1.oeis \
--path checkpoints/lineRecur_500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/lineRecur_500w/all_test/result1 \
& \
CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/oeis13-25pre1_all_4/test2.oeis \
--path checkpoints/lineRecur_500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/lineRecur_500w/all_test/result2 \
& \
CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/oeis13-25pre1_all_4/test3.oeis \
--path checkpoints/lineRecur_500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/lineRecur_500w/all_test/result3 \
& \
CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/oeis13-25pre1_all_4/test4.oeis \
--path checkpoints/lineRecur_500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 64 --nbest 32 --results-path checkpoints/lineRecur_500w/all_test/result4

cat /home/skm21/fairseq-0.12.0_213/result/eval_res/input.src | fairseq-interactive data-bin/new_vocab_an7-12 --source-lang src --target-lang tgt  \
    --path /home/skm21/fairseq-0.12.0_213/checkpoints/merge_data_4000w/checkpoint_19_950000.pt  --beam 32 --nbest 1 &> /home/skm21/fairseq-0.12.0_213/result/eval_res/result.txt


#解码 oeis easy测试集
CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/easy25_1w \
        --path /home/skm21/fairseq-0.12.0_213/checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 1 --results-path /home/skm21/fairseq-0.12.0_213/result/easy25_res

# 得到排序后的译文
grep ^H /home/skm21/fairseq-0.12.0_213/result/easy25_res/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 > /home/skm21/fairseq-0.12.0_213/result/easy25_res/pre_res.txt

# --nbest 32 生成32个候选项
CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/easy1w \
        --path /home/skm21/fairseq-0.12.0_213/checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path /home/skm21/fairseq-0.12.0_213/result/easy_res_nbest32

grep ^H /home/skm21/fairseq-0.12.0_213/result/easy_res_nbest32/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 > /home/skm21/fairseq-0.12.0_213/result/easy_res_nbest32/pre_res.txt


CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/easy25_1w \
        --path /home/skm21/fairseq-0.12.0_213/checkpoints/merge_data_4000w/checkpoint_19_950000.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path /home/skm21/fairseq-0.12.0_213/result/easy25_res_nbest32

grep ^H /home/skm21/fairseq-0.12.0_213/result/easy25_res_nbest32/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 > /home/skm21/fairseq-0.12.0_213/result/easy25_res_nbest32/pre_res.txt



# 解码自学习10轮的easy测试集, 35项
CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/easy1w \
        --path /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 1 --results-path /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/result/easy_res_nbest1

grep ^H /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/result/easy_res_nbest1/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 > /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/result/easy_res_nbest1/pre_res.txt

# 解码自学习10轮的easy测试集, 35项  nbest 32
CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/easy1w \
        --path /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/result/easy_res_nbest32

grep ^H /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/result/easy_res_nbest32/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 > /home/skm21/fairseq-0.12.0_213/checkpoints/sum_data_sl/result/easy_res_nbest32/pre_res.txt

# /home/skm21/fairseq-0.12.0_212/checkpoints/merge_data_4500w/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/oeis13-25pre1_all_4/test1.oeis \
--path checkpoints/merge_data_4500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 32 --nbest 32 --results-path checkpoints/merge_data_4500w/all_oeistest/result1 \
& \
CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/oeis13-25pre1_all_4/test2.oeis \
--path checkpoints/merge_data_4500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 32 --nbest 32 --results-path checkpoints/merge_data_4500w/all_oeistest/result2 \
& \
CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/oeis13-25pre1_all_4/test3.oeis \
--path checkpoints/merge_data_4500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 32 --nbest 32 --results-path checkpoints/merge_data_4500w/all_oeistest/result3 \
& \
CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/oeis13-25pre1_all_4/test4.oeis \
--path checkpoints/merge_data_4500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 32 --nbest 32 --results-path checkpoints/merge_data_4500w/all_oeistest/result4

CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/sign1w \
--path checkpoints/merge_data_4500w/checkpoint_best.pt \
--max-tokens 4096 --batch-size  128 --beam 32 --nbest 32 --results-path checkpoints/merge_data_4500w/sign_test2/result1

grep ^S /home/skm21/fairseq-0.12.0_212/checkpoints/merge_data_4500w/all_oeistest/result4/generate-test.txt | sort -n -k 2 -t '-' | cut -f 2 > /home/skm21/fairseq-0.12.0_212/checkpoints/merge_data_4500w/all_oeistest/result4/seq.src


### 数据增强（一个序列分成不同的长度：7,15,25 为了提高泛化能力，以更好的部署模型）训练模型
# 数据二进制化
fairseq-preprocess --source-lang src --target-lang tgt  \
--trainpref checkpoints/sum_data_sl_iter11_dropout0.2_7_15_25/data/train --validpref checkpoints/sum_data_sl_iter11_dropout0.2_7_15_25/data/valid --testpref checkpoints/sum_data_sl_iter11_dropout0.2_7_15_25/data/valid \
--srcdict data-bin/new_vocab_an7-12/dict.src.txt --tgtdict data-bin/new_vocab_an7-12/dict.tgt.txt \
--destdir data-bin/sum_data_sl_iter11_dropout0.2_7_15_25

##模型训练
CUDA_VISIBLE_DEVICES=2,3,4,5 fairseq-train data-bin/sum_data_sl_iter11_dropout0.2_7_15_25 \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 1000 --warmup-init-lr '1e-07' \
         --batch-size 256 --max-epoch 200 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/sum_data_sl_iter11_dropout0.2_7_15_25 \
         --log-interval 500   \
         --log-file checkpoints/sum_data_sl_iter11_dropout0.2_7_15_25/file.log


#训练：500wdata, 递推度为6，初始项为随机初始项
CUDA_VISIBLE_DEVICES=4,5 fairseq-train data-bin/500w_d6_randomInitTerm \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 4000 --warmup-init-lr '1e-07' \
         --batch-size 256 --max-epoch 10 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/500w_d6_randomInitTerm  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/500w_d6_randomInitTerm/file.log


grep ^S  checkpoints/merge_data_4500w/all_oeistest/result4/generate-test.txt  |  sort -n -k 2 -t '-' | cut -f 2 >  checkpoints/merge_data_4500w/all_oeistest/result4/seq.src

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train data-bin/model0_32_auto/iter50.oeis \
         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
         --warmup-updates 1000 --warmup-init-lr '1e-07' \
         --batch-size 256 --max-epoch 100 \
         --save-interval-updates=10000 \
         --arch transformer --save-dir checkpoints/model0_32_auto_208/iter50  --no-epoch-checkpoints \
         --log-interval 500 --fp16  \
         --log-file checkpoints/model0_32_auto_208/iter50/file.log


fairseq-preprocess --source-lang src --target-lang tgt  \
        --trainpref /home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train/iter51/result/train --validpref /home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train/iter51/result/valid --testpref /home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train/iter51/result/valid  \
        --srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
        --destdir data-bin/4500w_combine_train/iter51.oeis

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 fairseq-train data-bin/4500w_combine_train/iter51.oeis \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
        --warmup-updates 200 --warmup-init-lr '1e-07' \
        --batch-size 256 --max-epoch 100 \
        --arch transformer --save-dir checkpoints/4500w_combine_train/iter51 --keep-best-checkpoints 2 --no-epoch-checkpoints  \
        --log-interval 96  \
        --log-file checkpoints/4500w_combine_train/iter51/file.log


CUDA_VISIBLE_DEVICES=0 fairseq-generate ./data-bin/decoder_seq_set/oeis_seq_split3_small_seqSet/test_1                 --path checkpoints/test_1010/iter1/model1/checkpoint_best.pt                 --max-tokens 4096 --batch-size 128 --beam 1 --nbest 1 --results-path checkpoints/test_1010/iter1/result1