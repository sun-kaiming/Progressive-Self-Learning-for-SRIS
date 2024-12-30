# --model_path 推理的模型路径
# --beam_size 32  ## 解码时束搜素的宽度
# --nbest 32 \   ## 推理每个数列产生的候选公式数
# --gpuid 0 \   # 可以使用的GPU id  ，测试只需一个gpu即可
# --test_type easy \  ## 三种测试集: easy、sign、base
# --input_seq_len 25


python eval_test_acc.py \
--model_path save/model/4500w_combine_train_36w_iter50.pt 

