##### 参数解释
## 1. --max-epoch  每次迭代的最大训练轮次
## 2. --model-type 迭代的名称，每次新的迭代最好拟定一个新的名称，避免之前训练的模型被覆盖掉
## 3. --beam-size 束搜索的波束宽度，该参数越大，解码产生的结果可能越好，但是相应的解码时间会增加
## 4. --n-best 每个数列的候选公式个数
## 5. --is-combine  是否采用联合训练的方式，即是否在每轮迭代训练两个模型
## 6. --random-rate-model1 模型1随机选择全部OEIS数据的比率
## 7. --random-rate-model2 模型2随机选择全部OEIS数据的比率
## 8. --small-oeis-testset 是否采用小数据进行测试
## 9. --gpus-id 显卡的id字符串，如果采用多卡，则id之间用“,”连接; 单卡只写一个id即可
## 10. --sta-iter-id 开始迭代的轮次 id  一般为1
## 11. --end-iter-id 结束迭代的轮次 id  一般为50

python iter_script.py \
  --max-epoch 3 \
  --model-type test_1201-14 \
  --beam-size 32 \
  --n-best 32 \
  --is-combine True \
  --random-rate-model1 1.0 \
  --random-rate-model2 0.5 \
  --small-oeis-testset "False" \
  --gpus-id 0,1 \
  --sta-iter-id 1 \
  --end-iter-id 3