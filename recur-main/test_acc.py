import os

list_pt=[]


for i in range(67):
    os.system(f"python train.py --exp_name=test --exp_id=测试5M_固定训练数据_lr0.0021_6gpu  --eval_only=True --eval_from_exp=/home/skm21/recur-main/train/5M_固定训练数据_lr0.0021_6gpu --periodic_i_pth=periodic-{i}.pth --eval_data=/home/skm21/recur-main/data/test/10000.txt")