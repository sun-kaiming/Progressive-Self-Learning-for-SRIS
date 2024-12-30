import os

for i in range(1, 49):
    path = f'/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter{i}'
    os.system(f"rm -r  {path}/model*")
    os.system(f"rm -r  {path}/result1")
    os.system(f"rm -r  {path}/result2")
    os.system(f"rm -r  {path}/result3")
    os.system(f"rm -r  {path}/result4")
    os.system(f"rm -r  {path}/result5")
    os.system(f"rm -r  {path}/result6")
    os.system(f"rm -r  {path}/result/temp_res.csv")
    # os.system(f"rm -r  {path}/result/generate-test.csv")
