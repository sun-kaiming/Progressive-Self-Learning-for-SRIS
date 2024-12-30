import json

with open('/home/skm21/fairseq-0.12.0/checkpoints/4500w_iter_allOEIS/iter1/result/final_res.json', 'r') as f:
    dic = json.load(f)
    print(len(dic))
