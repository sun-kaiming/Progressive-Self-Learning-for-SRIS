import csv
# from recur.utils.decode import decode
# with open('/home/skm21/recur-main/data/10M/auto_formula_correct_temp.csv','r',encoding='utf-8')as f:
#      # lines=f.readlines()
#      reader=csv.reader(f)
#      for i,row in enumerate(reader):
#           print(row)
#           # exit()
#           if i>10000:
#                break
import time

for i in range(10):
    time.sleep(0.1)
    j = 0
    while True:
        time.sleep(0.1)
        j += 1
        if j > 3:
            break
        print("j:", j)
    print(i * 10)
