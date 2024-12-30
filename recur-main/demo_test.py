# import numpy as np
#
# for i in range(0, 100):
#     rng = np.random.RandomState(i)
#     p=rng.rand()
#     print(p)
import math
import tqdm
import numpy as np

x1=4
x2=0
x3=1
x4=-10

# for i in range(5,20):
#     x5 = x1 * ( i+ 48)
#
#     x1 = x2
#     x2 = x3
#     x3 = x4
#     x4 = x5
#
#
#     print(i,x5)

# y=(math.exp(1.8))/(1+math.exp(1.8))
# print(y)
#
# for i in range(1,100):
#     print(int(math.pow(2,i)-math.pow(i,2)),end=',')

# with open('/home/skm21/recur-main/data/1000M/data.prefix','r',encoding='utf-8') as f:
#     lines=f.readlines()
#     print(len(lines))
#     # for i in range(len(lines)):
#     #     if '- 9 + 8 - 2 - 4107 - 8195 - 1 2293 - 2 496 - 3 2787 - 4 9176 - 7 3768 - 11 651 - 16 3923 - 24 1787 - 35 6534 - 52 4553 - 77 436 - 113 1066 - 165 9715 - 243 4247 - 356 9409 - 523 3220 - 767 1563 - 1124 5068 - 1648 2384' in lines[i]:
#     #         print(lines[i])
import torch

x=torch.arange(210).view(21,10)
# print(x)
# print(x[1:])

sum=8+  6+ 12+ 13+ 10+ 16+  5+ 21+ 13+ 18
# print(sum)
#
# x=['1','2','3']
# # print(-8%4)
#
# # with open("test.txt",'a',encoding='utf-8') as f:
# #     for j in range(100000):
# #         print(j)
# #         for i in range(100000000):
# #
# #             f.write(str(i))
#
# # print(1/0)
# print(8/-13)

# with open("data/train/5M.txt",'r',encoding='utf-8')as f:
# print( 1 << 60)
#     lines=f.readlines()
#     print(len(lines))
# rng = np.random.RandomState(12333)
# for i in range(11):
#     print(rng.randint(10))
#
# list_data=[1,2,3,4]
# lis=iter(list_data)
# for i in range(80000):
#     if i%4==0:
#         lis = iter(list_data)
#     print(next(lis))


from tqdm import tqdm
for j in range(100):
    for i in tqdm(range(1000)):
         #do something
         for h in range(1000):
             g=1*123
         # pass