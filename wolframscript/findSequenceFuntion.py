import csv
import os
from tqdm import tqdm

import argparse

# ss='1,2,5,16,67,374,2825,29212,417199,8283458,229755605,8933488744,488176700923,37558989808526,4073773336877345,623476476706836148,134732283882873635911,41128995468748254231002,17741753171749626840952685,10817161765507572862559462656'
# print(len(ss.split(',')))
# exit()
# str=''
# for i in range(0,20):
#     str=f'python findLinearRecurrence.py --staid={6100+i*15030} --endid={6100+(i+1)*15030}'
#     print(str)
# # print(str)
# exit()
parser = argparse.ArgumentParser(description='命令行中传入一个数字')
# type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--staid', type=int, default=-1, help='传入的数字')
parser.add_argument('--endid', type=int, default=10001, help='传入的数字')
parser.add_argument('--file_name', type=str, default=35, help='文件名字')
args = parser.parse_args()

# 获得传入的参数
staid = args.staid
endid = args.endid
file_name = args.file_name

file_num = 35
# with open('/home/skm21/fairseq-0.12.0/data_oeis/wolfram_test/findlinarrecurrence_testdata.csv','r',encoding='utf-8')as f:
with open(f'/home/skm21/fairseq-0.12.0/data_oeis/{file_name}.csv', 'r',
          encoding='utf-8') as f:
    with open(f'/home/skm21/fairseq-0.12.0/wolframscript/result_findFuntion/{file_name}_res.csv', 'a',
              encoding='utf-8') as f2:
        reader = csv.reader(f)
        count = 0
        writer = csv.writer(f2)
        # writer.writerow(['seq_name','seq','pre_seq','pre_formula',"pre_formula_len","res"])
        # pool = Pool(5)  # 定义一个进程池，最大进程数48
        list_seq = []
        for j in range(1):
            for i, row in enumerate(tqdm(reader)):
                if i <= endid and i > staid:
                    if row[1][0] == ',':
                        row[1] = row[1][1:]
                        seq_list = row[1].split(',')
                    else:
                        seq_list = row[1].split(',')
                    len_seq = len(seq_list)

                    if len(seq_list) >= file_num:
                        seq = ','.join(seq_list[:file_num - 10])
                    else:
                        seq = ','.join(seq_list[:len_seq - 10])
                    print(seq)
                    print(len(seq.split(',')))
                    pre_seq = ''
                    pre_formula = ''
                    pre_formula_len = ''
                    res_final = ''
                    # 1,2,2,1,1,2,1,2,2,1,2,2,1,1,2,1,1,2,2,1,2,1,1,2,1,2,2,1,1,2,1,1
                    # seq='0,1,2,5,8,13,18,25,32,41,50,61,72,85,98,113,128,145,162'
                    # len_seq=30
                    command = f'wolframscript -file findsequenceFunction.sh {seq} {len_seq}'

                    res = os.popen(command)

                    info = res.readlines()  # 读取命令行的输出到一个list

                    if 'FindSequenceFunction' not in info[0]:
                        formula = info[0].strip()

                        pre_res = info[1].strip()[1:-1].replace(' ', '')

                        pre_seq = pre_res
                        pre_formula = formula
                        pre_formula_len = len_seq
                        if row[1] == pre_seq:
                            res_final = "success"
                            count += 1
                        else:
                            res_final = 'failed'
                    writer.writerow([row[0], "," + row[1], "," + pre_seq, pre_formula, pre_formula_len, res_final])
                    # if i>4:
                    #     exit()

        print("findsequenceFunction_正确发现的公式数量:", count)
