"""
检验自动生成的公式 是否能够正确生成OEIS数列，

由自动生成的公式 生成的数列，，初始值是随机从-10~10之间选的，所以在判断公式是否能够生成oeis数列时，初始值我们可以直接从oeis中选取，
或者干脆，把oeis的所有初始值都在这个公式上实验一下（可能比较耗费时间）
"""
import collections
import csv
import json
import os
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np

from recur.utils.split_oesi_seq import process_oeis_input_seq
from recur.utils.iter_need_method import test_1wan_easy_acc
from collections import defaultdict
import random
from recur.utils.get_env import get_env
from recur.utils.decode import decode
from opti_const_recur import recur_opti

root = ''
# with open('/home/skm21/fairseq-0.12.0/tree/oeis_tree_save_dict_seqNames3.pkl', 'rb') as f:
#     root = pickle.load(f)
env, file_name = get_env()


def obtain_init_term(path, lenght):
    """
    获取OEIS固定项数的初始项
    path:oeis数据路径
    """
    init_term_list = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[4:]
        init_term_dict = {}
        for line in lines:
            seq = line[9:-2]
            init_term = seq.split(',')[:lenght]

            flag = 0
            for term in init_term:
                if int(term) > 10 or int(term) < -10:
                    flag = 1
                    break

            if flag == 0:
                # print(','.join(init_term))
                input_init_term = process_oeis_input_seq(init_term, lenght)
                init_term_dict[input_init_term] = 1
        print(len(init_term_dict))
    return init_term_dict


def select_short_formula(path_formula, legth):
    """
    由于生成1500万公式，oeis中初始项在-10~10之间的大约有10万，每个公式遍历完每种初始项太耗时，所以先用短的公式进行尝试
    """
    formula_list = []
    with open(path_formula, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        count = 0
        formula_dict = defaultdict(int)
        for i, line in enumerate(lines):
            if i < 10000:
                if len(line.strip().split()) <= legth:
                    formula_dict[line.strip()] = 1
            else:
                break

        # print(len(formula_dict))
        for k, v in formula_dict.items():
            formula_list.append(k)
        print(len(formula_list))
        # print(formula_list[:10])
    return formula_list


def obtain_check_csv(save_path):
    """
    将每个项数少于10的oeis初始项，和所有长度小于10的公式匹配，保存在csv里面方便检验
    """

    init_term_dict = obtain_init_term("/home/skm21/fairseq-0.12.0/data_oeis/stripped", 12)
    formula_list = select_short_formula('/home/skm21/recur-main/data/10M/tgt7-12.txt', 10)
    with open(save_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        count = 0
        for k, v in init_term_dict.items():
            lis = []
            lis.append(k)
            lis.extend(formula_list)
            writer.writerow(lis)
            count += 1
            print(count)


# def delete_chongfu_
def find_special_formula(path, save_path):
    with open(path, 'r', encoding='utf-8') as f1:
        with open(save_path, 'w', encoding='utf-8') as f2:

            # for i,line in enumerate(lines):
            i = 0
            while i < 10000:
                line = f1.readline()
                flag = 0
                if 'x_0_1' in line.strip().split():
                    # flag=1
                    # print(i,line)
                    if 'x_0_11' in line.strip().split() and ('x_0_12' in line.strip().split()):
                        print(line)
                        flag = 1
                    if 'x_0_10' in line.strip().split() and ('x_0_11' in line.strip().split()):
                        print(line)
                        flag = 1
                    if 'x_0_9' in line.strip().split() and ('x_0_10' in line.strip().split()):
                        print(line)
                        flag = 1
                    if 'x_0_8' in line.strip().split() and ('x_0_9' in line.strip().split()):
                        print(line)
                        flag = 1
                    if 'x_0_7' in line.strip().split() and ('x_0_8' in line.strip().split()):
                        print(line)
                        flag = 1
                if flag == 1:
                    f2.write(line)
                i += 1


def get_an7_12dataset(path, save_path):
    """
    把得到的an7-12数列公式数据其中的1k条作为验证集，其他的作为训练集
    """
    with open(path + 'src7-12.txt', 'r', encoding='utf-8') as f1:
        with open(path + 'tgt7-12.txt', 'r', encoding='utf-8') as f2:
            with open(save_path + "train.src", 'w', encoding='utf-8') as f3:
                with open(save_path + 'train.tgt', 'w', encoding='utf-8') as f4:
                    with open(save_path + "valid.src", 'w', encoding='utf-8') as f5:
                        with open(save_path + 'valid.tgt', 'w', encoding='utf-8') as f6:
                            lines1 = f1.readlines()
                            lines2 = f2.readlines()

                            for i in range(len(lines1)):
                                if i < len(lines1) - 1000:
                                    f3.write(lines1[i])
                                    f4.write(lines2[i])
                                else:
                                    f5.write(lines1[i])
                                    f6.write(lines2[i])


def look_new_res_acc(path, testSetpath):
    """
    检验训练完递推度大于7的模型解码得到的4371个公式和原来发现的公式中含有测试集的结果
    """

    with open(path, 'r', encoding='utf-8') as f1:
        with open(testSetpath, 'r', encoding='utf-8') as f2:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            dic = {}
            count = 0
            for row in reader1:
                dic[row[0]] = 1
            for row in reader2:
                if row[0] in dic:
                    count += 1
            print(count / 10000)


def obtain_new_train_data(path):
    """
    处理新的数据为训练集和验证集
    """
    with open(path + "train.csv", 'r', encoding='utf-8') as f1:
        with open(path + "train.src", 'w', encoding='utf-8') as f3:
            with open(path + "train.tgt", 'w', encoding='utf-8') as f4:
                with open(path + "valid.src", 'w', encoding='utf-8') as f5:
                    with open(path + "valid.tgt", 'w', encoding='utf-8') as f6:
                        reader = csv.reader(f1)
                        valid_len = 1000
                        lis = []
                        for row in reader:
                            seq_lis = row[3].split(", ")
                            len_seq_lis = len(seq_lis)
                            if len(seq_lis) >= 13:
                                if len(seq_lis) <= 22:
                                    seq_input = process_oeis_input_seq(seq_lis[:12], 25)
                                if len(seq_lis) > 22:
                                    seq_input = process_oeis_input_seq(seq_lis[:len_seq_lis - 10], 25)
                            temp_lis = []
                            temp_lis.append(seq_input)
                            temp_lis.append(row[1])
                            lis.append(temp_lis)
                        random.shuffle(lis)  # 随机打乱训练集顺序，分割训练集验证集，利用验证选出最佳检查点，用来解码

                        for i, line in enumerate(lis):
                            f3.write(line[0] + '\n')
                            f4.write(line[1] + '\n')
                            if i < valid_len:
                                f5.write(line[0] + '\n')
                                f6.write(line[1] + '\n')


def real_size(path):
    """
    查看csv文件中不重复的数列数量，消除重复
    """
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        dic = {}
        for row in reader:
            dic[row[0]] = 1
        print(len(dic))


def process_data_delete(path1, path2, path_save):
    """
    合并path1，path2的内容。如果path2中的内容含有path1中的，选择path1的内容
    """
    with open(path1, 'r', encoding='utf-8') as f1:
        with open(path2, 'r', encoding='utf-8') as f2:
            with open(path_save, 'w', encoding='utf-8') as f3:
                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                writer = csv.writer(f3)
                dic = {}
                for row in reader1:
                    dic[row[0]] = row
                for row in reader2:
                    if row[0] not in dic:
                        dic[row[0]] = row
                for k, v in dic.items():
                    writer.writerow(v)


def get_all_formula(path, save_path):
    """
    获取自动生成数据的公式，并保村存到txt文件里面
    """

    with open(path, 'r', encoding='utf-8') as f1:
        with open(save_path, 'w', encoding='utf-8') as f2:
            lines = f1.readlines()
            # print(lines[0])
            for line in lines:
                line = json.loads(line)
                f2.write(line['x2'] + '\n')


def return_recurrece_deep(str):
    if 'x_0_12' in str:
        return 12
    elif 'x_0_11' in str:
        return 11
    elif 'x_0_10' in str:
        return 10
    elif 'x_0_9' in str:
        return 9
    elif 'x_0_8' in str:
        return 8
    elif 'x_0_7' in str:
        return 7
    if 'x_0_6' in str:
        return 6
    elif 'x_0_5' in str:
        return 5
    elif 'x_0_4' in str:
        return 4
    elif 'x_0_3' in str:
        return 3
    elif 'x_0_2' in str:
        return 2
    elif 'x_0_1' in str:
        return 1
    else:
        return 0


def obtain_init_term2(path, lenght):
    """
    获取OEIS固定项数的初始项
    path:oeis数据路径
    """

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[4:]
        init_term_list = []
        init_term_dict = {}
        for line in lines:
            seq = line[9:-2]
            seq_len = len(seq.split(','))
            if seq_len >= 20:

                init_term = seq.split(',')[:lenght]
                init_term6 = seq.split(',')[:6]
                flag = 0
                for term in init_term:
                    if abs(int(term)) > 1000:
                        flag = 1
                        break
                if flag == 0:
                    # input_init_term = process_oeis_input_seq(init_term, lenght)
                    input_init_term = ','.join(init_term)
                    init_term_dict[input_init_term] = 1

        for k, v in init_term_dict.items():
            init_term_list.append(k)
        # print(len(init_term_list))
    return init_term_list


def get_all_initterm_list():
    """
    获取所有类型的初始项列表
    """
    init_term1 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 1)
    init_term2 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 2)
    init_term3 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 3)
    init_term4 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 4)
    init_term5 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 5)
    init_term6 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 6)
    init_term7 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 7)
    init_term8 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 8)
    init_term9 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 9)
    init_term10 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 10)
    init_term11 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 11)
    init_term12 = obtain_init_term2('/home/skm21/recur-main/data/OEIS/stripped', 12)

    return init_term1, init_term2, init_term3, init_term4, init_term5, init_term6, init_term7, init_term8, init_term9, init_term10, init_term11, init_term12


def obatain_oeisseq(init_term, hpy, env):
    """
    给定一个公式，随机算去一个oeis初始项，然后返回这个公式生成的25项数列
    init_term:"+ 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0"
    hpy:"sub n 1"
    """

    src = init_term.split()
    hyp = hpy.split()
    list_seq, formula = env.pre_next_term(src, hyp, n_predict=25)
    # print(list_seq)
    return list_seq


def jiasu(args):  # 多进程加速
    termlist, save_path_data, line, env = args

    with open(save_path_data, 'a', encoding='utf-8') as f:
        writer = csv.writer(f)

        deep_formula = return_recurrece_deep(line)
        init_term_list = termlist[deep_formula - 1]
        index_random = random.randint(0, len(init_term_list) - 1)
        init_term = init_term_list[index_random]
        # print("i:",i+1)
        # print("init_term:",init_term)
        seq = obatain_oeisseq(init_term, line.strip(), env)
        if seq != 'error':
            seq = [str(num) for num in seq]
            seq_trans = process_oeis_input_seq(seq, 26)
            # print("seq_trans:",seq_trans)
            # print("formula:",line.strip())
            # print()
            writer.writerow(["," + seq_trans, line.strip()])


def generate_seq(path, save_path_data):
    """
    生成path文件中所有公式对应的seq,并将其存贮到文件中，每个公式的初始项从oeis中随机选取
    """
    init_term1, init_term2, init_term3, init_term4, init_term5, init_term6, init_term7, init_term8, init_term9, init_term10, init_term11, init_term12 = get_all_initterm_list()
    print(len(init_term1))
    print(len(init_term2))
    print(len(init_term3))
    print(len(init_term4))
    print(len(init_term5))
    print(len(init_term6))
    print(len(init_term7))
    print(len(init_term8))
    print(len(init_term9))
    print(len(init_term10))
    print(len(init_term11))
    print(len(init_term12))
    termlist = [init_term1, init_term2, init_term3, init_term4, init_term5, init_term6, init_term7, init_term8,
                init_term9, init_term10, init_term11, init_term12]

    # print(len(init_term1))
    # print(len(init_term2))

    pool = Pool(40)  # 定义一个进程池，最大进程数48
    print("----start----")
    time2 = datetime.now()
    env, file_name = get_env()
    with open(path, 'r', encoding='utf-8') as f1:

        lines = f1.readlines()
        for j in range(1):
            for i, line in enumerate(lines):
                # if i<100:
                if i % 10000 == 0:
                    print("处理到第", i, "行")

                args = (termlist, save_path_data, line, env)
                # args = (row, temp_res_path)
                pool.apply_async(jiasu, (args,))

        pool.close()
        # 等待po中所有子进程执行完成，必须放在close语句之后
        pool.join()
        print("-----end-----")
        print("检测seq时间：", datetime.now() - time2)


def check_2500wData(path):
    """
    检验2500w新生成数据（初始项为oeis初始项）中是否生成了base测试集中美哦与发现的线性递推公式a(n)=a(n-1)+a(n-11)-a(n-12)
    """
    dic = {"add x_0_1 sub x_0_11 x_0_12": 1,
           "add x_0_1 sub x_0_10 x_0_11": 1,
           "add x_0_1 sub x_0_9 x_0_10": 1,
           "add x_0_1 sub x_0_8 x_0_9": 1,
           "add x_0_1 sub x_0_7 x_0_8": 1
           }
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() in dic:
                print(line)


def get_base_test_Set_bin(path, save_path, test_type):
    """
    将测试集转化为二进制bin文件
    """
    with open(path, 'r', encoding='utf-8') as f1:
        with open(save_path + f'/{test_type}.src', 'w', encoding='utf-8') as f2:
            with open(save_path + f'/{test_type}.tgt', 'w', encoding='utf-8') as f3:
                reader = csv.reader(f1)
                for row in reader:
                    seq = row[1][1:]
                    print(seq)
                    seq_lis = seq.split(',')
                    res = process_oeis_input_seq(seq_lis, 15)
                    print(res)
                    # exit()
                    f2.write(res + '\n')
                    f3.write('eos' + '\n')


def sum_res(path1, path2, path3, path4, save_path):
    """
    汇总结果
    """
    with open(path1, 'r', encoding='utf-8') as f1:
        with open(path2, 'r', encoding='utf-8') as f2:
            with open(path3, 'r', encoding='utf-8') as f3:
                with open(path4, 'r', encoding='utf-8') as f4:
                    with open(save_path, 'w', encoding='utf-8') as f5:
                        reader1 = csv.reader(f1)
                        reader2 = csv.reader(f2)
                        reader3 = csv.reader(f3)
                        reader4 = csv.reader(f4)

                        writer = csv.writer(f5)

                        for row in reader1:
                            writer.writerow(row)
                        for row in reader2:
                            writer.writerow(row)
                        for row in reader3:
                            writer.writerow(row)
                        for row in reader4:
                            writer.writerow(row)


def hebing_res(path1, path2, save_path):
    """
    合并两个结果
    """

    with open(path1, 'r', encoding='utf-8') as f1:
        with open(path2, 'r', encoding='utf-8') as f2:
            with open(save_path, 'w', encoding='utf-8-sig', newline='') as f3:
                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                writer = csv.writer(f3)
                for row in reader1:
                    writer.writerow(row)
                for row in reader2:
                    writer.writerow(row)


# 测试模型的生成公式的acc: 该acc不是自学习累积求解正确公式的数量，而是对每一个序列预测后n项的acc
def return_test_res(seq_name, seq_input, formula_lis, input_seq_len, writer):
    sum_max = 0
    sum_max2 = 0
    seq_input = seq_input.replace(" ", '')
    # print("seq_input:",seq_input)
    seq_input_lis = seq_input.split(',')
    seq_input = process_oeis_input_seq(seq_input_lis, input_seq_len + 10)
    # f.write(seq_input)
    src = seq_input.split(" ")
    seq_length = len(seq_input_lis)

    # print("seq_input_lis:",seq_input_lis)
    # print("len_seq_input_lis:",len(seq_input_lis))
    # for i,num in enumerate(seq_input_lis):
    #     print(i,num,end=' ')
    # print()

    for formula in formula_lis:
        hyp = formula.split()
        if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
            continue
        pre_1_res = 0
        pre_10_res = 0
        pre_all_res = 0
        sum_max = 0
        sum_max2 = 0

        list_seq, eq_hyp = env.pre_next_term(src, formula.split(), n_predict=input_seq_len + 1)

        list_seq = [str(num) for num in list_seq]
        # print("formula: ",formula)
        # print("list_seq:",list_seq)
        # print("len_list_seq:",len(list_seq))
        # for i, num in enumerate(list_seq):
        #     print(i, num, end=' ')
        # print()
        real_pre1 = seq_input_lis[input_seq_len:input_seq_len + 1]
        pred_pre1 = list_seq[input_seq_len:input_seq_len + 1]
        if len(real_pre1) != 0 and real_pre1 == pred_pre1:
            pre_1_res = 1

            list_seq, eq_hyp = env.pre_next_term(src, formula.split(), n_predict=input_seq_len + 10)
            list_seq = [str(num) for num in list_seq]
            real_10_str = ','.join(seq_input_lis[input_seq_len:input_seq_len + 10])
            pre_10_str = ','.join(list_seq[input_seq_len:input_seq_len + 10])
            if real_10_str == pre_10_str:
                pre_10_res = 1

                list_seq, eq_hyp = env.pre_next_term(src, formula.split(), n_predict=seq_length)
                list_seq = [str(num) for num in list_seq]
                real_all_str = ','.join(seq_input_lis)
                pre_all_str = ','.join(list_seq)
                if real_all_str == pre_all_str:
                    pre_all_res = 1
                    # break
        if sum_max < pre_1_res + pre_10_res + pre_all_res:
            sum_max = pre_1_res + pre_10_res + pre_all_res
        if sum_max == 3:
            writer.writerow(
                [seq_name, ',' + real_all_str, formula, real_pre1[0], pred_pre1[0], ',' + real_10_str, ',' + pre_10_str,
                 ',' + real_all_str, ',' + pre_all_str])
            break
        # print(seq_input_lis[input_seq_len:input_seq_len+1], list_seq[input_seq_len:input_seq_len+1])
        # print("real_10_str:", real_10_str)
        # print("pre_10_str:", pre_10_str)
        # print("real_all_str:", real_all_str)
        # print("pre_all_str:", pre_all_str

    # if sum_max!=3:
    #     for formula in formula_lis:
    #         hyp = formula.split()
    #         if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
    #             continue
    #         pre_1_res = 0
    #         pre_10_res = 0
    #         pre_all_res = 0
    #         sum_max2 = 0
    #
    #         init_const = []
    #         const_index_lis=[]
    #         formula_lis=formula.split()
    #         for i, sym in enumerate(formula_lis):
    #             if sym.isdigit():
    #                 init_const.append(sym)
    #                 const_index_lis.append(i)
    #         init_const = np.array([int(num) for num in init_const])
    #         # init_const = np.array([1.0]*len(const_index_lis))
    #
    #         _, pred_xishu = recur_opti(src, formula, init_const)
    #         # print("const_index_lis:",const_index_lis)
    #         # print("formula_lis:",formula_lis)
    #         if len(const_index_lis)>0:
    #             for i,const in enumerate(pred_xishu):
    #                 formula_lis[const_index_lis[i]]=str(const)
    #
    #
    #         list_seq, eq_hyp = env.pre_next_term(src, formula_lis, n_predict=input_seq_len + 1)
    #
    #         list_seq = [str(num) for num in list_seq]
    #         # print("formula: ",formula)
    #         # print("list_seq:",list_seq)
    #         # print("len_list_seq:",len(list_seq))
    #         # for i, num in enumerate(list_seq):
    #         #     print(i, num, end=' ')
    #         # print()
    #         real_pre1 = seq_input_lis[input_seq_len:input_seq_len + 1]
    #         pred_pre1 = list_seq[input_seq_len:input_seq_len + 1]
    #         if len(real_pre1) != 0 and real_pre1 == pred_pre1:
    #             pre_1_res = 1
    #
    #             list_seq, eq_hyp = env.pre_next_term(src, formula.split(), n_predict=input_seq_len + 10)
    #             list_seq = [str(num) for num in list_seq]
    #             real_10_str = ','.join(seq_input_lis[input_seq_len:input_seq_len + 10])
    #             pre_10_str = ','.join(list_seq[input_seq_len:input_seq_len + 10])
    #             if real_10_str == pre_10_str:
    #                 pre_10_res = 1
    #
    #                 list_seq, eq_hyp = env.pre_next_term(src, formula.split(), n_predict=seq_length)
    #                 list_seq = [str(num) for num in list_seq]
    #                 real_all_str = ','.join(seq_input_lis)
    #                 pre_all_str = ','.join(list_seq)
    #                 if real_all_str == pre_all_str:
    #                     pre_all_res = 1
    #                     # break
    #         if sum_max2 < pre_1_res + pre_10_res + pre_all_res:
    #             sum_max2 = pre_1_res + pre_10_res + pre_all_res
    #         if sum_max2 == 3:
    #             writer.writerow([seq_name, ',' + real_all_str, formula, real_pre1[0], pred_pre1[0], ',' + real_10_str,
    #                              ',' + pre_10_str, ',' + real_all_str, ',' + pre_all_str])
    #             break
    #         # print("sum_max2:",sum_max2)
    #         # print(seq_input_lis[input_seq_len:input_seq_len+1], list_seq[input_seq_len:input_seq_len+1])
    #         # print("real_10_str:", real_10_str)
    #         # print("pre_10_str:", pre_10_str)
    #         # print("real_all_str:", real_all_str)
    #         # print("pre_all_str:", pre_all_str

    if sum_max == 1:
        res = [1, 0, 0]
    elif sum_max == 2:
        res = [1, 1, 0]
    elif sum_max == 3:
        res = [1, 1, 1]
    else:
        res = [0, 0, 0]

    if sum_max2 == 1:
        res2 = [1, 0, 0]
    elif sum_max2 == 2:
        res2 = [1, 1, 0]
    elif sum_max2 == 3:
        res2 = [1, 1, 1]
    else:
        res2 = [0, 0, 0]

    return res, res2


def eval_testset(path, path_res, nbest, input_seq_len, correct_res_save_path, acc_res_save_path):
    """
    评估测试集的acc
    """
    with open(path, 'r', encoding='utf-8') as f:
        with open(path_res, 'r', encoding='utf-8') as f2:
            with open(correct_res_save_path, 'w', encoding='utf-8-sig', newline='') as f3:
                with open(acc_res_save_path, 'w', encoding='utf-8') as f4:
                    reader = csv.reader(f)
                    lines = f2.readlines()
                    writer = csv.writer(f3)
                    writer.writerow(
                        ['seq_name', "input_seq", "pred_formula", 'real_pre1', "pred_pre1", 'real_pre10', "pred_pre10",
                         'real_pre_all', "pred_pre_all"])
                    count_pre1_correct = 0
                    count_pre10_correct = 0
                    count_preall_correct = 0
                    count_pre1_correct2 = 0
                    count_pre10_correct2 = 0
                    count_preall_correct2 = 0

                    data_size = 10000
                    count = 0
                    for i, row in enumerate(tqdm(reader)):
                        if count > data_size:
                            break
                        count += 1
                        seq_name = row[0]
                        seq = row[1][1:]
                        formula_lis = lines[i * nbest:(i + 1) * nbest]
                        # print(len(formula_lis))
                        res, res2 = return_test_res(seq_name, seq, formula_lis, input_seq_len, writer)
                        count_pre1_correct += res[0]
                        count_pre10_correct += res[1]
                        count_preall_correct += res[2]

                        count_pre1_correct2 += res2[0]
                        count_pre10_correct2 += res2[1]
                        count_preall_correct2 += res2[2]
                        # print(row)
                        # print("formula:",formula_lis)
                        # print(res)
                        # exit()
                    count_pre1_correct /= data_size
                    count_pre10_correct /= data_size
                    count_preall_correct /= data_size

                    count_pre1_correct2 /= data_size
                    count_pre10_correct2 /= data_size
                    count_preall_correct2 /= data_size
                    # print('预测后1位acc', count_pre1_correct)
                    # print('预测后10位acc', count_pre10_correct)
                    # print('预测所有位acc', count_preall_correct)

                    # print('优化常量-预测后1位acc', count_pre1_correct2)
                    # print('优化常量-预测后10位acc', count_pre10_correct2)
                    # print('优化常量-预测所有位acc', count_preall_correct2)
                    f4.write(f"预测后1位acc: {count_pre1_correct}\n")
                    f4.write(f"预测后10位acc: {count_pre10_correct}\n")
                    f4.write(f"预测所有位acc: {count_preall_correct}\n")
                    f4.write(f"优化常量-预测后1位acc: {count_pre1_correct2}\n")
                    f4.write(f"优化常量-预测后10位acc: {count_pre10_correct2}\n")
                    f4.write(f"优化常量-预测所有位acc: {count_preall_correct2}\n")

                    return "{:.2f}%".format(count_pre1_correct*100),"{:.2f}%".format(count_pre10_correct*100),"{:.2f}%".format(count_preall_correct*100)


def output_example(path_seq, path_formula):
    """
    输出部分训练数据
    """
    env, file_name = get_env()
    with open(path_seq, 'r', encoding='utf-8') as f1:
        with open(path_formula, 'r', encoding='utf-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

            for i in range(len(lines2)):
                if i > 10:
                    break
                seq = decode(lines1[i].strip().split())
                seq_str = ', '.join([str(num) for num in seq])

                formula, res = env.trans_qianzhui_formula(lines2[i].split())
                print(f"sequence{i}: ", seq_str)
                print(f"formula{i}: ", formula)
                print()


if __name__ == '__main__':
    # obtain_check_csv('/home/skm21/recur-main/data/10M/all_init_term_10len_formula.csv')
    # obtain_init_term()
    # init_term_dict=obtain_init_term("/home/skm21/fairseq-0.12.0/data_oeis/stripped",12)

    # formula_list=select_short_formula('/home/skm21/recur-main/data/10M/tgt7-12.txt',10)

    # find_special_formula('/home/skm21/recur-main/data/10M/tgt7-12.txt','/home/skm21/recur-main/data/10M/tgt7-12_special.txt')
    # get_an7_12dataset('/home/skm21/recur-main/data/10M/','/home/skm21/recur-main/data/10M/an7-12dataset/')
    # look_new_res_acc('/home/skm21/fairseq-0.12.0/checkpoints/baseline_an7_12/result/look.csv','/home/skm21/fairseq-0.12.0/data_oeis/base_1w_testdata_douhao.csv')
    # obtain_new_train_data('/home/skm21/fairseq-0.12.0/checkpoints/baseline_an7_12_2/iter101/')
    # real_size('/home/skm21/fairseq-0.12.0/checkpoints/baseline_an7_12_2/iter101/result/train.csv')

    # process_data_delete('/home/skm21/fairseq-0.12.0/checkpoints/baseline_an7_12/result/zuijia_res.csv','/home/skm21/fairseq-0.12.0/checkpoints/baseline_an7_12/result/train.csv','/home/skm21/fairseq-0.12.0/checkpoints/baseline_an7_12/result/train_new.csv')

    # get_all_formula('/home/skm21/recur-main/data/10M/data.prefix','/home/skm21/recur-main/data/10M/all_formula.txt')
    # init_term='+ 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0'
    # obatain_oeisseq(init_term)

    # generate_seq('/home/skm21/recur-main/data/10M/all_formula.txt','/home/skm21/recur-main/data/10M/train_data.csv')

    # init_term1, init_term2, init_term3, init_term4, init_term5, init_term6, init_term7, init_term8, init_term9, init_term10, init_term11, init_term12 = get_all_initterm_list()
    #
    # print(len(init_term1))
    # print(len(init_term2))
    # print(len(init_term3))
    # print(len(init_term4))
    # print(len(init_term5))
    # print(len(init_term6))
    # print(len(init_term7))
    # print(len(init_term8))
    # print(len(init_term9))
    # print(len(init_term10))
    # print(len(init_term11))
    # print(len(init_term12))
    # termlist = [init_term1, init_term2, init_term3, init_term4, init_term5, init_term6, init_term7, init_term8,
    #             init_term9, init_term10, init_term11, init_term12]
    #
    # deep_formula=5
    # init_term_list_all = termlist[deep_formula - 1]
    # index_random = random.randint(0, len(init_term_list_all) - 1)
    # init_term = init_term_list_all[index_random]
    # init_list=init_term.split(',')
    # print(init_list)

    # check_2500wData('/home/skm21/recur-main/data/lineRecur/train.tgt')
    # get_base_test_Set_bin('/home/skm21/fairseq-0.12.0/data_oeis/base_1w_testdata_douhao.csv','/home/skm21/fairseq-0.12.0/data_oeis/src_tgt_testSet','base')
    # get_base_test_Set_bin('/home/skm21/fairseq-0.12.0/data_oeis/sign_1w_testdata_douhao.csv','/home/skm21/fairseq-0.12.0/data_oeis/src_tgt_testSet','sign')
    # get_base_test_Set_bin('/home/skm21/fairseq-0.12.0/data_oeis/1wan_easy_testdata_35.csv','/home/skm21/fairseq-0.12.0/data_oeis/src_tgt_testSet','easy')
    # get_base_test_Set_bin('/home/skm21/fairseq-0.12.0/data_oeis/1wan_easy_testdata_25.csv','/home/skm21/fairseq-0.12.0/data_oeis/src_tgt_testSet','easy_25')
    # exit()
    # sum_res('/home/skm21/fairseq-0.12.0/checkpoints/lineRecur_500w/base_test_res/zuijia_res.csv','/home/skm21/fairseq-0.12.0/checkpoints/lineRecur_deep_sum/zuijia_res.csv','/home/skm21/fairseq-0.12.0/checkpoints/merge_data_4000w/sum_test_res/zuijia_res.csv','/home/skm21/fairseq-0.12.0/checkpoints/init_term_select_oeis/zuijia_res.csv','/home/skm21/fairseq-0.12.0/checkpoints/res_sum/zuijia_res4.csv')

    # res_path = '/home/skm21/fairseq-0.12.0/checkpoints/lineRecur_500w/base_test_res/zuijia_sum.csv'
    res_path = '/home/skm21/fairseq-0.12.0/checkpoints/merge_data_4500w/all_oeistest/zuijia_res.csv'
    # hebing_res('/home/skm21/fairseq-0.12.0/result/test/merge_data_4500w_checkpoint_2_140000.pt_sign_len25_beamsize64_nbest64/result.csv','/home/skm21/fairseq-0.12.0/checkpoints/res_sum/zuijia_res4.csv',res_path )
    # # exit()
    easy_acc_1w = test_1wan_easy_acc("/home/skm21/fairseq-0.12.0/data_oeis/1wan_easy_testdata_35.csv", res_path)
    sign_acc_1w = test_1wan_easy_acc("/home/skm21/fairseq-0.12.0/data_oeis/1wan_sign_testdata_35.csv", res_path)
    base_acc_1w = test_1wan_easy_acc("/home/skm21/fairseq-0.12.0/data_oeis/1wan_base_testdata_35.csv", res_path)
    print(easy_acc_1w)
    print(sign_acc_1w)
    print(base_acc_1w)

    # seq='1,1,2,3,5,7,11,15,22,30,42,56,77,101,135,176,231,297,385,490,627,792,1002,1255,1575,1958,2436,3010,3718,4565,5604,6842,8349,10143,12310,14883,17977,21637,26015,31185,37338,44583,53174,63261,75175,89134,105558,124754,147273,173525'
    # res=return_test_res(seq)

    # nbest=32 # 解码公式候选项数量
    # input_seq_len=25 #输入序列的长度
    # eval_testset("/home/skm21/fairseq-0.12.0/data_oeis/1wan_easy_testdata_35.csv",'/home/skm21/fairseq-0.12.0/checkpoints/sum_data_sl/result/easy_res_nbest32/pre_res.txt',nbest,input_seq_len)

    path_seq = '/home/skm21/recur-main/data/lineRecur_deep_sum_new/val.src'
    path_formula = '/home/skm21/recur-main/data/lineRecur_deep_sum_new/val.tgt'
    # output_example(path_seq,path_formula)
