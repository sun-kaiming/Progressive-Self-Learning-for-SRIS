import csv
import json
import pickle
from recur.utils.get_env import get_env
from recur.utils.decode import decode
from tqdm import tqdm
from datetime import datetime
import sys  # 导入sys模块
from recur.utils.split_oesi_seq import process_oeis_input_seq
from recur.utils.iter_need_method import obtain_csv_res, obtain_csv_res1, formula_symbolic_nums, return_recurrece_deep, \
    trans_formula_INT
from collections import defaultdict
from process_result.process_data import get_oeis_init_term_dict

sys.setrecursionlimit(300000)  # 将默认的递归深度修改为300000
import os
from multiprocessing import Pool
import random
from recur.utils.node_class import Node

root = ''


# with open('/amax/users/skm21/fairseq-0.12.0_213/recur/utils/oeis_tree_save_dict2.pkl', 'rb') as f:
#     root = pickle.load(f)

def isinclude_num(node, num):  # 检测某个节点的孩子节点中是否包含数据num
    child_node_list = node.l_child

    # for child_node in child_node_list:
    #     if child_node.val==num:
    #         return True,child_node
    # for key,value in child_node_list.items():
    #     print(key)
    # print(child_node_list)
    # exit()

    if num in child_node_list:
        return True, child_node_list[num]

    return False, None


def pre(tn):  # 先序遍历
    global res
    if not tn:
        return
    res += str(tn.val) + ' '
    for i in tn.l_child:
        pre(i)
    return (res)


def get_data_list():
    with open('/amax/users/skm21/OpenNMT-py-master/data_recur/stripped', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        oeis_list = []
        count = 0
        for i in range(4, len(lines)):
            # print(lines[i])
            dict_temp = {}
            if "A" in lines[i]:
                seq_name = lines[i][:7]
                # print(seq_name)
                # exit()
                oeis_seq = lines[i][9:-2].strip().split(',')
                # print(oeis_seq)
                # exit()
                dict_temp[seq_name] = oeis_seq
                oeis_list.append(dict_temp)
                # if count>10:
                #     break
                # count+=1
    # print(oeis_list)
    # exit()
    return oeis_list


def check_seq_in_oeis(root, list_seq):
    # with open('/amax/users/skm21/OpenNMT-py-master/recur/utils/oeis_tree_save.pkl', 'rb') as f:
    #     root = pickle.load(f)
    temp_node = root
    for i, term in enumerate(list_seq):
        flag, return_node = isinclude_num(temp_node, str(term))
        temp_node = return_node

        seq_name = 'unk'
        if return_node != None:
            seq_name = return_node.name
        if flag == False:
            # print(list_seq,"不在oeis中")
            return False, None, -1

        if temp_node.l_child == {}:
            # print(list_seq, "在oeis中")
            return True, seq_name, i + 1

    # print(list_seq, "在oeis中")
    # return True, seq_name
    return False, "recheck", -1


def check_seq_in_oeis2(root, list_seq):
    # with open('/amax/users/skm21/OpenNMT-py-master/recur/utils/oeis_tree_save.pkl', 'rb') as f:
    #     root = pickle.load(f)
    temp_node = root
    seq_name_list = []
    len_list = []
    for i, term in enumerate(list_seq):

        flag, return_node = isinclude_num(temp_node, str(term))
        temp_node = return_node
        seq_name = 'unk'

        # print("term:",term)
        if return_node != None:
            seq_nameLis = return_node.name
            if temp_node.flag == 1:
                seq_name_list.extend(seq_nameLis)
                for j in range(len(seq_nameLis)):
                    len_list.append(i + 1)

        if flag == False:
            # print(list_seq,"不在oeis中")
            if len(seq_name_list) != 0:
                # print("111111111")
                return True, seq_name_list, len_list
            else:
                return False, None, -1

        if temp_node.l_child == {}:
            # print("seq_name_list:", seq_name_list)
            # print("len(seq_name_list):", len(seq_name_list))
            # print(list_seq, "11在oeis中")
            return True, seq_name_list, len_list

    return False, "recheck", -1


def save_tree_file():  # 保存oeis转化成的树，方便快速加载
    res = ''
    # oeis_list=[[1, 1, 1, 1, 2, 3, 6, 11, 23, 47, 106, 235, 551, 1301, 3159, 7741, 19320, 48629, 123867, 317955, 823065, 2144505, 5623756, 14828074, 39299897, 104636890, 279793450, 751065460, 2023443032, 5469566585, 14830871802, 40330829030, 109972410221, 300628862480, 823779631721, 2262366343746, 6226306037178],[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271]]
    oeis_list = get_data_list()
    root = Node("S")
    for seq_dict in oeis_list:
        temp_node = root
        for seq_name, seq_list in seq_dict.items():
            for term in seq_list:
                if temp_node.l_child == {}:

                    new_node = Node(term, seq_name)
                    temp_node.add_child(new_node)
                    temp_node = new_node
                else:

                    flag, return_node = isinclude_num(temp_node, term)
                    if flag:
                        temp_node = return_node
                        # continue
                    else:
                        new_node = Node(term, seq_name)
                        temp_node.add_child(new_node)
                        temp_node = new_node
    # list_test = [1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 3, 2, 2, 4, 2, 2, 4, 2, 3, 4, 4, 2, 3, 4, 2, 6, 3, 2, 6, 4, 3, 4, 4, 4,
    #              6, 4, 2, 6, 4, 4, 8, 4, 3, 6, 4, 4, 5, 4, 4, 6, 6, 4, 6, 6, 4, 8, 4, 2, 9, 4, 6, 8, 4, 4, 8, 8, 3, 8,
    #              8, 4, 7, 4, 4, 10]
    # list_test = [str(i) for i in list_test]
    # # print(list_test)
    #
    # check_seq_in_oeis(root, list_test)

    tree_str = pickle.dumps(root)
    with open('oeis_tree_save_dict.pkl', 'wb') as f:
        f.write(tree_str)
    print("123")


def more_jincheng_worker(args):
    row, temp_res_path = args

    src = row[0].replace('=', '')
    src = src.replace('\ufeff', '')
    src = src.split(" ")

    with open(temp_res_path, 'a', encoding='utf-8-sig') as f:
        # with open('/amax/users/skm21/fairseq-0.12.0/checkpoints/iter1/result/recode.txt', 'w', encoding='utf-8') as f2:

        writer = csv.writer(f)

        for formula in row[1:]:
            hyp = formula.split()
            if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                continue
            list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)
            recheck_num = 35
            flag = 0
            while True:
                if "error" not in list_seq:
                    # res, seq_name, len_seq = check_seq_in_oeis(root, list_seq)
                    # # print("res:",res)
                    # # print("seq_name:",seq_name)
                    # if res == True:
                    #     deep_formula = return_recurrece_deep(formula)
                    #     if deep_formula < len_seq:
                    #         # print('112222')
                    #         if len_seq >= 7:
                    #             seq_name = seq_name.replace("\ufeff", "")
                    #             seq_name = seq_name.replace('﻿A', 'A')
                    #             # print(seq_name)
                    #             writer.writerow([seq_name, formula,
                    #                              ", ".join(str(i) for i in list_seq[:len_seq])])
                    #             # print(seq_name)
                    #
                    #     break

                    res, seq_name_list, len_seq_list = check_seq_in_oeis2(root, list_seq)
                    if res == True:
                        flag2 = 1
                        deep_formula = return_recurrece_deep(formula)
                        for i, seq_name in enumerate(seq_name_list):

                            len_seq = len_seq_list[i]
                            if deep_formula < len_seq:
                                if len_seq >= 13:
                                    seq_name = seq_name.replace("\ufeff", "")
                                    seq_name = seq_name.replace('﻿A', 'A')
                                    writer.writerow([seq_name, formula,
                                                     ", ".join(str(i) for i in list_seq[:len_seq])])
                        break

                    elif seq_name_list == "recheck":
                        recheck_num += 50
                        if flag == 1:
                            break
                        if recheck_num > 377:
                            flag = 1
                            recheck_num = 378

                        list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=recheck_num)
                        if list_seq == 'error':
                            break
                    else:
                        break
                else:
                    break


def more_jincheng_worker_pre1(args):
    row, temp_res_path = args
    # final_res_dict=mes[1]

    src = row[0].replace('=', '')
    src = src.split(" ")
    # print("#"*100)
    # print(src)
    # with open('/amax/users/skm21/fairseq-0.12.0/data_oeis/wolfram_test/findlinarrecurrence_testdata.csv','r',encoding='utf-8')as f2:
    #     reader=csv.reader(f2)
    #     list_seq_init=[]
    #     for i,row2 in enumerate(reader):
    #         if i<10001:
    #             list_seq_init.append(row2[1])

    with open(temp_res_path, 'a', encoding='utf-8-sig') as f:
        # with open('/amax/users/skm21/fairseq-0.12.0/checkpoints/iter1/result/recode.txt', 'w', encoding='utf-8') as f2:

        writer = csv.writer(f)

        # formula_dict=defaultdict(int)
        # for formula in row[1:]:
        #     formula_dict[formula]+=1

        # for formula, nums in formula_dict.items():
        for formula in row[1:]:
            # print("第",i,"个公式")
            # if len(src) < 3:  # 输入序列的项数必须>=3，否则产生的公式没什么用
            #     break
            hyp = formula.split()
            if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                continue
            # print("hyp",hyp)
            # for j,src_init in enumerate(list_seq_init):
            # list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)

            src_list = decode(src)
            # print("len_src_list", len(src_list))
            # print(src_list)
            list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=len(src_list) + 1)

            # exit()
            # src_lis = decode(src)
            # recheck_num = 35
            # flag = 0
            # while True:
            #     if "error" not in list_seq:
            # print('111111111111111111')

            res, seq_name, len_seq = check_seq_in_oeis2(root, list_seq)
            if res == True:
                deep_formula = return_recurrece_deep(formula)
                if deep_formula < len_seq:
                    # print('222222222222222222')
                    if len_seq >= 26:
                        seq_name = seq_name.replace("\ufeff", "")
                        seq_name = seq_name.replace('﻿A', 'A')
                        writer.writerow([seq_name, formula,
                                         ", ".join(str(i) for i in list_seq[:len_seq])])
                        # print('3333333333333333333')

                    # src_lis_str = ','.join(str(num) for num in src_lis)
                    # list_seq_str = ','.join(str(num) for num in list_seq)
                    # # print(src_lis_str)
                    # # print(list_seq_str)
                    # if list_seq_str in src_lis_str or src_lis_str in list_seq_str:
                    #     f2.write(src_lis_str + '\n')
                # break
                #     elif seq_name == "recheck":
                #         recheck_num += 50
                #         if flag == 1:
                #             break
                #         if recheck_num > 377:
                #             flag = 1
                #             recheck_num = 378
                #
                #         list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=recheck_num)
                #         if list_seq == 'error':
                #             break
                #     else:
                #         break
                # else:
                #     break


def more_jincheng_worker_pre10(args):
    row, temp_res_path = args
    # final_res_dict=mes[1]

    src = row[0].replace('=', '')
    src = src.split(" ")
    # print("#"*100)
    # print(src)
    # with open('/amax/users/skm21/fairseq-0.12.0/data_oeis/wolfram_test/findlinarrecurrence_testdata.csv','r',encoding='utf-8')as f2:
    #     reader=csv.reader(f2)
    #     list_seq_init=[]
    #     for i,row2 in enumerate(reader):
    #         if i<10001:
    #             list_seq_init.append(row2[1])

    with open(temp_res_path, 'a', encoding='utf-8-sig') as f:
        # with open('/amax/users/skm21/fairseq-0.12.0/checkpoints/iter1/result/recode.txt', 'w', encoding='utf-8') as f2:

        writer = csv.writer(f)

        # formula_dict=defaultdict(int)
        # for formula in row[1:]:
        #     formula_dict[formula]+=1

        # for formula, nums in formula_dict.items():
        for formula in row[1:]:
            # print("第",i,"个公式")
            # if len(src) < 3:  # 输入序列的项数必须>=3，否则产生的公式没什么用
            #     break
            hyp = formula.split()
            if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                continue
            # print("hyp",hyp)
            # for j,src_init in enumerate(list_seq_init):
            # list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)

            src_list = decode(src)
            # print("len_src_list", len(src_list))
            # print(src_list)
            list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=len(src_list) + 10)

            # exit()
            # src_lis = decode(src)
            # recheck_num = 35
            # flag = 0
            # while True:
            #     if "error" not in list_seq:
            # print('111111111111111111')

            res, seq_name, len_seq = check_seq_in_oeis2(root, list_seq)
            if res == True:
                deep_formula = return_recurrece_deep(formula)
                if deep_formula < len_seq:
                    # print('222222222222222222')
                    if len_seq >= 35:
                        # print(len(list_seq),list_seq)

                        seq_name = seq_name.replace("\ufeff", "")
                        seq_name = seq_name.replace('﻿A', 'A')
                        writer.writerow([seq_name, formula,
                                         ", ".join(str(i) for i in list_seq[:len_seq])])
                        # print('3333333333333333333')

                    # src_lis_str = ','.join(str(num) for num in src_lis)
                    # list_seq_str = ','.join(str(num) for num in list_seq)
                    # # print(src_lis_str)
                    # # print(list_seq_str)
                    # if list_seq_str in src_lis_str or src_lis_str in list_seq_str:
                    #     f2.write(src_lis_str + '\n')
                # break
                #     elif seq_name == "recheck":
                #         recheck_num += 50
                #         if flag == 1:
                #             break
                #         if recheck_num > 377:
                #             flag = 1
                #             recheck_num = 378
                #
                #         list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=recheck_num)
                #         if list_seq == 'error':
                #             break
                #     else:
                #         break
                # else:
                #     break


def check_get_final_res(path):  # 检测束搜索结果有多少在oeis中 ，将结果保存在csv文件里
    pool = Pool(40)  # 定义一个进程池，最大进程数48
    print("----start----")
    time2 = datetime.now()
    # print("加载文件时间：", time2 - sta_time)
    temp_res_path = path + 'temp_res.csv'
    beam_res_path = path + 'generate-test.csv'

    with open(beam_res_path, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        next(reader)
        for j in range(1):
            for k, row in enumerate(tqdm(reader)):
                if k % 1000 == 0:
                    print('\n检验第' + str(k) + "行的seq")
                # args.append((row, temp_res_path))
                args = (row, temp_res_path)
                pool.apply_async(more_jincheng_worker, (args,))

        pool.close()
        # 等待po中所有子进程执行完成，必须放在close语句之后
        pool.join()
        print("-----end-----")
        print("检测seq时间：", datetime.now() - time2)


def check_get_autoGenerate_res(path):  # 检测束搜索结果有多少在oeis中 ，将结果保存在csv文件里
    pool = Pool(1)  # 定义一个进程池，最大进程数48
    print("----start----")
    time2 = datetime.now()
    # print("加载文件时间：", time2 - sta_time)
    temp_res_path = path + 'temp_res.csv'
    # src_path = path + 'src7-12.txt'
    # tgt_path = path + 'tgt7-12.txt'
    # init_data_dict={}
    # with open(src_path, 'r', encoding='utf-8') as f1:
    #     with open(tgt_path, 'r', encoding='utf-8') as f2:
    #         for j in range(1):
    #             for i in tqdm(range(15900873)):
    #                 # if i<1000000:
    #                 line1 = f1.readline().strip()
    #                 line2 = f2.readline().strip()
    #                 init_term=','.join( str(num) for num in decode(line1.split())[:12])
    #                 if init_term not in init_data_dict:
    #                     init_data_dict[init_term]=[line2]
    #                 else:
    #                     init_data_dict[init_term].append(line2)
    #                 # else:
    #                 #     break
    #
    # with open(path+'trans_src_tgt.csv','w',encoding='utf-8')as f:
    #     writer=csv.writer(f)
    #     for k,v in init_data_dict.items():
    #         row=[]
    #         row.append(k)
    #         row.extend(v)
    #         writer.writerow(row)
    #
    # print(len(init_data_dict))
    # exit()

    with open(path + 'trans_src_tgt_change.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        list_len = [1]
        for j in range(1):
            for i, row in enumerate(tqdm(reader)):
                # line1 = f1.readline()
                # line2 = f2.readline()
                # print('检验第' + str(k) + "行的seq")
                # args.append((row, temp_res_path))
                # row=[]
                # row.append(line1)
                # row.append(line2)

                # print(row[0])

                # if i>96250216 and i<97250246:
                #     print(i,len(row),row[0])
                #     list_len.append(len(row))
                #     # print(row[0])
                # if i>97250246:
                #     print(max(list_len))
                #     exit()

                # print(row[0])
                # exit()
                # dict_init = get_oeis_init_term_dict()
                # print(row[0])
                # if row[0] in dict_init:
                # print(row[0])

                args = (row, temp_res_path)
                # more_jincheng_worker(args)
                # exit()
                pool.apply_async(more_jincheng_worker, (args,))

        pool.close()
        # 等待po中所有子进程执行完成，必须放在close语句之后
        pool.join()
        print("-----end-----")
        print("检测seq时间：", datetime.now() - time2)


def check_correct_trainset(path, new_train_set_path):
    env, file_name = get_env()
    # with open(path_src, 'r', encoding='utf-8') as f1:
    #     with open(path_tgt, 'r', encoding='utf-8') as f2:
    with open(path, 'r', encoding='utf-8') as f:
        with open(new_train_set_path + "new_trainset.csv", 'w', encoding='utf-8-sig') as f3:
            reader = csv.reader(f)
            writer = csv.writer(f3)
            # writer.writerow(["seq_name", 'formula_前缀', 'formula_中缀', 'sequence'])

            count = 0
            count2 = 0

            for i, row in enumerate(reader):
                src = row[0][1:].strip().replace('=', '')
                src = src.split(",")
                src = process_oeis_input_seq(src, 1125).split()
                hyp = row[1].strip().split()

                if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                    continue
                list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)
                src_lis = decode(src)
                recheck_num = 35
                flag = 0
                while True:
                    if list_seq != 'error':
                        res, seq_name, len_seq = check_seq_in_oeis(root, list_seq)
                        if res == True:
                            deep_formula = return_recurrece_deep(row[1].strip())
                            if deep_formula < len_seq:
                                writer.writerow([seq_name.replace("\ufeff", ""), row[1].strip(), eq_hyp,
                                                 ", ".join(str(i) for i in list_seq[:len_seq])])
                                count += 1
                                print("src_lis:", src_lis)
                                src_lis_str = ','.join(str(num) for num in src_lis)
                                list_seq_str = ','.join(str(num) for num in list_seq)

                                if list_seq_str in src_lis_str or src_lis_str in list_seq_str:
                                    count2 += 1
                            break
                        elif seq_name == "recheck":
                            recheck_num += 50
                            if flag == 1:
                                break
                            if recheck_num > 377:
                                flag = 1
                                recheck_num = 378

                            list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=recheck_num)
                            if list_seq == 'error':
                                break
                        else:
                            break
                    else:
                        break

            print("训练集中正确公式的个数：", count)
            print("训练集中在预测序列的公式数量：", count2)


def check_autoGenerate_correct_data(path, res_path):
    env, file_name = get_env()
    with open(path, 'r', encoding='utf-8') as f1:
        with open(res_path, 'w', encoding='utf-8-sig', newline='') as f2:
            lines = f1.readlines()
            writer = csv.writer(f2)

            # writer.writerow(["seq_name", 'formula_前缀', 'formula_中缀', 'sequence'])

            count = 0
            count2 = 0
            for j in range(1):
                for line in tqdm(lines):
                    data_dict = json.loads(line)
                    src = data_dict['x1']
                    src = src.split(" ")
                    hyp = data_dict['x2'].split()
                    if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                        continue
                    list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)
                    src_lis = decode(src)

                    recheck_num = 35
                    flag = 0
                    while True:
                        if list_seq != 'error':
                            res, seq_name, len_seq = check_seq_in_oeis(root, list_seq)
                            if res == True:
                                deep_formula = return_recurrece_deep(data_dict['x2'])
                                if deep_formula < len_seq:
                                    writer.writerow([seq_name.replace("\ufeff", ""), data_dict['x2'], eq_hyp,
                                                     ", ".join(str(i) for i in list_seq[:len_seq])])
                                    count += 1

                                    src_lis_str = ','.join(str(num) for num in src_lis)
                                    list_seq_str = ','.join(str(num) for num in list_seq)

                                    if list_seq_str in src_lis_str or src_lis_str in list_seq_str:
                                        count2 += 1
                                break
                            elif seq_name == "recheck":
                                recheck_num += 50
                                if flag == 1:
                                    break
                                if recheck_num > 377:
                                    flag = 1
                                    recheck_num = 378

                                list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=recheck_num)
                                if list_seq == 'error':
                                    break
                            else:
                                break
                        else:
                            break

                print("训练集中正确公式的个数：", count)
                print("训练集中在预测自己徐磊的公式数量：", count2)


def process_temp_res(path):
    with open(path + 'temp_res.csv', 'r', encoding='utf-8') as f1:
        # with open(path + 'zuijia_res2.csv', 'r', encoding='utf-8') as f1:
        with open(path + 'final_res.json', 'w', encoding='utf-8') as f2:
            with open(path + 'final_formula.json', 'w', encoding='utf-8') as f4:
                with open(path + 'zuijia_res.csv', 'w', encoding='utf-8-sig', newline='') as f3:
                    dict_final = {}
                    dict_formula = {}
                    reader = csv.reader(f1)
                    writer = csv.writer(f3)
                    writer.writerow(["seq_name", 'formula_前缀', 'formula_中缀', 'sequence'])
                    count = 0
                    for row in reader:
                        key = row[0].replace("\ufeff", '')
                        if key in dict_final:
                            dict_final[key][1][row[1]] += 1
                        else:
                            dict_final[key] = (row[2], defaultdict(int))
                            dict_final[key][1][row[1]] += 1

                        count += 1

                    for seq_name, tuple_seq in dict_final.items():
                        if seq_name not in dict_formula:
                            dict_formula[seq_name] = (tuple_seq[0], [])
                        for k, v in tuple_seq[1].items():
                            dict_formula[seq_name][1].append(k)

                    json.dump(dict_final, f2)
                    json.dump(dict_formula, f4)

                    print("count:", count)
                    print("总共发现的oeis公式数量：", len(dict_formula))
                    print("总共发现的oeis公式数量2：", len(dict_final))

                    for seq, formual_lis in dict_formula.items():
                        min = 0
                        # print(seq)
                        for i, temp_formula in enumerate(formual_lis[1]):

                            if len(temp_formula.split()) < len(formual_lis[1][min].split()):
                                min = i
                            elif len(temp_formula.split()) == len(formual_lis[1][min].split()):
                                if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[1][min]):
                                    min = i
                        zhongzhui_formula, res = env.trans_qianzhui_formula(formual_lis[1][min].split())
                        if res == len(formual_lis[1][min].split()):
                            writer.writerow([seq, formual_lis[1][min], zhongzhui_formula, formual_lis[0]])

                    # print("总共发现的oeis公式数量：", len(dict_final))


def process_temp_res2(path_before, path, path2, env):
    with open(path + 'temp_res.csv', 'r', encoding='utf-8') as f1:
        with open(path_before + 'final_res.json', 'r', encoding='utf-8') as f2:
            with open(path + 'final_res.json', 'w', encoding='utf-8') as f3:
                with open(path2 + 'train.csv', 'w', encoding='utf-8-sig', newline='') as f4:
                    # dict_final = {}
                    dict_formula = {}
                    reader = csv.reader(f1)
                    writer = csv.writer(f4)

                    dict_final = json.load(f2)
                    # writer.writerow(["seq_name", 'formula_前缀', 'formula_中缀', 'sequence'])
                    count = 0
                    for row in reader:
                        key = row[0].replace("\ufeff", '')
                        if key in dict_final:
                            if row[1] in dict_final[key][1]:
                                dict_final[key][1][row[1]] += 1
                            else:
                                dict_final[key][1][row[1]] = 1

                        else:
                            dict_final[key] = (row[2], defaultdict(int))
                            dict_final[key][1][row[1]] += 1

                        count += 1

                    for seq_name, tuple_seq in dict_final.items():
                        if seq_name not in dict_formula:
                            dict_formula[seq_name] = (tuple_seq[0], [])
                        for k, v in tuple_seq[1].items():
                            dict_formula[seq_name][1].append(k)

                    json.dump(dict_final, f3)
                    # json.dump(dict_formula, f4)

                    # print("count:", count)
                    # print("总共发现的oeis公式数量：", len(dict_formula))
                    # print("总共发现的oeis公式数量2：", len(dict_final))

                    for seqname, formual_lis in dict_formula.items():
                        min_degree_idx = 0
                        min_len_idx = 0
                        # print(seq)
                        seq = formual_lis[0]
                        seq_len = len(seq.split(','))
                        if seq_len >= 13:

                            # 分别找出具有最小递推度和最少符号个数的公式下标min_degree、min_len
                            for i, temp_formula in enumerate(formual_lis[1]):
                                # if len(temp_formula.split()) < len(formual_lis[1][min].split()):
                                #     min = i
                                # elif len(temp_formula.split()) == len(formual_lis[1][min].split()):
                                #     if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[1][min]):
                                #         min = i
                                # 在利用奥卡姆剃刀原理选择最佳公式时，应该首先选择递推阶数较小的，在阶数一样的情况下再选择符号个数较少的，
                                # 这样可以保证在序列项数较少时，不会用a_n_6直接拟合6项，从而失去公式的意义

                                # 找出阶数最小的公式
                                if return_recurrece_deep(temp_formula) < return_recurrece_deep(
                                        formual_lis[1][min_degree_idx]):
                                    min_degree_idx = i
                                elif return_recurrece_deep(temp_formula) == return_recurrece_deep(
                                        formual_lis[1][min_degree_idx]):
                                    if formula_symbolic_nums(temp_formula) < formula_symbolic_nums(
                                            formual_lis[1][min_degree_idx]):
                                        min_degree_idx = i

                                # 找出操作符数量最少的公式
                                if formula_symbolic_nums(temp_formula) < formula_symbolic_nums(
                                        formual_lis[1][min_len_idx]):
                                    min_len_idx = i
                                elif formula_symbolic_nums(temp_formula) == formula_symbolic_nums(
                                        formual_lis[1][min_len_idx]):
                                    if return_recurrece_deep(temp_formula) < return_recurrece_deep(
                                            formual_lis[1][min_len_idx]):
                                        min_len_idx = i

                            min_degree_idx_lis = [min_degree_idx]
                            min_len_idx_lis = [min_len_idx]
                            # 如果最小递推度下标和最少操作符公式不仅仅存在一个，则分别至多保存三个，最多共六个
                            for i, temp_formula in enumerate(formual_lis[1]):
                                min_degree_formula = formual_lis[1][min_degree_idx]
                                min_degree_num = return_recurrece_deep(min_degree_formula)  # 最小递推度的值
                                min_degree_ops_num = formula_symbolic_nums(min_degree_formula)  # 最小递推度的符号数量

                                min_len_formula = formual_lis[1][min_len_idx]
                                min_len_num = formula_symbolic_nums(min_len_formula)  # 最少操作符数量
                                min_len_degree_num = return_recurrece_deep(min_len_formula)  # 最少操作符公式的递推度值

                                if i != min_degree_idx and len(min_degree_idx_lis) < 3:
                                    if min_degree_num == return_recurrece_deep(
                                            temp_formula) and min_degree_ops_num == formula_symbolic_nums(temp_formula):
                                        min_degree_idx_lis.append(i)
                                if i != min_len_idx and len(min_len_idx_lis) < 3:
                                    if min_len_num == formula_symbolic_nums(
                                            temp_formula) and min_len_degree_num == return_recurrece_deep(temp_formula):
                                        min_len_idx_lis.append(i)

                                if len(min_degree_idx_lis) >= 3 and len(min_len_idx_lis) >= 3:
                                    break

                            sum_idx_lis = min_len_idx_lis
                            for idx in min_degree_idx_lis:
                                if idx not in sum_idx_lis:
                                    sum_idx_lis.append(idx)

                            for idx in sum_idx_lis:
                                zhongzhui_formula, res = env.trans_qianzhui_formula(formual_lis[1][idx].split())
                                zhongzhui_formula = str(zhongzhui_formula)

                                if res == len(formual_lis[1][idx].split()):
                                    formula_INT = trans_formula_INT(env, formual_lis[1][idx])
                                    writer.writerow([seqname, formula_INT, zhongzhui_formula, formual_lis[0]])

                    # print("总共发现：", len(dict_final))


def test_demo22():
    env, file_name = get_env()
    src = '+ 2 + 3 + 22'.split()
    hyp = 'sub mul 16 x_0_2 10'.split()
    # lis = ['+ 2 + 3 + 5 + 7 + 11 + 29 + 127 + 1931 + 30 9121 + 477 7789 6349 + 7609 912 6066 21 4447 + 120 6213 9544 3859 8216 2081 7698 2342 2453 4627 + 63 8136 8876 6771 9602 3561 3705 4941 5134 3867 4258 9661 637 7223 9941 7500 4925 4375 9703', 'sub mul x_0_1 add 5 idiv 0', 'sub mul x_0_1 add 5 sign INT+', 'n mul x_0_1 idiv 5 sub sign INT+ x_0_1', 'n mul x_0_1 idiv 5 idiv 7', 'n mul x_0_1 add 5 idiv 0', 'n mul x_0_1 idiv 5 idiv 0', 'sub mul add 7 n x_0_1 x_0_1 x_0_6', 'sub mul x_0_1 add 7 3 INT+ idiv 0', 'n mul x_0_1 idiv 5 sub sign INT+ x_0_6', 'n mul idiv 5 x_0_1 sub sign INT+ x_0_1', 'sub mul x_0_1 add 5 3 INT+ idiv 0', 'n mul x_0_1 idiv 5 idiv 9', 'sub mul add 5 n x_0_1 x_0_1 idiv 0', 'n mul add 5 x_0_1 sub sign INT+ x_0_1', 'sub mul x_0_1 add 7 n sign INT+ x_0_1', 'sub mul x_0_1 add 5 idiv 9', 'sub mul add 7 n x_0_1 x_0_1 idiv 0', 'n mul x_0_1 idiv 5 sign INT+', 'sub mul add 5 n x_0_1 x_0_1 x_0_6', 'n mul add 7 n x_0_1 x_0_1 sign INT+', 'n mul add 5 x_0_1 sub sign INT+ x_0_6', 'n mul x_0_1 idiv 5 idiv sqr', 'n mul x_0_1 add 5 idiv 7', 'sub mul x_0_1 add 7 idiv 5', 'n mul x_0_1 add 5 idiv 9', 'sub mul x_0_1 idiv 5 sign INT+', 'sub mul x_0_1 add 5 idiv 6', 'sub mul x_0_1 add 5 idiv x_0_4', 'sub mul x_0_1 add 5 add 0', 'sub mul x_0_1 add 7 idiv 0', 'sub mul add 7 n x_0_1 x_0_1 x_0_1']

    # src = lis[0].split()
    # for i in range(1, len(lis) - 1):
    #     print(i)
    #     hyp = lis[i].split()
    list_seq, formula = env.pre_next_term(src, hyp, n_predict=20)
    res = formula
    print(list_seq)
    print(formula)
    print(type(formula))

    # for i,term in enumerate(list_seq):
    #     print(i,term)
    # print(list_seq)
    # res, seq_name, len_seq = check_seq_in_oeis(root, list_seq)
    # if res == True:
    #     print(seq_name)
    #     print(len_seq)
    #     print(list_seq)
    #     print(formula)
    #     print()


def check_new_formula(path, path2):
    with open(path + 'train.csv', 'r', encoding='utf-8') as f1:
        with open(path2 + 'train.csv', 'r', encoding='utf-8') as f2:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)

            dic1 = {}
            dic2 = {}
            for row in reader1:
                dic1[row[0].replace("\ufeff", "")] = 1
            for row in reader2:
                dic2[row[0].replace("\ufeff", "")] = 1
            train_nums = len(dic2)
            new_find_formula_nums = len(dic2) - len(dic1)

    return train_nums, new_find_formula_nums


def check_isfind_complex_formula(path, path_interesting_oeisSeq):
    with open(path_interesting_oeisSeq, 'r', encoding='utf-8') as f:
        with open(path + 'final_res.json', 'r', encoding='utf-8') as f2:
            # with open("")
            lines = f.readlines()
            dict_formula = json.load(f2)

            str_instrest_seq = ''
            count = 0
            for seq_name in lines:
                seq_name = seq_name.strip()
                if seq_name in dict_formula:
                    print(seq_name)
                    str_instrest_seq += (seq_name + ' ')
                    count += 1
    return str_instrest_seq, count


def obtain_new_train_set(path2, args):
    with open(path2 + "train.csv", 'r', encoding='utf-8') as f1:
        reader1 = csv.reader(f1)

        train_data_list = []
        for row in reader1:
            seq_lis = row[3].split(", ")
            len_seq_lis = len(seq_lis)

            if len(seq_lis) >= 13:
                if len(seq_lis) < 23:
                    seq_input = process_oeis_input_seq(seq_lis[:12], 25)
                if len(seq_lis) >= 23:
                    seq_input = process_oeis_input_seq(seq_lis[:len_seq_lis - 10], 25)
                formula = row[1]
                train_data_list.append([seq_input, formula])

                row[0] = row[0].replace("\ufeff", "")
                row[0] = row[0].replace('﻿A', 'A')

        random.shuffle(train_data_list)  # 随机打乱训练集顺序，分割训练集验证集，利用验证选出最佳检查点，用来解码
    if args.is_combine:
        train_one_data_size=int(len(train_data_list) * args.random_rate_model1)
        train_two_data_size=int(len(train_data_list) * args.random_rate_model2)
        if train_one_data_size<train_two_data_size:
            temp=train_one_data_size
            train_one_data_size=train_two_data_size
            train_two_data_size=temp

    with open(path2 + "train.src", 'w', encoding='utf-8') as f3:
        with open(path2 + "train.tgt", 'w', encoding='utf-8') as f4:
            with open(path2 + "valid.src", 'w', encoding='utf-8') as f5:
                with open(path2 + "valid.tgt", 'w', encoding='utf-8') as f6:
                    valid_len = 2000

                    for i, line in enumerate(train_data_list):
                        if i < train_one_data_size:
                            f3.write(line[0].strip() + '\n')
                            f4.write(line[1].strip() + '\n')
                        if i < valid_len:
                            f5.write(line[0].strip() + '\n')
                            f6.write(line[1].strip() + '\n')

    random.shuffle(train_data_list)  # 随机打乱训练集顺序，分割训练集验证集，利用验证选出最佳检查点，用来解码                  
    if args.is_combine:
        with open(path2 + "train2.src", 'w', encoding='utf-8') as f3:
            with open(path2 + "train2.tgt", 'w', encoding='utf-8') as f4:
                with open(path2 + "valid2.src", 'w', encoding='utf-8') as f5:
                    with open(path2 + "valid2.tgt", 'w', encoding='utf-8') as f6:
                        valid_len = 2000

                        chazhi_data_size=train_one_data_size-train_two_data_size
                        train_two_data=train_data_list[:train_two_data_size]+train_data_list[:chazhi_data_size]
                        for i, line in enumerate(train_two_data):
                            
                            f3.write(line[0].strip() + '\n')
                            f4.write(line[1].strip() + '\n')
                            if i < valid_len:
                                f5.write(line[0].strip() + '\n')
                                f6.write(line[1].strip() + '\n')


def trans_error_formula(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f1:
        with open(path2, 'w', encoding='utf-8') as f2:
            reader = csv.reader(f1)
            # if "zuijia" in path1:
            next(reader)
            writer = csv.writer(f2)

            for row in reader:
                zhongzhui_formula, res = env.trans_qianzhui_formula(row[1].split())
                if res == len(row[1].split()):
                    row[2] = zhongzhui_formula
                    writer.writerow(row)


def check_oeis_no_formula(path):
    with open(path + 'train.csv', 'r', encoding='utf-8') as f1:
        with open('/amax/users/skm21/fairseq-0.12.0/data_oeis/waiting_find_seq.csv', 'r', encoding='utf-8') as f2:
            with open(path + 'oeis_no_formula.csv', 'w', encoding='utf-8-sig') as f3:
                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                header = next(reader2)

                writer = csv.writer(f3)
                writer.writerow(
                    ['seq_name', 'description', 'formula', 'formula_zhongzhui', 'seq', "formula_len", 'seq_len'])
                wait_find_seq_dict = {}
                for row in reader2:
                    wait_find_seq_dict[row[0]] = [row[1], row[3]]

                for row in reader1:
                    if row[0] in wait_find_seq_dict:
                        if len(row[3].split(', ')) > 10:
                            row2 = []
                            row2.append(row[0])
                            row2.append(wait_find_seq_dict[row[0]][0])
                            row2.append(row[1])
                            row2.append(row[2])
                            row2.append(row[3])

                            row2.append(len(row[1].split(' ')))
                            row2.append(len(row[3].split(', ')))
                            writer.writerow(row2)


def change_init_term(path, path2):
    """
    把初始项的格式改为训练使得格式 如： 1,2,3,4,56567 => + 1 + 2 + 3 + 4 + 5 5567
    """
    with open(path, 'r', encoding='utf-8') as f1:
        with open(path2, 'w', encoding='utf-8-sig', newline='') as f2:
            reader = csv.reader(f1)
            writer = csv.writer(f2)
            for row in reader:
                init_6_term = ','.join(row[0].split(',')[:6])
                # print(init_6_term)
                # exit()
                if init_6_term in dict_init:
                    row[0] = process_oeis_input_seq(row[0].split(','), 13)
                    writer.writerow(row)
                    # print(row)
                    # exit()


def check_get_autoGenerate_res2(path_save, check_path):  # 检测自动生成的公式数据有多少能够生成oeis束搜索结果有多少能够生成OEIS数列
    pool = Pool(40)  # 定义一个进程池，最大进程数48
    print("----start----")
    time2 = datetime.now()

    with open(check_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        list_len = [1]
        for j in range(1):
            for i, row in enumerate(tqdm(reader)):
                args = (row, path_save)
                pool.apply_async(more_jincheng_worker, (args,))
        pool.close()
        # 等待po中所有子进程执行完成，必须放在close语句之后
        pool.join()
        print("-----end-----")
        print("检测seq时间：", datetime.now() - time2)


def save_train_to_dict(train_path, dict_path):
    with open(train_path, 'r', encoding='utf-8') as f1:
        with open(dict_path, 'w', encoding='utf-8') as f2:
            reader = csv.reader(f1)
            dict_final = {}
            for row in reader:
                key = row[0].replace("\ufeff", '')
                if key in dict_final:
                    if row[1] in dict_final[key][1]:
                        dict_final[key][1][row[1].strip()] += 1
                    else:
                        dict_final[key][1][row[1].strip()] = 1
                else:
                    dict_final[key] = (row[3], defaultdict(int))
                    dict_final[key][1][row[1].strip()] += 1
            json.dump(dict_final, f2)
            print("len_dict", len(dict_final))


if __name__ == '__main__':
    env, file_name = get_env()
    sta_time = datetime.now()

    test_demo22()
    exit()
    path = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/init_train_data/init_train_data/'
    path2 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/init_train_data/init_train_data/'
    # path3 = f"/amax/users/skm21/fairseq-0.12.0_213/checkpoints/merge_data_4000w/sum_test_res/"
    # trans_error_formula(path+"train.csv",path+"train2.csv")
    # trans_error_formula(path+"zuijia_res.csv",path+"zuijia_res2.csv")

    # # #
    # obtain_csv_res(path, path, 240, 1, 2, 3, 4)
    # obtain_csv_res1(path,path,240,1)
    # check_get_final_res(path)
    # process_temp_res2(path, env)
    # process_temp_res(path)
    # # # # #
    # # #
    # check_new_formula(path)
    path_before = "/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/iter30/result/"
    path = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/iter31/result/'
    path2 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_9w/iter1/result/'
    # process_temp_res2(path_before,path,path2,env)
    # path2='/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/iter31/result/'
    # obtain_new_train_set(path2)
    # save_train_to_dict('/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/iter1/result/train.csv','/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/iter0/result/final_res.json')
    # str_instrest_seq, count=check_isfind_complex_formula(path)
    # print("发现有趣数列数量：",count)
    # print("发现有趣序列名称：",str_instrest_seq)

    # check_oeis_no_formula(path)

    path = '/amax/users/skm21/recur-main/data/10M/'
    # path2='/amax/users/skm21/recur-main/data/big_200g_data/auto_sta_trainset/'
    # check_autoGenerate_correct_data('/amax/users/skm21/recur-main/data/train/5M.txt','/amax/users/skm21/recur-main/data/train/5M_correct_formula.csv')
    # check_get_autoGenerate_res(path)
    # process_temp_res2(path,env)
    # obtain_new_train_set(path, path)
    # obtain_new_train_set('/amax/users/skm21/fairseq-0.12.0/checkpoints/model0_32/iter51/result/','/amax/users/skm21/fairseq-0.12.0/checkpoints/model0_32/iter52/result/')
    #
    # dict_init = get_oeis_init_term_dict()
    # check_get_autoGenerate_res('/amax/users/skm21/recur-main/data/10M/')

    # change_init_term('/amax/users/skm21/recur-main/data/10M/trans_src_tgt.csv','/amax/users/skm21/recur-main/data/10M/trans_src_tgt_change.csv')

    # check_get_autoGenerate_res2('/amax/users/skm21/recur-main/data/10M/auto_formula_correct_temp.csv','/amax/users/skm21/recur-main/data/10M/all_init_term_10len_formula.csv')
    formula = 'add add x_0_2 sub x_0_1 add x_0_3 x_0_6 add x_0_6 INT+ 2'
    zhongzhui_formula, res = env.trans_qianzhui_formula(formula.split())
    print(zhongzhui_formula)
