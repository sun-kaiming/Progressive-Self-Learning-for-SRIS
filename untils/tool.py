import csv
import json
from collections import defaultdict
from recur.utils.iter_need_method import obtain_csv_res, obtain_csv_res1, formula_symbolic_nums, return_recurrece_deep

from tqdm import tqdm
from recur.utils.split_oesi_seq import process_big_num, str_insert
from recur.utils.get_env import get_env
import os,random


def find_different(res1, res2, save_res):
    """
    寻找出我们美哦与找到公式的序列，而findLinearRecurrence找到的序列，将这些另存到save_res里面
    res1:我们的最优结果
    res2:而findLinearRecurrence找到的序列结果
    """
    with open(res1, 'r', encoding='utf-8') as f1:
        with open(res2, 'r', encoding='utf-8') as f2:
            with open(save_res, 'w', encoding='utf-8-sig', newline='') as f3:
                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                writer = csv.writer(f3)
                dict_seq = {}
                for i, row in enumerate(reader1):
                    dict_seq[row[0].replace("\ufeff", "")] = row[1:]
                # print(dict_seq)
                for i, row in enumerate(reader2):
                    # if len(row[1][1:].split(',')) >= 35:
                    seq_name = row[0].replace("\ufeff", "")
                    if seq_name not in dict_seq:
                        print(row)
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


def delete_chongfu_seq(path, save_path):
    """
    删除初始训练集中的重复数列公式
    """
    with open(path, 'r', encoding='utf-8') as f1:
        with open(save_path, 'w', encoding='utf-8-sig', newline='') as f2:
            reader = csv.reader(f1)
            writer = csv.writer(f2)
            dic = {}
            dic_seqName = {}
            for row in reader:
                if ', ' in row[3]:
                    seq_len = len(row[3].split(', '))
                else:
                    seq_len = len(row[3].split(','))
                if seq_len >= 13:
                    seqname_formula = row[0] + row[1]
                    dic[seqname_formula] = row
                    dic_seqName[row[0]] = 1
            print(len(dic))
            print(len(dic_seqName))
            for k, v in dic.items():
                writer.writerow(v)


def comp_mathmatical_pre1_10_acc(path):
    """计算数据函数预测后1项，10项的acc"""
    with open(path, 'r', encoding='utf-8') as f1:
        pre1_acc = 0
        pre10_acc = 0
        pre_all_acc = 0
        reader = csv.reader(f1)

        for i, row in enumerate(reader):
            seq_name = row[0]
            if len(row[2]) < 10000 and row[2] != ',' and row[2] != '' and 'DifferenceRoot[Function' not in row[3]:
                real10_lis = row[1][1:].split(',')[25:35]
                pre10_lis = row[2][1:].split(',')[25:35]
                # print(pre10_lis)
                # print(real10_lis)
                # print()

                print(i)
                # print(row)
                print(pre10_lis[0])
                print(real10_lis[0])
                if pre10_lis[0] == real10_lis[0]:
                    pre1_acc += 1
                if pre10_lis == real10_lis:
                    # print(pre10_lis)
                    # print(real10_lis)
                    # print()
                    pre10_acc += 1
                if row[1] == row[2]:
                    pre_all_acc += 1

        print("pre1_acc:", pre1_acc / 10000)
        print("pre10_acc:", pre10_acc / 10000)
        print("pre_all_acc:", pre_all_acc / 10000)


def merge_all_finalRes(path, save_path):
    """
        合并所有生成的final_res.josn文件，选出其中每个序列最优的6个公式加入训练集；
        其中部分公式常数没有用”INT+ 1 2 3 “的形式表示，将该部分公式转化成这种形式
    """
    with open(save_path, 'r', encoding='utf-8') as f1:
        dict_final_sum = json.load(f1)

    with open(path, 'r', encoding='utf-8') as f2:
        dict_final = json.load(f2)

        for seq_name, seq_lis in dict_final.items():
            for formula in seq_lis[1]:
                formula = trans_formula_INT(formula)
                if seq_name in dict_final_sum:
                    if formula in dict_final_sum[seq_name][1]:
                        dict_final_sum[seq_name][1][formula] += 1
                    else:
                        dict_final_sum[seq_name][1][formula] = 1
                else:
                    # print(seq_name)
                    # print(seq_lis)
                    dict_final_sum[seq_name] = (seq_lis[0], defaultdict(int))
                    dict_final_sum[seq_name][1][formula] += 1

    with open(save_path, 'w', encoding='utf-8') as f3:
        json.dump(dict_final_sum, f3)


def iter_update_final_res_sum():
    """
    合并所有iter的final_res结果到一个文件中
    """
    save_path = '/home/skm21/fairseq-0.12.0/data_oeis/final_res_sum/final_res.json'
    for i in tqdm(range(84, 101)):
        # path = f'/home/skm21/fairseq-0.12.0/checkpoints/model0_32_newdata/iter{i}/result/'
        path = f'/home/skm21/fairseq-0.12.0/checkpoints/model0_32_auto_208/iter{i}/result/final_res.json'
        merge_all_finalRes(path, save_path)


def json_file_size(path, path_test):
    """
    返回一个json文件的大小，其中的序列长度要在大于等于13
    """
    with open(path, 'r', encoding='utf-8') as f:
        dic = json.load(f)
        count = 0

        dic_seqName_easy = get_test_seqNmae_dic(path_test + '1wan_easy_testdata_35.csv')
        dic_seqName_sign = get_test_seqNmae_dic(path_test + '1wan_sign_testdata_35.csv')
        dic_seqName_base = get_test_seqNmae_dic(path_test + '1wan_base_testdata_35.csv')

        easy_count_acc = 0
        sign_count_acc = 0
        base_count_acc = 0
        for seq_name, seq_lis in dic.items():
            if len(seq_lis[0].split(', ')) >= 13:
                count += 1
                if seq_name in dic_seqName_easy:
                    easy_count_acc += 1
                if seq_name in dic_seqName_sign:
                    sign_count_acc += 1
                if seq_name in dic_seqName_base:
                    base_count_acc += 1
        print(count)
        print("easy_acc:", easy_count_acc)
        print("sign_acc:", sign_count_acc)
        print("base_acc:", base_count_acc)


def get_test_seqNmae_dic(path):
    """得到测试集序列名字典"""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        dic = {}
        for row in reader:
            dic[row[0]] = 1
        return dic


def trans_formula_INT(formula):
    """
    将公式中的不是用INT形式表示的常数用INT形式表示
    """

    formula_lis = formula.split()
    formula_tree, res = env.trans_qianzhui_formula(formula_lis)
    # print(formula)
    # print(formula_tree)

    formula_new_lis = env.zhongzhui_trans_qianzhui(formula_tree)
    formula_new_str = ' '.join(formula_new_lis)
    # print(formula_new_str)
    return formula_new_str


def trans_first_final_res(path, save_path):
    """
    转化第一个final_res.json中的公式中常数为INT形式
    """
    with open(path, 'r', encoding='utf-8') as f1:
        with open(save_path, 'w', encoding='utf-8') as f2:
            final_res = json.load(f1)
            final_res_new = {}
            for seq_name, lis in tqdm(final_res.items()):
                formula_dic_new = {}
                formula_dic = lis[1]
                for formula, nums in formula_dic.items():
                    # print(formula)
                    # print(nums)
                    # exit()
                    formula_INT = trans_formula_INT(formula)
                    formula_dic_new[formula_INT] = nums
                    if formula_INT != formula:
                        print(formula)
                        print(formula_INT)
                        print()
                lis[1] = formula_dic_new
                final_res_new[seq_name] = lis

            json.dump(final_res_new, f2)


def simply_final_res(path, save_path):
    """
    化简final_res.json文件，现在太大了 有500多mb，每个序列只保留最优的几个公式
    """
    with open(path, 'r', encoding='utf-8') as f1:
        with open(save_path, 'w', encoding='utf-8') as f2:
            final_res = json.load(f1)
            final_res_new = {}

            for seq_name, seq_lis in final_res.items():
                formula_dic = seq_lis[1]
                formula_lis = []
                formula_dic_new = {}
                for formula, nums in formula_dic.items():
                    formula_lis.append(formula)
                formula_lis_simply = get_zuijia_formula2(formula_lis)
                for formula in formula_lis_simply:
                    nums = formula_dic[formula]
                    formula_dic_new[formula] = nums
                seq_lis[1] = formula_dic_new
                final_res_new[seq_name] = seq_lis

            json.dump(final_res_new, f2)


def get_zuijia_formula2(formula_list):
    """
    发现候选公式列表中最优的几个公式
    """
    zuijia_formula_list = {}
    min_degree_idx = 0
    min_len_idx = 0

    # 分别找出具有最小递推度和最少符号个数的公式下标min_degree、min_len
    for i, temp_formula in enumerate(formula_list):

        # 找出阶数最小的公式
        if return_recurrece_deep(temp_formula) < return_recurrece_deep(formula_list[min_degree_idx]):
            min_degree_idx = i
        elif return_recurrece_deep(temp_formula) == return_recurrece_deep(formula_list[min_degree_idx]):
            if formula_symbolic_nums(temp_formula) < formula_symbolic_nums(formula_list[min_degree_idx]):
                min_degree_idx = i

        # 找出操作符数量最少的公式
        if formula_symbolic_nums(temp_formula) < formula_symbolic_nums(formula_list[min_len_idx]):
            min_len_idx = i
        elif formula_symbolic_nums(temp_formula) == formula_symbolic_nums(formula_list[min_len_idx]):
            if return_recurrece_deep(temp_formula) < return_recurrece_deep(formula_list[min_len_idx]):
                min_len_idx = i

    min_degree_idx_lis = [min_degree_idx]
    min_len_idx_lis = [min_len_idx]
    # 如果最小递推度下标和最少操作符公式不仅仅存在一个，则分别至多保存三个，最多共六个
    for i, temp_formula in enumerate(formula_list):
        min_degree_formula = formula_list[min_degree_idx]
        min_degree_num = return_recurrece_deep(min_degree_formula)  # 最小递推度的值
        min_degree_ops_num = formula_symbolic_nums(min_degree_formula)  # 最小递推度的符号数量

        min_len_formula = formula_list[min_len_idx]
        min_len_num = formula_symbolic_nums(min_len_formula)  # 最少操作符数量
        min_len_degree_num = return_recurrece_deep(min_len_formula)  # 最少操作符公式的递推度值

        if i != min_degree_idx and len(min_degree_idx_lis) < 3:
            if min_degree_num == return_recurrece_deep(temp_formula) and min_degree_ops_num == formula_symbolic_nums(
                    temp_formula):
                min_degree_idx_lis.append(i)
        if i != min_len_idx and len(min_len_idx_lis) < 3:
            if min_len_num == formula_symbolic_nums(temp_formula) and min_len_degree_num == return_recurrece_deep(
                    temp_formula):
                min_len_idx_lis.append(i)

        if len(min_degree_idx_lis) >= 3 and len(min_len_idx_lis) >= 3:
            break

    sum_idx_lis = min_len_idx_lis
    for idx in min_degree_idx_lis:
        if idx not in sum_idx_lis:
            sum_idx_lis.append(idx)

    return_formula_lis = []
    for idx in sum_idx_lis:
        formula = formula_list[idx]
        return_formula_lis.append(formula)

    return return_formula_lis


def save_res22(path, iter_num):
    with open(path + f"iter{iter_num}/result/key_message_log.txt", 'r', encoding='utf-8') as f:
        with open(path + 'iter_res.csv', 'a', encoding='utf-8') as f2:
            lines = f.readlines()

            row = []
            # print(lines[5])
            row.append(f"iter{iter_num}")
            row.append(lines[5][10:].strip())
            row.append(lines[6][10:].strip())

            easy_acc = float(lines[9][24:].strip()) * 100

            row.append(easy_acc)

            row.append(lines[12][18:].strip())

            writer = csv.writer(f2)

            writer.writerow(row)


def test_result():
    ####################保存每一轮迭代结果到csv文件中，方便数据的记录######################################################
    path = "/home/skm21/fairseq-0.12.0/checkpoints/model0_128_auto/"
    for i in range(1, 7):
        save_res22(path, i)


def process_oeis_input_seq(oeis_seq, max_len):  # 处理oeis整数序列： 例如 "1,-1,1,-2,2,-4567893"=> "+ 1 - 1 1 - 2 2 - 456 7893"
    # oeis_seq='A000087 ,2,1,2,4,10,37,138,628,2972,14903,76994,409594,-2222628,12281570,-68864086,391120036,2246122574,13025720000,76101450042,449105860008,2666126033850,15925105028685,95664343622234,577651490729530,'

    if len(oeis_seq) > max_len:
        oeis_seq = oeis_seq[:max_len]

    # oeis_seq.reverse() #逆序seq

    oeis_seq2 = []
    for num in oeis_seq:
        if len(num) > 4:
            num = process_big_num(num)

        if '-' not in num:
            oeis_seq2.append('+')
            oeis_seq2.extend(num.split())
        else:
            oeis_seq2.append('-')
            oeis_seq2.extend(num[1:].split())
    # print(oeis_seq)
    # print(type(oeis_seq[0]))

    # print(oeis_seq2)
    oeis_seq3 = []
    for n in oeis_seq2:
        if len(n) == 4:
            if n == "0000":
                oeis_seq3.append('0')
            elif n[:3] == "000":
                oeis_seq3.append(n[-1])
            elif n[:2] == "00":
                oeis_seq3.append(n[2:])
            elif n[:1] == "0":
                oeis_seq3.append(n[1:])
            else:
                oeis_seq3.append(n)
        else:
            oeis_seq3.append(n)
    # print(oeis_seq3)
    # exit()
    return ' '.join(oeis_seq3)


def split_test_data(path, num, split_seq_path, small_oeis_testset):  # num是把测试集分成几份！
    with open(path, 'r') as f:
        reader = csv.reader(f)
        lines = []
        max_decoder_seqnums = 400000  # 解码的最大OEIS数列数量

        if small_oeis_testset=="True":
            max_decoder_seqnums = 1000

        count = 0
        for i, row in enumerate(reader):
            try:
                count += 1
                if count > max_decoder_seqnums:
                    break
                if row[1][0] == ",":
                    lines.append(row[1][1:])
                else:
                    lines.append(row[1])
            except:
                pass

        # with open(path, 'r', encoding='utf-8') as f1:
        #     lines = f1.readlines()
        lines2 = []
        for line in lines:
            # seq=decode(line.split(','))
            # seq=[str(num) for num in seq]
            # print(line)
            seq = line.split(',')
            if len(seq) >= 13 and len(seq) <= 22:
                seq_input = process_oeis_input_seq(seq[:12], 25)
            if len(seq) >= 23:
                seq_input = process_oeis_input_seq(seq[:len(seq) - 10], 25)
            lines2.append(seq_input)

        test_len = len(lines2)
        save_path = split_seq_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(1, num + 1):
            sta_index = int(test_len / num + 1) * (i - 1)
            end_index = int(test_len / num + 1) * i
            if end_index > test_len:
                end_index = test_len
            print(sta_index, end_index)
            dic_path = os.path.dirname(path)
            # save_path=dic_path + f'/easy36w_35_split{num}'

            with open(f'{save_path}/test_{i}.src', 'w', encoding='utf-8') as f2:
                with open(f'{save_path}/test_{i}.tgt', 'w', encoding='utf-8') as f3:
                    for j in range(sta_index, end_index):
                        f2.write(lines2[j] + '\n')
                        f3.write('eos' + '\n')
        # else:
        #     print(f"已经分割过了{save_path}数列")
        return count

def obtain_new_train_set_init(train_path, save_path,args): ## 联合训练，将训练数据按照不同比率随机分为两种不同的训练数据，并将比率较小的数据填充到和比例较大的数据一样多
    if not  os.path.exists(save_path):
        os.makedirs(save_path)

    with open(train_path + "train.csv", 'r', encoding='utf-8') as f1:
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
    train_one_data_size=int(len(train_data_list) * args.random_rate_model1)
    if args.is_combine=="True":
        train_two_data_size=int(len(train_data_list) * args.random_rate_model2)
        if train_one_data_size<train_two_data_size:
            temp=train_one_data_size
            train_one_data_size=train_two_data_size
            train_two_data_size=temp

    with open(save_path + "train.src", 'w', encoding='utf-8') as f3:
        with open(save_path + "train.tgt", 'w', encoding='utf-8') as f4:
            with open(save_path + "valid.src", 'w', encoding='utf-8') as f5:
                with open(save_path + "valid.tgt", 'w', encoding='utf-8') as f6:
                    valid_len = 2000

                    for i, line in enumerate(train_data_list):
                        if i < train_one_data_size:
                            f3.write(line[0].strip() + '\n')
                            f4.write(line[1].strip() + '\n')
                        if i < valid_len:
                            f5.write(line[0].strip() + '\n')
                            f6.write(line[1].strip() + '\n')

    random.shuffle(train_data_list)  # 随机打乱训练集顺序，分割训练集验证集，利用验证选出最佳检查点，用来解码                  
    if args.is_combine=="True":
        with open(save_path + "train2.src", 'w', encoding='utf-8') as f3:
            with open(save_path + "train2.tgt", 'w', encoding='utf-8') as f4:
                with open(save_path + "valid2.src", 'w', encoding='utf-8') as f5:
                    with open(save_path + "valid2.tgt", 'w', encoding='utf-8') as f6:
                        valid_len = 2000

                        chazhi_data_size=train_one_data_size-train_two_data_size
                        train_two_data=train_data_list[:train_two_data_size]+train_data_list[:chazhi_data_size]
                        for i, line in enumerate(train_two_data):
                            
                            f3.write(line[0].strip() + '\n')
                            f4.write(line[1].strip() + '\n')
                            if i < valid_len:
                                f5.write(line[0].strip() + '\n')
                                f6.write(line[1].strip() + '\n')

if __name__ == '__main__':
    env, file_name = get_env()
    res1 = '/home/skm21/fairseq-0.12.0_213/result/sign_analyze/best_res_sign.csv'
    res2 = '/home/skm21/fairseq-0.12.0_213/result/sign_analyze/sign_final_success_buquan.csv'
    save_res = '/home/skm21/fairseq-0.12.0_213/result/sign_analyze/no_find_oeisSeq.csv'
    # find_different(res1,res2,save_res)

    path1 = '/amax/users/skm21/fairseq-0.12.0_149/checkpoints/4500w_SL_all_initData/init_train_data/model0.csv'
    path2 = '/amax/users/skm21/fairseq-0.12.0_149/checkpoints/4500w_SL_all_initData/init_train_data/sum_data6.csv'
    save_path = '/amax/users/skm21/fairseq-0.12.0_149/checkpoints/4500w_SL_all_initData/init_train_data/sum_data7.csv'
    # hebing_res(path1, path2, save_path)

    path = '/amax/users/skm21/fairseq-0.12.0_149/checkpoints/4500w_SL_all_initData/init_train_data/sum_data6.csv'
    save_path = '/amax/users/skm21/fairseq-0.12.0_149/checkpoints/4500w_SL_all_initData/init_train_data/sum_data6_delete_chongfuSeq.csv'
    # delete_chongfu_seq(path,save_path)

    path = '/kercing/skm21/fairseq-0.12.0_211/result/final_res/linearRecurrence/base_final_res.csv'
    # comp_mathmatical_pre1_10_acc(path)

    path = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/iter30/final_res.json'
    save_path = '/home/skm21/fairseq-0.12.0/data_oeis/final_res_sum/final_res_iter30_add_all.json'
    # merge_all_finalRes(path,save_path)

    test_path = '/home/skm21/fairseq-0.12.0/data_oeis/'
    # json_file_size(path, test_path)
    # json_file_size(save_path,test_path)

    path = '/home/skm21/fairseq-0.12.0/data_oeis/final_res_sum/final_res_simply.json'
    # json_file_size(path,test_path)

    formula = 'mul x_0_4 INT+ 1 73'
    # formula_new=trans_formula_INT(formula)

    path = '/home/skm21/fairseq-0.12.0/data_oeis/final_res_sum/final_res.json'
    save_path = '/home/skm21/fairseq-0.12.0/data_oeis/final_res_sum/final_res_new.json'
    # trans_first_final_res(path,save_path)

    # iter_update_final_res_sum()
    path = '/home/skm21/fairseq-0.12.0/data_oeis/final_res_sum/final_res_iter30_add_all.json'
    save_path = '/home/skm21/fairseq-0.12.0/data_oeis/final_res_sum/final_res_simply.json'
    # simply_final_res(path,save_path)

    test_result()
