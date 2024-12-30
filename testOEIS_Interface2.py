import json
import os
from recur.utils.get_env import get_env

from check_oeis import process_temp_res, check_new_formula, obtain_new_train_set, check_isfind_complex_formula, \
    check_seq_in_oeis, return_recurrece_deep, process_temp_res2, check_seq_in_oeis2
import pickle
from recur.utils.node_class import Node
from recur.utils.decode import decode
import collections
from recur.utils.split_oesi_seq import process_oeis_input_seq
from recur.utils.iter_need_method import return_oeis_pre_n

root = ''
with open('tree/oeis_tree_save_dict_seqNames3.pkl', 'rb') as f:
    root = pickle.load(f)

with open('data_oeis/oeis_dict.json', 'r', encoding="utf-8-sig") as f2:
    oeis_dict_seq = json.load(f2)

env, file_name = get_env()


def get_zuijia_formula(dict_formula, seq_len):
    zuijia_formula_list = []
    for seqname, formual_lis in dict_formula.items():
        min = 0
        zuijia = ''
        temp_dict = {}
        for i, temp_formula in enumerate(formual_lis):
            # if len(temp_formula.split()) < len(formual_lis[1][min].split()):
            #     min = i
            # elif len(temp_formula.split()) == len(formual_lis[1][min].split()):
            #     if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[1][min]):
            #         min = i
            # 在利用奥卡姆剃刀原理选择最佳公式时，应该首先选择递推阶数较小的，在阶数一样的情况下再选择符号个数较少的，
            # 这样可以保证在序列项数较少时，不会用a_n_6直接拟合6项，从而失去公式的意义
            if seq_len <= 7:
                if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[min]):
                    min = i
                elif return_recurrece_deep(temp_formula) == return_recurrece_deep(formual_lis[min]):
                    if len(temp_formula.split()) < len(formual_lis[min].split()):
                        min = i
            elif seq_len > 7:
                if len(temp_formula.split()) < len(formual_lis[min].split()):
                    min = i
                elif len(temp_formula.split()) == len(formual_lis[min].split()):
                    if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[min]):
                        min = i
        # print(formual_lis[min])
        zhongzhui_formula, res = env.trans_qianzhui_formula(formual_lis[min].split())
        # print(res)
        if res == len(formual_lis[min].split()):
            zuijia = str(zhongzhui_formula)

        temp_dict['seq_name'] = seqname
        temp_dict['seq25'] = return_oeis_pre_n(oeis_dict_seq, seqname, 25)
        temp_dict['zuijia_formula'] = zuijia
        temp_dict['ohther_formula'] = []

        for formula in formual_lis:
            if formula != formual_lis[min]:
                zhongzhui_formula_other, res = env.trans_qianzhui_formula(formula.split())
                temp_dict['ohther_formula'].append(str(zhongzhui_formula_other))

        zuijia_formula_list.append(temp_dict)
    return zuijia_formula_list


def get_zuijia_formula2(formula_list):
    zuijia_formula_list = {}

    min = 0
    zuijia = ''

    for i, formual in enumerate(formula_list):
        if len(formual.split()) < len(formula_list[min].split()):
            min = i
        elif len(formual.split()) == len(formula_list[min].split()):
            if return_recurrece_deep(formual) < return_recurrece_deep(formula_list[min]):
                min = i

        zhongzhui_formula, res = env.trans_qianzhui_formula(formula_list[min].split())

        if res == len(formula_list[min].split()):
            zuijia = str(zhongzhui_formula)

        zuijia_formula_list['zuijia_formula'] = zuijia
        zuijia_formula_list['ohther_formula'] = []

        for formula in formula_list:
            if formula != formula_list[min]:
                zhongzhui_formula_other, res = env.trans_qianzhui_formula(formula.split())
                zuijia_formula_list['ohther_formula'].append(str(zhongzhui_formula_other))

    return zuijia_formula_list


def comp_error_rate(input_lis, pre_lis):
    pre_lis = pre_lis[:len(input_lis)]
    if input_lis == pre_lis:
        return round(0.000, 2)
    error_sum = 0
    for i in range(len(input_lis)):
        if input_lis[i] == 0:
            error_sum += (abs(input_lis[i] - pre_lis[i])) / abs(input_lis[i] + 1)
        else:
            error_sum += (abs(input_lis[i] - pre_lis[i])) / abs(input_lis[i])
    res = round(((error_sum / len(input_lis)) * 100), 2)
    print("res:", res)
    return res


def formula_symbolic_nums(formula):
    """
    得到一个公式中运算符号的数量
    """
    operators_int = {
        'add': 2,
        'sub': 2,
        'mul': 2,
        'idiv': 2,
        'mod': 2,
        'abs': 1,
        'sqr': 1,
        'relu': 1,
        'sign': 1,
        # 'step': 1,
    }
    count = 0
    for sym in formula.strip().split():
        if sym in operators_int:
            count += 1
    return count


def formula2latex(formula):
    formula=str(formula)
    an_dic = {
              "x_0_12": "a_{n-12}",
              "x_0_11": "a_{n-11}",
              "x_0_10": "a_{n-10}",
              "x_0_9": "a_{n-9}",
              "x_0_8": "a_{n-8}",
              "x_0_7": "a_{n-7}",
              "x_0_6": "a_{n-6}",
              "x_0_5": "a_{n-5}",
              "x_0_4": "a_{n-4}",
              "x_0_3": "a_{n-3}",
              "x_0_2": "a_{n-2}",
              "x_0_1": "a_{n-1}",
              "add": "+",
              "sub": "-",
              "mul": "*",
              "idiv": "//",
              "mod": "\\\\%",
              "sqr": "^2",
              "**": "^",
              "- -": "+",
              "+ -": "-",
              }
    for ani in an_dic.keys():
        if ani in formula:
            formula = formula.replace(ani, an_dic[ani])
    return "a_{n}="+formula


def return_oeis_formula(seq_input, beam_size, nbest, pred_n,model_name,gpuid=0):
    src_path = 'result/inferance/input.src'
    path_result = 'result/inferance/result.txt'
    # check_point_path = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter50/model0/checkpoint_best.pt'
    check_point_path=f'save/model/{model_name}.pt'
    
    if model_name=="merge_data_4500w_epoch10":
        vocab_path = 'data-bin/vocab5'
    else:
        vocab_path = 'data-bin/vocab4'

    with open(src_path, 'w', encoding='utf-8') as f:
        print("seq_inpt:", seq_input)
        seq_input = seq_input.replace(" ", '')
        input_list = seq_input.split(",")
        seq_input = process_oeis_input_seq(input_list, 25)
        print("seq_inpt:", seq_input)
        f.write(seq_input)

    os.system(f"CUDA_VISIBLE_DEVICES={gpuid} cat {src_path} | fairseq-interactive {vocab_path} --source-lang src --target-lang tgt  \
    --path  {check_point_path} --beam {beam_size} --nbest {nbest} &> {path_result}")

    with open(path_result, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        list_write = []
        for i, line in enumerate(lines):
            if line[0] == "S":
                line = line.replace("=", '')
                seq_list = line.strip().split()[1:]
                seq = ' '.join(seq_list)
                # list_write.append(seq)
            if line[0] == "H":
                ans_list = line.strip().split()[2:]
                ans = ' '.join(ans_list)
                list_write.append(ans)
        # print("seq:",seq)
        src = seq.split(" ")

        seq_list = decode(seq.split())
        correct_formula = {}
        # correct_nihe_formula=[]
        not_correct_list = []
        error_rate = 0
        final_res = []
        for formula in list_write:
            hyp = formula.split()
            if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                continue
            flag2 = 0
            list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=len(seq_list))
            if list_seq == seq_list:
                # correct_nihe_formula.append(formula)
                flag2 = 1
            list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)
            recheck_num = 35
            flag = 0
            while True:
                if "error" not in list_seq:
                    res, seq_name_list, len_seq_list = check_seq_in_oeis2(root, list_seq)
                    if res == True:
                        deep_formula = return_recurrece_deep(formula)
                        for i, seq_name in enumerate(seq_name_list):
                            len_seq = len_seq_list[i]
                            if deep_formula < len_seq:
                                if len_seq >= 13:
                                    seq_name = seq_name.replace("\ufeff", "")
                                    seq_name = seq_name.replace('﻿A', 'A')
                                    pred_seq_lis = list_seq[:len(seq_list) + pred_n]
                                    error_rate = comp_error_rate(seq_list, pred_seq_lis)
                                    ops_nums = formula_symbolic_nums(formula)
                                    zhongzhui_formula_other, res = env.trans_qianzhui_formula(formula.split())
                                    formula_latex = formula2latex(zhongzhui_formula_other)
                                    temp_tuple = (error_rate, ops_nums, formula_latex, pred_seq_lis, seq_name)
                                    final_res.append(temp_tuple)
                        break
                    elif seq_name_list == "recheck":
                        recheck_num += 50
                        if flag == 1:
                            break
                        if recheck_num > 377:
                            flag = 1
                            recheck_num = 378
                        list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=recheck_num)

                        if 'error' in list_seq:
                            # if flag2==0 and "error" not in eq_hyp and eq_hyp!='a(n) = _0_' and eq_hyp!='a(n) = ':
                            # not_correct_list.append(eq_hyp)
                            break
                    else:
                        print("eq_hyp:",eq_hyp)
                        print("eq_hyp_type:",type(eq_hyp))
                        if flag2 == 0 and "error" not in eq_hyp and eq_hyp != 'a(n) = _0_' and eq_hyp != 'a(n) = ':
                            pred_seq_lis = list_seq[:(len(seq_list) + pred_n)]
                            error_rate = comp_error_rate(seq_list, pred_seq_lis)
                            ops_nums = formula_symbolic_nums(formula)

                            zhongzhui_formula_other, res = env.trans_qianzhui_formula(formula.split())
                            formula_latex = formula2latex(zhongzhui_formula_other)
                            temp_tuple = (error_rate, ops_nums, formula_latex, pred_seq_lis, "--")
                            final_res.append(temp_tuple)
                            # not_correct_list.append((eq_hyp,pred_seq_lis,error_rate))
                        break
                else:
                    # if flag2==0 and "error" not in eq_hyp and eq_hyp!='a(n) = _0_' and eq_hyp!='a(n) = ':
                    #     not_correct_list.append(eq_hyp)
                    print("formula:", formula)
                    break
    final_res.sort()
    print()
    print(final_res)
    return final_res


if __name__ == '__main__':
    # seq_input='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20'
    seq_input = '	0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987'
    # list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=len(seq_list) + pred_n)
    # res=zuijia_formula_dict = return_oeis_formula(seq_input,32,32,10)
    # print(res)

    # for i in range(100):
    #     seq_input=input("请输入seq,用逗号隔开：\n")
    #     zuijia_formula_dict=return_oeis_formula(seq_input)
    #     print(zuijia_formula_dict)
    # zhongzhui_formula_other, res = env.trans_qianzhui_formula('sqr n'.split())
    # print(zhongzhui_formula_other)
    src = "+ 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29".split(" ")
    hyp = 'add sub x_0_3 x_0_6 sub x_0_3 sub x_0_7 x_0_8'.split()
    list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)
    print(list_seq)
