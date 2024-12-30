import os
from recur.utils.get_env import get_env

from check_oeis import process_temp_res, check_new_formula, obtain_new_train_set, check_isfind_complex_formula, \
    check_seq_in_oeis, return_recurrece_deep, process_temp_res2
import pickle
from recur.utils.node_class import Node
from recur.utils.decode import decode
import collections
from recur.utils.split_oesi_seq import process_oeis_input_seq

root = ''
with open('/home/skm21/fairseq-0.12.0/recur/utils/oeis_tree_save_dict.pkl', 'rb') as f:
    root = pickle.load(f)


def get_zuijia_formula(dict_formula, seq_len):
    zuijia_formula_dict = {}

    for seqname, formual_lis in dict_formula.items():
        # print(seqname, formual_lis)
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
        temp_dict['zuijia_formula'] = zuijia
        temp_dict['ohther_formula'] = []

        for formula in formual_lis:
            if formula != formual_lis[min]:
                zhongzhui_formula_other, res = env.trans_qianzhui_formula(formula.split())

                temp_dict['ohther_formula'].append(str(zhongzhui_formula_other))
        zuijia_formula_dict[seqname] = temp_dict

    return zuijia_formula_dict


def return_oeis_formula(seq_input):
    src_path = '/home/skm21/fairseq-0.12.0/test_oeis_interface/input.src'
    path_result = '/home/skm21/fairseq-0.12.0/test_oeis_interface/result.txt'

    with open(src_path, 'w', encoding='utf-8') as f:
        seq_input = seq_input.replace(" ", '')
        seq_input = process_oeis_input_seq(seq_input.split(","), 25)
        f.write(seq_input)

    os.system(f"cat {src_path} | fairseq-interactive /home/skm21/fairseq-0.12.0/data-bin/model0_32_newdata/iter3.oeis --source-lang src --target-lang tgt  \
    --path checkpoints/model0_32_newdata2/iter100/checkpoint_best.pt  --beam 32 --nbest 32 &> {path_result}")

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
        src = seq.split(" ")

        seq_list = decode(seq.split())
        seq_str = ','.join([str(num) for num in seq_list])
        correct_formula = collections.defaultdict(list)
        for formula in list_write:
            hyp = formula.split()
            if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                continue

            list_seq, eq_hyp = env.pre_next_term(src, hyp, n_predict=35)
            recheck_num = 35
            flag = 0
            while True:
                if "error" not in list_seq:
                    res, seq_name, len_seq = check_seq_in_oeis(root, list_seq)
                    if res == True:
                        deep_formula = return_recurrece_deep(formula)
                        if deep_formula < len_seq:
                            if len_seq >= 7:
                                seq_name = seq_name.replace("\ufeff", "")
                                seq_name = seq_name.replace('﻿A', 'A')

                                list_seq_str = ','.join([str(num) for num in list_seq])
                                if seq_str in list_seq_str:
                                    correct_formula[seq_name].append(formula)
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
    zuijia_formula_dict = get_zuijia_formula(correct_formula, len_seq)

    return zuijia_formula_dict


if __name__ == '__main__':
    seq_input = '	1, 4, 8, 20, 36, 84, 148, 340, 596, 1364, 2388, 5460, 9556, 21844, 38228, 87380, 152916, 349524, 611668, 1398100, 2446676, 5592404, 9786708, 22369620, 39146836, 89478484, 156587348, 357913940'
    env, file_name = get_env()
    zuijia_formula_dict = return_oeis_formula(seq_input)
    print(zuijia_formula_dict)
