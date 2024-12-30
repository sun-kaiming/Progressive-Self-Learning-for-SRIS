import json


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
        # print(sym)
        if sym in operators_int:
            # print(sym)
            count += 1

    return count


# 分别找出具有最小递推度和最少符号个数的公式下标min_degree、min_len
def find_easy_formula(formual_lis):
    min_degree_idx = 0
    min_len_idx = 0
    for i, temp_formula in enumerate(formual_lis):
        # if len(temp_formula.split()) < len(formual_lis[1][min].split()):
        #     min = i
        # elif len(temp_formula.split()) == len(formual_lis[1][min].split()):
        #     if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[1][min]):
        #         min = i
        # 在利用奥卡姆剃刀原理选择最佳公式时，应该首先选择递推阶数较小的，在阶数一样的情况下再选择符号个数较少的，
        # 这样可以保证在序列项数较少时，不会用a_n_6直接拟合6项，从而失去公式的意义

        # 找出阶数最小的公式
        if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[min_degree_idx]):
            min_degree_idx = i
        elif return_recurrece_deep(temp_formula) == return_recurrece_deep(formual_lis[min_degree_idx]):
            if formula_symbolic_nums(temp_formula) < formula_symbolic_nums(formual_lis[min_degree_idx]):
                min_degree_idx = i

        # 找出操作符数量最少的公式
        if formula_symbolic_nums(temp_formula) < formula_symbolic_nums(formual_lis[min_len_idx]):
            min_len_idx = i
        elif formula_symbolic_nums(temp_formula) == formula_symbolic_nums(formual_lis[min_len_idx]):
            if return_recurrece_deep(temp_formula) < return_recurrece_deep(formual_lis[min_len_idx]):
                min_len_idx = i

    min_degree_idx_lis = [min_degree_idx]
    min_len_idx_lis = [min_len_idx]
    # 如果最小递推度下标和最少操作符公式不仅仅存在一个，则分别至多保存三个，最多共六个
    for i, temp_formula in enumerate(formual_lis):
        min_degree_formula = formual_lis[min_degree_idx]
        min_degree_num = return_recurrece_deep(min_degree_formula)  # 最小递推度的值
        min_degree_ops_num = formula_symbolic_nums(min_degree_formula)  # 最小递推度的符号数量
        # print("min_degree_num:",min_degree_num)
        # print("min_degree_ops_num:",min_degree_ops_num)
        min_len_formula = formual_lis[min_len_idx]
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
    # print("min_len_idx_lis:",min_len_idx_lis)
    # print("min_degree_idx_lis:",min_degree_idx_lis)
    for idx in min_degree_idx_lis:
        if idx not in sum_idx_lis:
            sum_idx_lis.append(idx)
    formual_lis_simply = []

    for id in sum_idx_lis:
        formual_lis_simply.append(formual_lis[id])
    return formual_lis_simply


find_seq_name = 'A030978'
for i in range(1, 50):
    path = f'/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter{i}/result/final_res.json'
    try:
        with open(path, 'r') as f:
            dic = json.load(f)
            if find_seq_name in dic:
                lis_formulas = []
                for formula, nums in dic[find_seq_name][1].items():
                    lis_formulas.append(formula)
                formual_lis_simply = find_easy_formula(lis_formulas)
                print(f"iter{i}", len(dic[find_seq_name][1]), formual_lis_simply)
    except:
        pass
