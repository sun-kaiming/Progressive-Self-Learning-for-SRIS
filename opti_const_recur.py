import csv
from tqdm import tqdm
from recur_float.utils.get_env import get_env
# from utils import relativeErr,lossFunc
from scipy.optimize import brent, fmin_ncg, minimize
import numpy as np
import math
from datetime import datetime

env, file_name = get_env()


def relativeErr(y, yHat, info=False, eps=1e-4):
    yHat = np.reshape(yHat, [1, -1])[0]
    y = np.reshape(y, [1, -1])[0]
    if len(y) > 0 and len(y) == len(yHat):
        # print("yHat - y:",yHat - y)
        try:
            err = min((abs(yHat - y) ** 2 / np.linalg.norm(y + eps)), np.array([100.0]))
        except:
            err = np.array([100.0])
        if info:
            for _ in range(5):
                i = np.random.randint(len(y))
                print('yPR,yTrue:{},{}, Err:{}'.format(yHat[i], y[i], err[i]))
    else:
        err = np.array([100.0])

    return np.mean(err)


def recur_opti(terms_lis, formula_qianzhui, init_const):
    def lossFunc2(constants, eq, indexs, terms, eps=1e-5):
        err = 0
        n_predict = 25
        terms = terms[:n_predict]
        # constants2 = [int(n) for n in constants]

        # print('constants:',constants)
        eq = eq.replace('C', '{}').format(*constants)
        # print("eq:", eq)
        if 'x_0_12' in eq:
            src_sta = terms[:12]
        elif 'x_0_11' in eq:
            src_sta = terms[:11]
        elif 'x_0_10' in eq:
            src_sta = terms[:10]
        elif 'x_0_9' in eq:
            src_sta = terms[:9]
        elif 'x_0_8' in eq:
            src_sta = terms[:8]
        elif 'x_0_7' in eq:
            src_sta = terms[:7]
        elif 'x_0_6' in eq:
            src_sta = terms[:6]
        elif 'x_0_5' in eq:
            src_sta = terms[:5]
        elif 'x_0_4' in eq:
            src_sta = terms[:4]
        elif 'x_0_3' in eq:
            src_sta = terms[:3]
        elif 'x_0_2' in eq:
            src_sta = terms[:2]
        elif 'x_0_1' in eq:
            src_sta = terms[:1]
        else:
            src_sta = []
        terms_next = terms[len(src_sta):]
        list_seq, eq_hyp = env.pre_next_term(terms, eq.split(), n_predict=len(terms_next))
        # print("list_seq:",list_seq)
        if "error" not in list_seq:
            pred_seq_lis = list_seq[len(src_sta):]
            for real_term, pred_term in zip(terms_next, pred_seq_lis):
                err += relativeErr(real_term, pred_term)  # (y-yHat)**2
            # print("pred_seq_lis  str: ",''.join(pred_seq_lis))
            if "nan" in ''.join(str(num) for num in pred_seq_lis):
                err = 1000000
        else:
            # print("error!!!!")
            err = 1000000

        err /= len(terms_next)
        err = abs(err)
        # print("err111111111111:",err)
        return abs(err)

    # terms_lis=[1, 6, 24, 82, 261, 804, 2440, 7356, 22113, 66394, 199248, 597822, 1793557, 5380776, 16142448, 48427480, 145282593, 435847950, 1307544040, 3922632330, 11767897221, 35303691916, 105911076024, 317733228372, 953199685441]

    index_lis = [i + 1 for i in range(len(terms_lis))]

    terms_lis = terms_lis[:25]
    index_lis = index_lis[:25]
    # formula_qianzhui='add mul x_0_1 INT+ 3 div add n sqr n INT+ C'
    formula_qianzhui = formula_qianzhui.replace("idiv", 'div')
    formula_qianzhui_lis = formula_qianzhui.split()
    counyt_num = 0
    ans_lis = []
    for i, sym in enumerate(formula_qianzhui_lis):
        if sym.isdigit():
            counyt_num += 1
            formula_qianzhui_lis[i] = "C"
            ans_lis.append(sym)
    if counyt_num == 0:
        return [0], [0]
    formula_template = ' '.join(formula_qianzhui_lis)

    c = np.array([1.] * counyt_num)
    # print("初始常量c：",c)

    # Powell 鲍威尔法
    # COBYLA 线性近似法

    # print("formula_template:",formula_template)
    # print("terms_lis:",terms_lis)
    cHat = minimize(lossFunc2, c, args=(formula_template, index_lis, terms_lis), method='Powell', tol=1e-6

                    )
    # print("cHat.x",cHat.x)
    predicted = formula_template.replace('C', '{}').format(*cHat.x)
    # print('迭代终止是否成功：', cHat.success)
    # print(cHat.x)
    # print(predicted)
    # try:
    return ans_lis, [round(x) for x in cHat.x]
    # except:
    #     return [0], [1]


def get_acc(terms_lis, formula_qianzhui):
    # terms_lis = [1, 3, 11, 41, 153, 571, 2131, 7953, 29681, 110771, 413403, 1542841, 5757961, 21489003, 80198051, 299303201, 1117014753, 4168755811, 15558008491, 58063278153, 216695104121, 808717138331, 3018173449203, 11263976658481]
    # formula_qianzhui = 'sub mul INT+ 4 x_0_1 x_0_2'
    # print("111formula_qianzhui:", formula_qianzhui)
    ans_lis, pred_lis = recur_opti(terms_lis, formula_qianzhui)
    print("formula_qianzhui:", formula_qianzhui)
    print("pred_lis:", pred_lis)
    print("ans_lis:", ans_lis)
    if ' '.join([str(num) for num in pred_lis]) == ' '.join([str(num) for num in ans_lis]):
        return True
    else:
        return False


def main(file_path):
    sta_time = datetime.now()
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        correct_count = 0
        all_count = 0
        max_count = 5000
        dic = {}
        for row in tqdm(reader):
            all_count += 1
            if all_count > max_count:
                break
            if 'idiv' not in row[1] and 'mod' not in row[1]:
                dic[row[0]] = row

        for seq_name, row in tqdm(dic.items()):
            terms_lis = [int(num) for num in row[3].split(', ')]
            # print(formula_qianzhui)
            # print(terms_lis)
            if get_acc(terms_lis, row[1]):
                # print("预测正确:",row[1])
                correct_count += 1
        print("acc:", correct_count / len(dic))
    print("用时：", datetime.now() - sta_time)


if __name__ == '__main__':
    # main("/home/skm21/fairseq-0.12.0/data_oeis/train (1).csv")
    terms_lis = [1, 6, 12, 19, 27, 36, 46, 57, 69, 82, 96, 111, 127, 144, 162, 181, 201, 222, 244, 267, 291, 316, 342,
                 369, 397, 426, 456, 487, 519, 552, 586, 621, 657, 694, 732, 771, 811, 852, 894, 937, 981, 1026, 1072,
                 1119, 1167, 1216, 1266, 1317, 1369, 1422, 1476]
    formula = 'idiv mod INT- 5 mul n add INT+ 7 n INT+ 2'
    # res=get_acc(terms_lis,formula)
    # print(res)
    # formula=
    init_const = []
    const_index_lis = []
    formula_lis = formula.split()
    for i, sym in enumerate(formula.split()):
        if sym.isdigit():
            init_const.append(sym)
            const_index_lis.append(i)
    # init_const=np.array([int(num) for num in init_const])
    init_const = np.array([1.0] * len(const_index_lis))
    print(init_const)
    _, pred_xishu = recur_opti(terms_lis, formula, init_const)
    print(pred_xishu)
    for i, const in enumerate(pred_xishu):
        formula_lis[const_index_lis[i]] = const
    print(formula_lis)
