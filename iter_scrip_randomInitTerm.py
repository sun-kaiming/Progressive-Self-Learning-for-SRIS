import os
import time

from recur.utils.create_path import create_no_path
from recur.utils.iter_need_method import obtain_csv_res, comp_1wan_oeis_acc, readom_select_train_data, obtain_csv_res3, \
    obtain_csv_res2, test_1wan_easy_acc, find_hard_more_formula, return_oeis_all_init_term, return_oeis_random_init_term
from recur.utils.get_env import get_env
from check_oeis import process_temp_res, check_new_formula, obtain_new_train_set, check_isfind_complex_formula, \
    check_seq_in_oeis, return_recurrece_deep, process_temp_res2, return_recurrece_deep
import pickle
from datetime import datetime

env, file_name = get_env()
import csv
from multiprocessing import Pool
from tqdm import tqdm
from recur.utils.node_class import Node

root = ''
with open('/home/skm21/fairseq-0.12.0_213/recur/utils/oeis_tree_save_dict.pkl', 'rb') as f:
    root = pickle.load(f)


def more_jincheng_worker(args):
    row, temp_res_path, init_term_list = args

    src = row[0].replace('=', '')
    src = src.split(" ")

    with open(temp_res_path, 'a', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        for formula in row[1:]:
            hyp = formula.split()
            if len(hyp) > 50:  # 避免过程长的公式导致程序运行不了
                continue
            deep_formula = return_recurrece_deep(formula)
            init_term_random_list = []
            if deep_formula != 0:
                init_term_random_list = return_oeis_random_init_term(init_term_list[deep_formula - 1], 9)
            init_term_random_list.append(src)
            # print(init_term_random_list)
            # exit()
            for init_term in init_term_random_list:
                list_seq, eq_hyp = env.pre_next_term(init_term, hyp, n_predict=35)
                recheck_num = 35
                flag = 0
                while True:
                    if list_seq != 'error':
                        res, seq_name, len_seq = check_seq_in_oeis(root, list_seq)
                        if res == True:

                            if deep_formula < len_seq:
                                if len_seq >= 7:
                                    seq_name = seq_name.replace("\ufeff", "")
                                    seq_name = seq_name.replace('﻿A', 'A')
                                    # print(seq_name, "success")
                                    writer.writerow([seq_name, formula,
                                                     ", ".join(str(i) for i in list_seq[:len_seq])])
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


def check_get_final_res(path):  # 检测束搜索结果有多少在oeis中 ，将结果保存在csv文件里
    pool = Pool(40)  # 定义一个进程池，最大进程数48

    print("----start----")

    time2 = datetime.now()

    temp_res_path = path + 'temp_res.csv'
    beam_res_path = path + 'generate-test.csv'

    init_term_list = []
    for i in range(1, 7):
        init_term_list_temp = return_oeis_all_init_term(i)
        init_term_list.append(init_term_list_temp)

    with open(beam_res_path, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        next(reader)
        for j in range(1):
            for k, row in enumerate(tqdm(reader)):
                # print('检验第' + str(k) + "行的seq")
                args = (row, temp_res_path, init_term_list)
                pool.apply_async(more_jincheng_worker, (args,))
        # 关闭进程池，关闭后po不再接收新的请求
        pool.close()
        # 等待po中所有子进程执行完成，必须放在close语句之后
        pool.join()
        print("-----end-----")
        print("检测seq时间：", datetime.now() - time2)

    return temp_res_path


for iter_num in range(2, 102):
    model_type = 'model0_32_random_init_term'

    path = f"/home/skm21/fairseq-0.12.0_213/checkpoints/{model_type}/iter{iter_num}/result/"
    path2 = f'/home/skm21/fairseq-0.12.0_213/checkpoints/{model_type}/iter{iter_num + 1}/result/'
    path3 = f"/home/skm21/fairseq-0.12.0_213/checkpoints/{model_type}/iter{iter_num}/"  # 为多卡解码做准备
    rate = 0.5  # 随机选择训练集中的多少比例来进行训练
    with open(path + "key_message_log.txt", 'w', encoding='utf-8') as f:
        create_no_path(path)
        create_no_path(path2)
        #
        sta_time = datetime.now()
        print("-----------------------------开始第", iter_num, "轮迭代-----------------------------")
        print("-----------------------------开始处理数据！-----------------------------")

        os.system(f"fairseq-preprocess --source-lang src --target-lang tgt  \
        --trainpref {path}train --validpref {path}valid --testpref {path}valid  \
        --srcdict data-bin/vocab/dict.src.txt --tgtdict data-bin/vocab/dict.tgt_big.txt \
        --destdir data-bin/{model_type}/iter{iter_num}.oeis")

        print("-----------------------------处理数据完成！-----------------------------")

        process_data_time = datetime.now()

        print("-----------------------------开始训练！-----------------------------")

        os.system(f"CUDA_VISIBLE_DEVICES=2,3,4,5 fairseq-train data-bin/{model_type}/iter{iter_num}.oeis \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
        --warmup-updates 1000 --warmup-init-lr '1e-07' \
        --batch-size 256 --max-epoch 100 \
        --arch transformer --save-dir checkpoints/{model_type}/iter{iter_num}  --no-epoch-checkpoints \
        --log-interval 96 --fp16  \
        --log-file checkpoints/{model_type}/iter{iter_num}/file.log")

        print("-----------------------------训练结束！-----------------------------")

        train_time = datetime.now()

        print("-----------------------------开始束搜索解码！-----------------------------")
        os.system(f"CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/oeis6-25pre1_all_4/test1.oeis \
        --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result1 \
                  & CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/oeis6-25pre1_all_4/test2.oeis \
        --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result2 \
                  & CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/oeis6-25pre1_all_4/test3.oeis \
        --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result3 \
                  & CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/oeis6-25pre1_all_4/test4.oeis \
        --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result4"
                  )

        # os.system(f"CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/oeis6-25pre1_all_3/test1.oeis \
        # --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        # --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result1 \
        #           & CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/oeis6-25pre1_all_3/test2.oeis \
        # --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        # --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result2 \
        #           & CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/oeis6-25pre1_all_3/test3.oeis \
        # --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        # --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result3"
        #           )

        # os.system(f"CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/oeis6-25_2/test1.oeis \
        #                 --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        #                 --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result1 \
        #                           & CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/oeis6-25_2/test2.oeis \
        #                 --path checkpoints/{model_type}/iter{iter_num}/checkpoint_best.pt \
        #                 --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result2 "
        #           )

        time.sleep(60)

        print("-----------------------------束搜索结束！-----------------------------")

        beam_search_time = datetime.now()
        #
        obtain_csv_res(path3, path, 240, 1, 2, 3, 4)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # obtain_csv_res3(path3, path, 240, 1, 2, 3)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # obtain_csv_res2(path3, path, 32, 1, 2)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # result2,result3,result4,result5

        print("-----------------------------开始检查！-----------------------------")
        check_sta_time = datetime.now()
        temp_res_path = check_get_final_res(path)
        check_end_time = datetime.now()
        print("-----------------------------检查结束！-----------------------------")

        process_temp_res2(path, env, temp_res_path)

        train_nums, new_find_formula_nums, no_find_train_nums, shorter_formula_nums = check_new_formula(path)

        obtain_new_train_set(path, path2)  # 获取新的训练数据
        # readom_select_train_data(path2, 0.5) #获取训练数据中的一定比例来进行训练
        str_instrest_seq, instrest_count = check_isfind_complex_formula(path)

        acc_1w = test_1wan_easy_acc(35, path)

        # 发现hard,more的公式的数量
        hard_more_nums = find_hard_more_formula(
            '/home/skm21/fairseq-0.12.0_213/process_result/oeis_data_all_keyword.csv', path)

        end_time = datetime.now()
        f.write(f"iter{iter_num}耗时:" + str(end_time - sta_time) + '\n')
        f.write("process_data_bin_time:" + str(process_data_time - sta_time) + '\n')
        f.write("train_time:" + str(train_time - process_data_time) + '\n')
        f.write("beam_search_time:" + str(beam_search_time - train_time) + '\n')
        f.write("check_time:" + str(check_end_time - check_sta_time) + '\n')
        f.write("已发现序列公式数量:" + str(train_nums) + '\n')
        f.write("新发现序列公式数量:" + str(new_find_formula_nums) + '\n')
        f.write("未发现训练集序列公式数量:" + str(no_find_train_nums) + '\n')
        f.write("序列公式变得更加简洁数量:" + str(shorter_formula_nums) + '\n')
        f.write("发现前一万条easy的大于35term的acc:" + str(acc_1w) + '\n')
        f.write("发现有趣oeis序列数量:" + str(instrest_count) + '\n')
        f.write("发现的有趣序列如下:" + str_instrest_seq + '\n')
        f.write("发现的hard，more序列数量是:" + str(hard_more_nums) + '\n')

        print("-----------------------------第", iter_num, "轮迭代结束-----------------------------")
        # exit()
