import os
import time

from recur.utils.create_path import create_no_path
from recur.utils.iter_need_method import obtain_csv_res, comp_1wan_oeis_acc, readom_select_train_data, obtain_csv_res3, \
    obtain_csv_res2, test_1wan_easy_acc, find_hard_more_formula, trans_formula_INT, return_recurrece_deep, \
    obtain_csv_res1, obtain_csv_res6, obtain_csv_res8
from recur.utils.get_env import get_env
from check_oeis import process_temp_res, check_new_formula, obtain_new_train_set, check_isfind_complex_formula, \
    check_seq_in_oeis, check_seq_in_oeis2, process_temp_res2
import pickle
from datetime import datetime

env, file_name = get_env()
import csv
from multiprocessing import Pool
from tqdm import tqdm
from recur.utils.node_class import Node

root = ''
with open('tree/oeis_tree_save_dict_seqNames3.pkl', 'rb') as f:
    root = pickle.load(f)


def more_jincheng_worker(args):
    row, temp_res_path = args

    src = row[0].replace('=', '')
    src = src.split(" ")

    with open(temp_res_path, 'a', encoding='utf-8-sig') as f:
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
                    # if res == True:
                    #     deep_formula = return_recurrece_deep(formula)
                    #     if deep_formula < len_seq:
                    #         if len_seq >= 7:
                    #             seq_name = seq_name.replace("\ufeff", "")
                    #             seq_name = seq_name.replace('﻿A', 'A')
                    #             writer.writerow([seq_name, formula,
                    #                              ", ".join(str(i) for i in list_seq[:len_seq])])
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


def check_get_final_res(path):  # 检测束搜索结果有多少在oeis中 ，将结果保存在csv文件里
    pool = Pool(40)  # 定义一个进程池，最大进程数48

    print("----start----")

    time2 = datetime.now()

    temp_res_path = path + 'temp_res.csv'
    beam_res_path = path + 'generate-test.csv'

    with open(beam_res_path, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        next(reader)
        for j in range(1):
            for k, row in enumerate(tqdm(reader)):
                if k % 1000 == 0:
                    print('检验第' + str(k) + "行的seq")
                args = (row, temp_res_path)
                pool.apply_async(more_jincheng_worker, (args,))
        # 关闭进程池，关闭后po不再接收新的请求
        pool.close()
        # 等待po中所有子进程执行完成，必须放在close语句之后
        pool.join()
        print("-----end-----")
        print("检测seq时间：", datetime.now() - time2)


for iter_num in tqdm(range(1, 51)):
    # model_type = 'model0_32_newdata2'
    model_type = '4500w_combine_train_36w'

    path_before = f"checkpoints/{model_type}/iter{iter_num - 1}/result/"
    path = f"checkpoints/{model_type}/iter{iter_num}/result/"
    path2 = f'checkpoints/{model_type}/iter{iter_num + 1}/result/'
    path3 = f"checkpoints/{model_type}/iter{iter_num}/"  # 为多卡解码做准备
    path_interesting_oeisSeq = 'data_oeis/Interesting_OEIS_seq.txt'
    keywords_path = 'data_oeis/oeis_data_all_keyword.csv'

    with open(path + "key_message_log.txt", 'w', encoding='utf-8') as f:
        create_no_path(path)

        create_no_path(path2)
        #
        sta_time = datetime.now()
        print("-----------------------------开始第", iter_num, "轮迭代-----------------------------")
        print("-----------------------------开始处理数据！-----------------------------")

        os.system(f"fairseq-preprocess --source-lang src --target-lang tgt  \
        --trainpref {path}train --validpref {path}valid --testpref {path}valid  \
        --srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
        --destdir data-bin/{model_type}/iter{iter_num}.oeis \
                 & fairseq-preprocess --source-lang src --target-lang tgt  \
        --trainpref {path}train2 --validpref {path}valid2 --testpref {path}valid2  \
        --srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
        --destdir data-bin/{model_type}/iter{iter_num}_2.oeis")

        print("-----------------------------处理数据完成！-----------------------------")

        process_data_time = datetime.now()

        print("-----------------------------开始训练！-----------------------------")
        # transformer_tiny
        os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train data-bin/{model_type}/iter{iter_num}.oeis \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
        --warmup-updates 200 --warmup-init-lr '1e-07' \
        --batch-size 256 --max-epoch 100 \
        --arch transformer --save-dir checkpoints/{model_type}/iter{iter_num}/model0  --keep-best-checkpoints 1 --no-epoch-checkpoints  \
        --log-interval 96 --fp16  \
        --log-file checkpoints/{model_type}/iter{iter_num}/file.log \
                  & CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train data-bin/{model_type}/iter{iter_num}_2.oeis \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0004 --clip-norm 0.1 --dropout 0.2 \
        --warmup-updates 200 --warmup-init-lr '1e-07' \
        --batch-size 256 --max-epoch 200 \
        --arch transformer --save-dir checkpoints/{model_type}/iter{iter_num}/model1  --keep-best-checkpoints 1 --no-epoch-checkpoints  \
        --log-interval 96 --fp16  \
        --log-file checkpoints/{model_type}/iter{iter_num}/file.log")
        time.sleep(10)

        print("-----------------------------训练结束！-----------------------------")

        train_time = datetime.now()

        print("-----------------------------开始束搜索解码！-----------------------------")
        os.system(f"CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/easy36w_35_split4/test1 \
        --path checkpoints/{model_type}/iter{iter_num}/model0/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result1 \
                  & CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/easy36w_35_split4/test2 \
        --path checkpoints/{model_type}/iter{iter_num}/model0/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result2 \
                  & CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/easy36w_35_split4/test3 \
        --path checkpoints/{model_type}/iter{iter_num}/model0/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result3 \
                    & CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/easy36w_35_split4/test4 \
        --path checkpoints/{model_type}/iter{iter_num}/model0/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result4 \
        \
                  & CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/easy36w_35_split4/test1 \
        --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
        --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result5 \
                & CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/easy36w_35_split4/test2 \
       --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
       --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result6 \
                & CUDA_VISIBLE_DEVICES=6 fairseq-generate data-bin/easy36w_35_split4/test3 \
       --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
       --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result7 \
                 & CUDA_VISIBLE_DEVICES=7 fairseq-generate data-bin/easy36w_35_split4/test4 \
       --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
       --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result8 \
                  ")

        time.sleep(30)

        print("-----------------------------束搜索结束！-----------------------------")

        beam_search_time = datetime.now()
        #
        obtain_csv_res8(path3, path, 240, 1, 2, 3, 4, 5, 6, 7, 8)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # obtain_csv_res6(path3, path, 240, 1, 2, 3, 4,5,6)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # obtain_csv_res3(path3, path, 240, 1, 2, 3)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # obtain_csv_res2(path3, path, 32, 1, 2)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # obtain_csv_res1(path3, path, 32, 1)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
        # result2,result3,result4,result5

        print("-----------------------------开始检查！-----------------------------")
        check_sta_time = datetime.now()
        check_get_final_res(path)
        check_end_time = datetime.now()
        print("-----------------------------检查结束！-----------------------------")

        process_temp_res2(path_before, path, path2, env)

        train_nums, new_find_formula_nums = check_new_formula(path, path2)

        obtain_new_train_set(path2)  # 获取新的训练数据
        # readom_select_train_data(path2, 0.5) #获取训练数据中的一定比例来进行训练
        str_instrest_seq, instrest_count = check_isfind_complex_formula(path, path_interesting_oeisSeq)

        easy_acc_1w = test_1wan_easy_acc("data_oeis/1wan_easy_testdata_35.csv", path2 + 'train.csv')
        sign_acc_1w = test_1wan_easy_acc("data_oeis/1wan_sign_testdata_35.csv", path2 + 'train.csv')
        base_acc_1w = test_1wan_easy_acc("data_oeis/1wan_base_testdata_35.csv", path2 + 'train.csv')

        # 发现hard,more的公式的数量
        hard_more_nums = find_hard_more_formula(keywords_path, path2)

        end_time = datetime.now()
        f.write(f"iter{iter_num}耗时:" + str(end_time - sta_time) + '\n')
        f.write("process_data_bin_time:" + str(process_data_time - sta_time) + '\n')
        f.write("train_time:" + str(train_time - process_data_time) + '\n')
        f.write("beam_search_time:" + str(beam_search_time - train_time) + '\n')
        f.write("check_time:" + str(check_end_time - check_sta_time) + '\n')
        f.write("已发现序列公式数量:" + str(train_nums) + '\n')
        f.write("新发现序列公式数量:" + str(new_find_formula_nums) + '\n')
        f.write("发现前一万条easy的大于35term的acc:" + str(easy_acc_1w) + '\n')
        f.write("发现前一万条sign的大于35term的acc:" + str(sign_acc_1w) + '\n')
        f.write("发现前一万条base的大于35term的acc:" + str(base_acc_1w) + '\n')
        f.write("发现有趣oeis序列数量:" + str(instrest_count) + '\n')
        f.write("发现的有趣序列如下:" + str_instrest_seq + '\n')
        f.write("发现的hard，more序列数量是:" + str(hard_more_nums) + '\n')

        print("-----------------------------第", iter_num, "轮迭代结束-----------------------------")
        # exit()
