import os
import time

from recur.utils.create_path import create_no_path
from recur.utils.iter_need_method import obtain_csv_res, comp_1wan_oeis_acc, readom_select_train_data, obtain_csv_res3, \
    obtain_csv_res2, test_1wan_easy_acc, find_hard_more_formula, trans_formula_INT, return_recurrece_deep, \
    obtain_csv_res1, obtain_csv_res6, obtain_csv_res_circle
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
import argparse
from untils.tool import split_test_data,obtain_new_train_set_init

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

    # print("----start----")

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
        # print("-----end-----")
        # print("检测seq时间：", datetime.now() - time2)


def create_decoder_seq_bin(split_nums, decoder_seq_set_path, split_seq_path, args):  # 创建解码数列二进制文件
    
    if not os.path.exists(decoder_seq_set_path):
        os.makedirs(decoder_seq_set_path)
    if len(os.listdir(decoder_seq_set_path))==0 :
        split_test_data('data_oeis/oeis_data_all_keyword.csv', split_nums, split_seq_path, args.small_oeis_testset)
        print("split_nums:", split_nums)
        for i in range(split_nums):
            os.system(f"fairseq-preprocess --source-lang src --target-lang tgt  \
                                        --trainpref {split_seq_path}/test_{i + 1} --validpref {split_seq_path}/test_{i + 1} --testpref {split_seq_path}/test_{i + 1}  \
                                        --srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
                                        --destdir {decoder_seq_set_path}/test_{i + 1}")
    else:
        print(f"解码数列{decoder_seq_set_path}文件已存在！")

# def create_split_seq(split_seq_path,args):
#     if not os.path.exists(split_seq_path):
#         os.makedirs(split_seq_path)

def train_main(args):
    sta_iter_id = args.sta_iter_id
    end_iter_id = args.end_iter_id
    model_type = args.model_type
    is_combine = args.is_combine
    gpus_id = args.gpus_id
    gpus_lis = gpus_id.split(",")

    gpus_nums = len(gpus_lis)

    if gpus_nums < 2:  # gpu个数小于2，不能进行联合训练
        is_combine = "False"
    if is_combine=="True":  # 如果采用联合训练，则需要将gpus_id分为两部分
        print("00000")
        if len(gpus_lis) % 2 != 0:
            gpus_nums -= 1
            gpus_lis = gpus_lis[:-1]
        gpus_id_model1 = ','.join(gpus_lis[:gpus_nums // 2])
        gpus_id_model2 = ','.join(gpus_lis[gpus_nums // 2:])
    print("#"*100)
    print("is_combine:",is_combine)
    print("is_combine_type:",type(is_combine))

    if is_combine=="False":
        print("11111111111111``")
        split_nums = len(gpus_lis)
        decoder_seq_set_path = f"./data-bin/decoder_seq_set/oeis_seq_split{split_nums}"
        split_seq_path = f'./data_oeis/oeis_seq_split{split_nums}'
    else:
        print("222222222222222222222222")
        split_nums = gpus_nums // 2
        decoder_seq_set_path = f"./data-bin/decoder_seq_set/oeis_seq_split{split_nums}"
        split_seq_path = f'./data_oeis/oeis_seq_split{split_nums}'

        obtain_new_train_set_init("/home/skm21/OEIS_Sequence_Formula_discovery-main/save_formulas/init_oeis_data/","/home/skm21/OEIS_Sequence_Formula_discovery-main/save_formulas/init_oeis_data_rate/",args)


    print("23333333333333333333333333333args.small_oeis_testset:",args.small_oeis_testset)
    if args.small_oeis_testset=="True":
        decoder_seq_set_path += '_small_seqSet'
        split_seq_path += '_small_seqSet'
    
    # create_split_seq(split_seq_path,args) #创建OEIS序列按照GPU数量进行切割的并将其转换为可以直接进行编码的格式
    create_decoder_seq_bin(split_nums, decoder_seq_set_path, split_seq_path, args)

    train_log=f"checkpoints/{model_type}/train.log"
    # exit()
    for iter_num in tqdm(range(sta_iter_id, end_iter_id+1)):
        path_before = f"checkpoints/{model_type}/iter{iter_num - 1}/result/"
        path = f"checkpoints/{model_type}/iter{iter_num}/result/"
        path2 = f'checkpoints/{model_type}/iter{iter_num + 1}/result/'
        path3 = f"checkpoints/{model_type}/iter{iter_num}/"  # 为多卡解码做准备
        path_interesting_oeisSeq = 'data_oeis/Interesting_OEIS_seq.txt'
        keywords_path = 'data_oeis/oeis_data_all_keyword2.csv'

        path_iter0 = f"checkpoints/{model_type}/iter0/result/"
        path_iter1 = f"checkpoints/{model_type}/iter1/result/"

        
        if not os.path.exists(path_iter0):
            os.makedirs(path_iter0)
            os.makedirs(path_iter1)
            os.system(f"cp ./save_formulas/final_res.json {path_iter0}")
            os.system(f"cp /home/skm21/OEIS_Sequence_Formula_discovery-main/save_formulas/init_oeis_data/train.csv /home/skm21/OEIS_Sequence_Formula_discovery-main/save_formulas/init_oeis_data_rate")
            os.system(f"cp /home/skm21/OEIS_Sequence_Formula_discovery-main/save_formulas/init_oeis_data_rate/* {path_iter1}")

        with open(path + "key_message_log.txt", 'w', encoding='utf-8') as f,open(train_log, 'a', encoding='utf-8') as f_train_log:
            create_no_path(path)

            create_no_path(path2)
            #
            sta_time = datetime.now()
            
            print("-----------------------------开始第", iter_num, "轮迭代-----------------------------")
            print("-----------------------------开始处理数据！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 开始迭代\n")
            f_train_log.write(f"Iter{iter_num} 开始处理数据！\n")
            f_train_log.flush()
            os.system(f"fairseq-preprocess --source-lang src --target-lang tgt  \
                    --trainpref {path}train --validpref {path}valid --testpref {path}valid  \
                    --srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
                    --destdir data-bin/{model_type}/iter{iter_num}.oeis")

            if is_combine=="True":  # 采用联合训练，处理第二部分的数据
                os.system(f"fairseq-preprocess --source-lang src --target-lang tgt  \
                        --trainpref {path}train2 --validpref {path}valid2 --testpref {path}valid2  \
                        --srcdict data-bin/vocab4/dict.src.txt --tgtdict data-bin/vocab4/dict.tgt.txt \
                        --destdir data-bin/{model_type}/iter{iter_num}_2.oeis")

            print("-----------------------------处理数据完成！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 处理数据完成！\n")
            f_train_log.flush()
            process_data_time = datetime.now()

            print("-----------------------------开始训练！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 开始训练！\n")
            f_train_log.flush()
            
            if is_combine=="True":
                os.system(f"CUDA_VISIBLE_DEVICES={gpus_id_model1} fairseq-train data-bin/{model_type}/iter{iter_num}.oeis \
                --optimizer {args.optimizer} --lr-scheduler {args.lr_scheduler} --lr {args.lr} --clip-norm {args.clip_norm} --dropout {args.dropout} \
                --warmup-updates {args.warmup_updates} --warmup-init-lr {args.warmup_init_lr} \
                --batch-size {args.batch_size} --max-epoch {args.max_epoch} \
                --arch {args.arch} --save-dir checkpoints/{model_type}/iter{iter_num}/model1  --keep-best-checkpoints {args.save_checkpoints_nums} --no-epoch-checkpoints  \
                --log-interval {args.log_interval} --fp16  \
                --log-file checkpoints/{model_type}/iter{iter_num}/file.log \
                        & CUDA_VISIBLE_DEVICES={gpus_id_model2} fairseq-train data-bin/{model_type}/iter{iter_num}_2.oeis \
               --optimizer {args.optimizer} --lr-scheduler {args.lr_scheduler} --lr {args.lr} --clip-norm {args.clip_norm} --dropout {args.dropout} \
                --warmup-updates {args.warmup_updates} --warmup-init-lr {args.warmup_init_lr} \
                --batch-size {args.batch_size} --max-epoch {args.max_epoch} \
                --arch {args.arch} --save-dir checkpoints/{model_type}/iter{iter_num}/model2  --keep-best-checkpoints {args.save_checkpoints_nums} --no-epoch-checkpoints  \
                --log-interval {args.log_interval} --fp16  \
                --log-file checkpoints/{model_type}/iter{iter_num}/file.log")
            else:
                os.system(f"CUDA_VISIBLE_DEVICES={gpus_id} fairseq-train data-bin/{model_type}/iter{iter_num}.oeis \
                            --optimizer {args.optimizer} --lr-scheduler {args.lr_scheduler} --lr {args.lr} --clip-norm {args.clip_norm} --dropout {args.dropout} \
                            --warmup-updates {args.warmup_updates} --warmup-init-lr {args.warmup_init_lr} \
                            --batch-size {args.batch_size} --max-epoch {args.max_epoch} \
                            --arch {args.arch} --save-dir checkpoints/{model_type}/iter{iter_num}/model1  --keep-best-checkpoints {args.save_checkpoints_nums} --no-epoch-checkpoints  \
                            --log-interval {args.log_interval} --fp16  \
                            --log-file checkpoints/{model_type}/iter{iter_num}/file.log")
            time.sleep(10)

            print("-----------------------------训练完成！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 训练完成！\n")
            train_time = datetime.now()

            print("-----------------------------开始束搜索解码！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 开始束搜索解码！\n")
            f_train_log.flush()

            if is_combine=="True":
                command_lis = []
                for i in range(split_nums):
                    command_lis.append(
                        f"CUDA_VISIBLE_DEVICES={gpus_lis[i]} fairseq-generate {decoder_seq_set_path}/test_{i + 1} \
                --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
                --max-tokens {args.max_tokens} --batch-size {args.batch_size_decoder} --beam {args.beam_size} --nbest {args.n_best} --results-path checkpoints/{model_type}/iter{iter_num}/result{i + 1}")

                for i in range(split_nums):
                    command_lis.append(
                        f"CUDA_VISIBLE_DEVICES={gpus_lis[i + split_nums]} fairseq-generate {decoder_seq_set_path}/test_{i + 1} \
                --path checkpoints/{model_type}/iter{iter_num}/model2/checkpoint_best.pt \
                --max-tokens {args.max_tokens} --batch-size {args.batch_size_decoder} --beam {args.beam_size} --nbest {args.n_best} --results-path checkpoints/{model_type}/iter{iter_num}/result{i + 1 + split_nums}")
                command = ' & '.join(command_lis)
                print("解码命令：", command)
                os.system(command)
            #  os.system(f"CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/easy36w_35_split3/test1 \
            #  --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
            #  --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result1 \
            #            & CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/easy36w_35_split3/test2 \
            #  --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
            #  --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result2 \
            #            & CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/easy36w_35_split3/test3 \
            #  --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
            #  --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result3 \
            #  \
            #            & CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/easy36w_35_split3/test1 \
            #  --path checkpoints/{model_type}/iter{iter_num}/model2/checkpoint_best.pt \
            #  --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result4 \
            #          & CUDA_VISIBLE_DEVICES=4 fairseq-generate data-bin/easy36w_35_split3/test2 \
            # --path checkpoints/{model_type}/iter{iter_num}/model2/checkpoint_best.pt \
            # --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result5 \
            #          & CUDA_VISIBLE_DEVICES=5 fairseq-generate data-bin/easy36w_35_split3/test3 \
            # --path checkpoints/{model_type}/iter{iter_num}/model2/checkpoint_best.pt \
            # --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32 --results-path checkpoints/{model_type}/iter{iter_num}/result6 \
            #            ")
            else:
                command_lis = []
                for i in range(split_nums):
                    command_lis.append(
                        f"CUDA_VISIBLE_DEVICES={gpus_lis[i]} fairseq-generate {decoder_seq_set_path}/test_{i + 1} \
                                --path checkpoints/{model_type}/iter{iter_num}/model1/checkpoint_best.pt \
                                --max-tokens {args.max_tokens} --batch-size {args.batch_size_decoder} --beam {args.beam_size} --nbest {args.n_best} --results-path checkpoints/{model_type}/iter{iter_num}/result{i + 1}")

                command = ' & '.join(command_lis)
                print("解码命令：", command)
                os.system(command)

            time.sleep(30)

            print("-----------------------------束搜索完成！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 束搜索解码完成！\n")
            f_train_log.flush()

            beam_search_time = datetime.now()
            #
            if  is_combine=="False":
                obtain_csv_res_circle(path3, path, 240, split_nums)
            else:
                obtain_csv_res_circle(path3, path, 240, split_nums * 2)
            # obtain_csv_res6(path3, path, 240, 1, 2, 3, 4,5,6)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
            # obtain_csv_res3(path3, path, 240, 1, 2, 3)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
            # obtain_csv_res2(path3, path, 32, 1, 2)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
            # obtain_csv_res1(path3, path, 32, 1)  # 把束搜索结果转化为csv格式，方便后续处理 #2,3,4,5为对应的result路径
            # result2,result3,result4,result5

            print("-----------------------------开始检查！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 开始检查！\n")
            f_train_log.flush()
            
            check_sta_time = datetime.now()
            check_get_final_res(path)
            check_end_time = datetime.now()
            print("-----------------------------检查完成！-----------------------------")
            f_train_log.write(f"Iter{iter_num} 检查完成！\n")
            f_train_log.flush()


            process_temp_res2(path_before, path, path2, env)

            train_nums, new_find_formula_nums = check_new_formula(path, path2)

            obtain_new_train_set(path2, args)  # 获取新的训练数据
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

            f_train_log.write(f"iter{iter_num}耗时:" + str(end_time - sta_time) + '\n')
            f_train_log.write("process_data_bin_time:" + str(process_data_time - sta_time) + '\n')
            f_train_log.write("train_time:" + str(train_time - process_data_time) + '\n')
            f_train_log.write("beam_search_time:" + str(beam_search_time - train_time) + '\n')
            f_train_log.write("check_time:" + str(check_end_time - check_sta_time) + '\n')
            f_train_log.write(f"已发现序列公式数量:" + str(train_nums) + '\n')
            f_train_log.write(f"新发现序列公式数量:" + str(new_find_formula_nums) + '\n')
            f_train_log.write(f"发现前一万条easy的大于35term的acc:" + str(easy_acc_1w) + '\n')
            f_train_log.write(f"发现前一万条sign的大于35term的acc:" + str(sign_acc_1w) + '\n')
            f_train_log.write(f"发现前一万条base的大于35term的acc:" + str(base_acc_1w) + '\n')
            f_train_log.write(f"发现有趣oeis序列数量:" + str(instrest_count) + '\n')
            f_train_log.write(f"发现有趣oeis序列如下:" + str_instrest_seq + '\n')
            f_train_log.write(f"发现hard，more序列数量是:" + str(hard_more_nums) + '\n')
            
            

            print("-----------------------------第", iter_num, "轮迭代完成-----------------------------")
            f_train_log.write(f"Iter{iter_num} 迭代完成！\n\n")
            f_train_log.flush()

            # exit()


def get_args():
    parser = argparse.ArgumentParser(usage="参数", description="help info.")

    ## 基础参数
    parser.add_argument('--sta-iter-id', type=int, default=1, help='开始迭代的id,默认从1开始')
    parser.add_argument('--end-iter-id', type=int, default=5, help='迭代结束的id,默认以5结束')
    parser.add_argument('--model-type', type=str, default="test_demo", help='迭代的名称，每个迭代最好设置一个不同的名称，防止检查点被覆盖')
    parser.add_argument('--is-combine', type=str, default="False", help='是否采用联合训练的方式，联合训练在每次迭代时同时训练两个模型，详细情况参见大论文第四章。')
    parser.add_argument('--gpus-id', type=str, default="0,1,2,3,4,5", help='可用的GPUs Id，注意：联合训练至少要存在两张显卡，否则无法进行联合训练')

    ## 训练参数
    parser.add_argument('--optimizer', type=str, default="adam", help='优化器')
    parser.add_argument('--lr-scheduler', type=str, default="inverse_sqrt", help='lr衰减策略')
    parser.add_argument('--lr', type=float, default=0.0004, help='学习率')
    parser.add_argument('--clip-norm', type=float, default=0.1, help='lr衰减策略')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--warmup-updates', type=int, default=200, help='学习率预热步数')
    parser.add_argument('--warmup-init-lr', type=str, default="1e-07", help='预热初始学习率')
    parser.add_argument('--batch-size', type=int, default=256, help='batch-size')
    parser.add_argument('--max-epoch', type=int, default=100, help='最大epochs')
    parser.add_argument('--arch', type=str, default="transformer", help='训练所用的模型架构')
    parser.add_argument('--save-checkpoints-nums', type=int, default=1, help='每轮迭代保存的检查点/模型数量')
    parser.add_argument('--log-interval', type=int, default=96, help='每轮迭代保存的检查点/模型数量')

    ## 解码参数  --max-tokens 4096 --batch-size 128 --beam 32 --nbest 32
    parser.add_argument('--max-tokens', type=int, default=4096, help='解码中同一批次的最大tokens数量')
    parser.add_argument('--batch-size-decoder', type=int, default=128, help='解码过程中的批次大小')
    parser.add_argument('--beam-size', type=int, default=32, help='束搜索宽度')
    parser.add_argument('--n-best', type=int, default=32, help='每次解码保留的候选解数量')
    parser.add_argument('--small-oeis-testset', type=str, default="False", help='较小的OEIS序列测试集，便于在更短的时间内检查代码是否正确')

    ## 引入随机性的参数，该参数选择当前数据的一部分用于迭代训练，以缓解自学习的提前收敛问题
    parser.add_argument('--random-rate-model1', type=float, default=1, help='在随机自学习策略训练过程中，第一个模型选择全部OEIS的多少用做第二个模型的数据，')
    parser.add_argument('--random-rate-model2', type=float, default=0.5,
                        help='在随机自学习策略训练过程中，第二个模型选择全部OEIS的多少用做第二个模型的数据，')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    # exit()
    train_main(args)
