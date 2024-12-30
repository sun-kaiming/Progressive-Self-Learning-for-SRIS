import argparse
import os
import re,csv

from check_auto_formula import eval_testset


def eval_test(model_path,gpuid = 0,input_seq_len = 25,beam_size =32,nbest=32,max_tokens=4096,batch_size=128):
    """
    评估测试集的 acc
    """
    
    # model_path = args.model_path
    # test_type = args.test_type
    # max_degree = args.max_degree
    # input_seq_len = args.input_seq_len
    # beam_size = args.beam_size
    # nbest = args.nbest

    # max_tokens = args.max_tokens

    file_name_with_extension = os.path.basename(model_path)

    # 去掉扩展名
    model_name = os.path.splitext(file_name_with_extension)[0]

    # env, file_name = get_env()

    results_path = f'result/test/{model_name}_beamsize{beam_size}_nbest{nbest}/'
    if model_name=="merge_data_4500w_epoch10":
        test_data_list=[
            ("data-bin/easy1w_vocab5","data_oeis/1wan_easy_testdata_35.csv"),
            ("data-bin/sign1w_vocab5","data_oeis/1wan_sign_testdata_35.csv"),
            ("data-bin/base1w_vocab5","data_oeis/1wan_base_testdata_35.csv"),
            ("data-bin/random1w_vocab5","data_oeis/random_1w_testData_new.csv")
        ]
    else:
        test_data_list=[
            ("data-bin/easy1w_vocab4","data_oeis/1wan_easy_testdata_35.csv"),
            ("data-bin/sign1w_vocab4","data_oeis/1wan_sign_testdata_35.csv"),
            ("data-bin/base1w_vocab4","data_oeis/1wan_base_testdata_35.csv"),
            ("data-bin/random1w_vocab4","data_oeis/random_1w_testData_new.csv")
        ]
    # test_path = f"data_oeis/1wan_{test_type}_testdata_{input_seq_len + 10}.csv"

    # if input_seq_len == 15:
    #     data_bin_path = f'data-bin/{test_type}1w_25_vocab4'
    # elif max_degree == 12:
    #     data_bin_path = f'data-bin/{test_type}1w'
    # elif max_degree == 6:
    #     data_bin_path = f'data-bin/{test_type}1w_0-6recur'
    # else:
    #     data_bin_path = f'data-bin/{test_type}1w_{vocab_type}'
    # data_bin_path='data-bin/vocab4'
    # data_bin_path='data-bin/test_1201-4/iter1.oeis'
    with open("save/result/test_result.csv",'a',encoding='utf-8',newline='')as fw, open("save/result/test_result.log",'w',encoding='utf-8')as fw_log:
        writer=csv.writer(fw)
        # writer.writerow(['model_name','easy_predict_1',"easy_predict_10","easy_predict_all",'sign_predict_1',"sign_predict_10","sign_predict_all",'base_predict_1',"base_predict_10","base_predict_all",'random_predict_1',"random_predict_10","random_predict_all"])
        res_list=[model_name]
        fw_log.write(f"请注意，评测一个测试集大约需要10分钟，共有4个测试集，大约需要40分钟。测试日志不会异步刷新，需要点击刷新按钮才可刷新。"+"\n\n")
        fw_log.flush()
        for data_bin_path,test_path in test_data_list:
            print("data_bin_path:",data_bin_path)
            print("model_name:",model_name)
            print("正在评测：",test_path)
            fw_log.write(f"正在评测：{test_path}\n")
            fw_log.flush()
            jiema = f"CUDA_VISIBLE_DEVICES={gpuid} fairseq-generate {data_bin_path} \
                --path {model_path} \
                --max-tokens {max_tokens} --batch-size {batch_size} --beam {beam_size} --nbest {nbest} --results-path {results_path}"
            proce_res = f"grep ^H {results_path}/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 > {results_path}pre_res.txt"
            #
            fw_log.write(f"开始解码"+"\n")
            fw_log.flush()

            os.system(jiema)
            
            os.system(proce_res)

            formula_res_path = f"{results_path}pre_res.txt"
            correct_res_save_path = f"{results_path}" + 'result.csv'
            acc_res_save_path = f"{results_path}" + 'result.txt'
        
            fw_log.write(f"开始评测 "+"\n")
            fw_log.flush()

            count_pre1_correct,count_pre10_correct,count_preall_correct=eval_testset(test_path, formula_res_path, nbest, input_seq_len, correct_res_save_path, acc_res_save_path)
            res_list.extend([count_pre1_correct,count_pre10_correct,count_preall_correct])
            fw_log.write(f"评测结果如下 ")
            fw_log.write('预测后1位acc: '+count_pre1_correct+"\n")
            fw_log.write('预测后10位acc: '+count_pre10_correct+"\n")
            fw_log.write('预测所有位acc: '+count_preall_correct+"\n\n")
            fw_log.flush()

            print('预测后1位acc', count_pre1_correct)
            print('预测后10位acc', count_pre10_correct)
            print('预测所有位acc', count_preall_correct)
        writer.writerow(res_list)




if __name__ == '__main__':
    parser2 = argparse.ArgumentParser(description='eval_test_acc')

    # parser2.add_argument('--gpuid', type=int, default=0, help='使用的gpuid')

    parser2.add_argument('--gpuid', type=int, default=2, help='使用的gpuid')
    # parser2.add_argument('--test_type', type=str, default="base", help='选择待测试的数据及类型，有三种：easy,sign,base')
    # parser2.add_argument('--max_degree', type=int, default=-1, help='公式的最大递推度，会影响到词表的选择，进而影响bin文件的选择')
    parser2.add_argument('--input_seq_len', type=int, default=25, help='测试集输入序列的项数')
    parser2.add_argument('--beam_size', type=int, default=32, help='解码束搜索的宽度')
    parser2.add_argument('--nbest', type=int, default=32, help='输出的候选解数量')
    parser2.add_argument('--model_path', type=str,
                         default="/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter50/model0/checkpoint_best.pt",
                         help='待测试模型路径')

    parser2.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser2.add_argument('--max_tokens', type=int, default=4096, help='批次中最大tokens数量')

    # parser2.add_argument('--data_type', type=str, default='oeis',
    #                      help='测试数据类型为两类，一类是random: 随机生成的1w条,第二类是oeis: 测试OEIS测试集')
    # parser2.add_argument('--vocab_type', type=str, default='vocab4', help='词表类型有三种：vocab2,vocab3,vocab4')

    args2 = parser2.parse_args()
    model_path=args2.model_path
    eval_test(model_path)
