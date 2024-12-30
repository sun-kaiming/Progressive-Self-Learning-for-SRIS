import csv


def new_testSet_formulas():
    path1 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/train.csv'  # SL-All iter50的训练集
    path2 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train/iter51/result/train.csv'  # 协同训练的 Iter50  输入项数为15的训练集
    path3 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter35/result/train.csv'  # 协同训练的 Iter2  输入项数为25的训练集

    test_easy_path = '/home/skm21/fairseq-0.12.0/data_oeis/1wan_easy_testdata_35.csv'
    test_sign_path = '/home/skm21/fairseq-0.12.0/data_oeis/1wan_sign_testdata_35.csv'
    test_base_path = '/home/skm21/fairseq-0.12.0/data_oeis/1wan_base_testdata_35.csv'

    save_path = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/found_new_testSet_formulas.csv'
    with open(path1, 'r') as f1, open(path2, 'r') as f2, open(path3, 'r') as f3:
        with open(test_easy_path, 'r') as f_easy, open(test_sign_path, 'r') as f_sign, open(test_base_path,
                                                                                            'r') as f_base:
            with open(save_path, 'w') as f_new:
                sl_all_reader = csv.reader(f1)
                ct_15_reader = csv.reader(f2)
                ct_25_reader = csv.reader(f3)

                easy_reader = csv.reader(f_easy)
                sign_reader = csv.reader(f_sign)
                base_reader = csv.reader(f_base)

                new_writer = csv.writer(f_new)

                dic_test_set = {}
                for row in easy_reader:
                    dic_test_set[row[0].strip()] = 1
                for row in sign_reader:
                    dic_test_set[row[0].strip()] = 1
                for row in base_reader:
                    dic_test_set[row[0].strip()] = 1

                dic_sl_all_testSet = {}
                for row in sl_all_reader:
                    if row[0].strip() in dic_test_set:
                        dic_sl_all_testSet[row[0].strip()] = 1

                dic_ct_testSet = {}
                for row in ct_15_reader:
                    if row[0].strip() in dic_test_set:
                        if row[0].strip() not in dic_ct_testSet:
                            dic_ct_testSet[row[0].strip()] = [row]
                        else:
                            dic_ct_testSet[row[0].strip()].append(row)
                for row in ct_25_reader:
                    if row[0].strip() in dic_test_set:
                        if row[0].strip() not in dic_ct_testSet:
                            dic_ct_testSet[row[0].strip()] = [row]
                        else:
                            dic_ct_testSet[row[0].strip()].append(row)

                count = 0
                for k, row_lis in dic_ct_testSet.items():
                    if k not in dic_sl_all_testSet:
                        print(k)
                        count += 1
                        for row in row_lis:
                            new_writer.writerow(row)
                print(count)


def new_hard_more_formulas():
    path1 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_SL_all_initData/find_hard_more.csv'  # SL-All iter50的训练集
    path2 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train/iter51/result/find_hard_more.csv'  # 协同训练的 Iter50  输入项数为15的训练集
    path3 = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter37/result/find_hard_more.csv'  # 协同训练的 Iter38  输入项数为25的训练集

    save_path = '/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/found_new_hard_more_formulas.csv'
    with open(path1, 'r') as f1, open(path2, 'r') as f2, open(path3, 'r') as f3:
        with open(save_path, 'w') as f_new:
            sl_all_reader = csv.reader(f1)
            ct_15_reader = csv.reader(f2)
            ct_25_reader = csv.reader(f3)

            new_writer = csv.writer(f_new)

            dic_sl_all_testSet = {}
            for row in sl_all_reader:
                dic_sl_all_testSet[row[0].strip()] = 1

            dic_ct_testSet = {}
            for row in ct_15_reader:
                if row[0].strip() not in dic_ct_testSet:
                    dic_ct_testSet[row[0].strip()] = [row]
                else:
                    dic_ct_testSet[row[0].strip()].append(row)
            for row in ct_25_reader:
                if row[0].strip() not in dic_ct_testSet:
                    dic_ct_testSet[row[0].strip()] = [row]
                else:
                    dic_ct_testSet[row[0].strip()].append(row)

            count = 0
            for k, row_lis in dic_ct_testSet.items():
                if k not in dic_sl_all_testSet:
                    print(k)
                    count += 1
                    for row in row_lis:
                        new_writer.writerow(row)
            print(count)


if __name__ == '__main__':
    # new_testSet_formulas()
    new_hard_more_formulas()
