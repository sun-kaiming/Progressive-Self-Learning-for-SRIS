import csv
import datetime
import random
from recur.utils.split_oesi_seq import process_oeis_input_seq
from recur.utils.decode import decode


def obtain_csv_res_circle(res_path, csv_path, beam_nbest_size, result_nums):  # 把束搜索的结果转化为csv结构,
    print()
    lines = []
    with open(csv_path + 'generate-test.csv', 'w', encoding='utf-8-sig', newline='') as f_w:
        for i in range(result_nums):
            with open(res_path + f'result{i + 1}/generate-test.txt', 'r', encoding='utf-8') as f_r:

                if beam_nbest_size > 31:
                    beam_nbest_size = 31
                lines += f_r.readlines()

        writer = csv.writer(f_w)
        header = ['seq'] + ["ans" + str(ans) for ans in range(beam_nbest_size)]
        # print(header)
        # exit()
        writer.writerow(header)
        list_write = []
        flag = 0
        for i, line in enumerate(lines):
            # print(i)
            if line[0] == "S":
                if flag == 1:
                    writer.writerow(list_write)
                flag = 1
                list_write = []
                line = line.replace("=", '')
                seq_list = line.strip().split()[1:]
                seq = ' '.join(seq_list)
                list_write.append(seq)
            if line[0] == "H":
                ans_list = line.strip().split()[2:]
                ans = ' '.join(ans_list)
                list_write.append(ans)
                # print(list_write)
                # exit()
        writer.writerow(list_write)
        print("处理完成！")


def obtain_csv_res(res_path, csv_path, beam_nbest_size, num1, num2, num3, num4):  # 把束搜索的结果转化为csv结构,
    with open(res_path + f'result{num1}/generate-test.txt', 'r', encoding='utf-8') as f1:
        with open(res_path + f'result{num2}/generate-test.txt', 'r', encoding='utf-8') as f2:
            with open(res_path + f'result{num3}/generate-test.txt', 'r', encoding='utf-8') as f3:
                with open(res_path + f'result{num4}/generate-test.txt', 'r', encoding='utf-8') as f4:
                    with open(csv_path + 'generate-test.csv', 'w', encoding='utf-8-sig', newline='') as f5:
                        if beam_nbest_size > 31:
                            beam_nbest_size = 31
                        lines1 = f1.readlines()
                        lines2 = f2.readlines()
                        lines3 = f3.readlines()
                        lines4 = f4.readlines()

                        lines = lines1 + lines2 + lines3 + lines4

                        writer = csv.writer(f5)
                        header = ['seq'] + ["ans" + str(ans) for ans in range(beam_nbest_size)]
                        # print(header)
                        # exit()
                        writer.writerow(header)
                        list_write = []
                        flag = 0
                        for i, line in enumerate(lines):
                            # print(i)
                            if line[0] == "S":
                                if flag == 1:
                                    writer.writerow(list_write)
                                flag = 1
                                list_write = []
                                line = line.replace("=", '')
                                seq_list = line.strip().split()[1:]
                                seq = ' '.join(seq_list)
                                list_write.append(seq)
                            if line[0] == "H":
                                ans_list = line.strip().split()[2:]
                                ans = ' '.join(ans_list)
                                list_write.append(ans)
                                # print(list_write)
                                # exit()
                        writer.writerow(list_write)
                    print("处理完成！")


def obtain_csv_res6(res_path, csv_path, beam_nbest_size, num1, num2, num3, num4, num5, num6):  # 把束搜索的结果转化为csv结构,
    with open(res_path + f'result{num1}/generate-test.txt', 'r', encoding='utf-8') as f1:
        with open(res_path + f'result{num2}/generate-test.txt', 'r', encoding='utf-8') as f2:
            with open(res_path + f'result{num3}/generate-test.txt', 'r', encoding='utf-8') as f3:
                with open(res_path + f'result{num4}/generate-test.txt', 'r', encoding='utf-8') as f4:
                    with open(res_path + f'result{num5}/generate-test.txt', 'r', encoding='utf-8') as f5:
                        with open(res_path + f'result{num6}/generate-test.txt', 'r', encoding='utf-8') as f6:
                            with open(csv_path + 'generate-test.csv', 'w', encoding='utf-8-sig', newline='') as f7:
                                if beam_nbest_size > 31:
                                    beam_nbest_size = 31
                                lines1 = f1.readlines()
                                lines2 = f2.readlines()
                                lines3 = f3.readlines()
                                lines4 = f4.readlines()
                                lines5 = f5.readlines()
                                lines6 = f6.readlines()

                                lines = lines1 + lines2 + lines3 + lines4 + lines5 + lines6

                                writer = csv.writer(f7)
                                header = ['seq'] + ["ans" + str(ans) for ans in range(beam_nbest_size)]
                                # print(header)
                                # exit()
                                writer.writerow(header)
                                list_write = []
                                flag = 0
                                for i, line in enumerate(lines):
                                    # print(i)
                                    if line[0] == "S":
                                        if flag == 1:
                                            writer.writerow(list_write)
                                        flag = 1
                                        list_write = []
                                        line = line.replace("=", '')
                                        seq_list = line.strip().split()[1:]
                                        seq = ' '.join(seq_list)
                                        list_write.append(seq)
                                    if line[0] == "H":
                                        ans_list = line.strip().split()[2:]
                                        ans = ' '.join(ans_list)
                                        list_write.append(ans)
                                        # print(list_write)
                                        # exit()
                                writer.writerow(list_write)
                            print("处理完成！")


def obtain_csv_res8(res_path, csv_path, beam_nbest_size, num1, num2, num3, num4, num5, num6, num7,
                    num8):  # 把束搜索的结果转化为csv结构,
    with open(res_path + f'result{num1}/generate-test.txt', 'r', encoding='utf-8') as f1:
        with open(res_path + f'result{num2}/generate-test.txt', 'r', encoding='utf-8') as f2:
            with open(res_path + f'result{num3}/generate-test.txt', 'r', encoding='utf-8') as f3:
                with open(res_path + f'result{num4}/generate-test.txt', 'r', encoding='utf-8') as f4:
                    with open(res_path + f'result{num5}/generate-test.txt', 'r', encoding='utf-8') as f5:
                        with open(res_path + f'result{num6}/generate-test.txt', 'r', encoding='utf-8') as f6:
                            with open(res_path + f'result{num7}/generate-test.txt', 'r', encoding='utf-8') as f7:
                                with open(res_path + f'result{num8}/generate-test.txt', 'r', encoding='utf-8') as f8:
                                    with open(csv_path + 'generate-test.csv', 'w', encoding='utf-8-sig',
                                              newline='') as f9:
                                        if beam_nbest_size > 31:
                                            beam_nbest_size = 31
                                        lines1 = f1.readlines()
                                        lines2 = f2.readlines()
                                        lines3 = f3.readlines()
                                        lines4 = f4.readlines()
                                        lines5 = f5.readlines()
                                        lines6 = f6.readlines()
                                        lines7 = f7.readlines()
                                        lines8 = f8.readlines()

                                        lines = lines1 + lines2 + lines3 + lines4 + lines5 + lines6 + lines7 + lines8

                                        writer = csv.writer(f9)
                                        header = ['seq'] + ["ans" + str(ans) for ans in range(beam_nbest_size)]
                                        # print(header)
                                        # exit()
                                        writer.writerow(header)
                                        list_write = []
                                        flag = 0
                                        for i, line in enumerate(lines):
                                            # print(i)
                                            if line[0] == "S":
                                                if flag == 1:
                                                    writer.writerow(list_write)
                                                flag = 1
                                                list_write = []
                                                line = line.replace("=", '')
                                                seq_list = line.strip().split()[1:]
                                                seq = ' '.join(seq_list)
                                                list_write.append(seq)
                                            if line[0] == "H":
                                                ans_list = line.strip().split()[2:]
                                                ans = ' '.join(ans_list)
                                                list_write.append(ans)
                                                # print(list_write)
                                                # exit()
                                        writer.writerow(list_write)
                                    print("处理完成！")


def obtain_csv_res3(res_path, csv_path, beam_nbest_size, num1, num2, num3):  # 把束搜索的结果转化为csv结构,
    with open(res_path + f'result{num1}/generate-test.txt', 'r', encoding='utf-8') as f1:
        with open(res_path + f'result{num2}/generate-test.txt', 'r', encoding='utf-8') as f2:
            with open(res_path + f'result{num3}/generate-test.txt', 'r', encoding='utf-8') as f3:
                # with open(res_path+ f'result{num4}/generate-test.txt','r',encoding='utf-8')as f4:
                with open(csv_path + 'generate-test.csv', 'w', encoding='utf-8-sig', newline='') as f5:
                    if beam_nbest_size > 240:
                        beam_nbest_size = 240
                    lines1 = f1.readlines()
                    lines2 = f2.readlines()
                    lines3 = f3.readlines()
                    # lines4=f4.readlines()

                    lines = lines1 + lines2 + lines3

                    writer = csv.writer(f5)
                    header = ['seq'] + ["ans" + str(ans) for ans in range(beam_nbest_size)]
                    # print(header)
                    # exit()
                    writer.writerow(header)
                    list_write = []
                    flag = 0
                    for i, line in enumerate(lines):
                        # print(i)
                        if line[0] == "S":
                            if flag == 1:
                                writer.writerow(list_write)
                            flag = 1
                            list_write = []
                            line = line.replace("=", '')
                            seq_list = line.strip().split()[1:]
                            seq = ' '.join(seq_list)
                            list_write.append(seq)
                        if line[0] == "H":
                            ans_list = line.strip().split()[2:]
                            ans = ' '.join(ans_list)
                            list_write.append(ans)
                            # print(list_write)
                            # exit()
                    writer.writerow(list_write)
                print("处理完成！")


def obtain_csv_res2(res_path, csv_path, beam_nbest_size, num1, num2):  # 把束搜索的结果转化为csv结构,
    with open(res_path + f'result{num1}/generate-test.txt', 'r', encoding='utf-8') as f1:
        with open(res_path + f'result{num2}/generate-test.txt', 'r', encoding='utf-8') as f2:
            # with open(res_path+ f'result{num3}/generate-test.txt','r',encoding='utf-8')as f3:
            # with open(res_path+ f'result{num4}/generate-test.txt','r',encoding='utf-8')as f4:
            with open(csv_path + 'generate-test.csv', 'w', encoding='utf-8-sig', newline='') as f5:
                if beam_nbest_size > 240:
                    beam_nbest_size = 240
                lines1 = f1.readlines()
                lines2 = f2.readlines()
                # lines3=f3.readlines()
                # lines4=f4.readlines()

                lines = lines1 + lines2

                writer = csv.writer(f5)
                header = ['seq'] + ["ans" + str(ans) for ans in range(beam_nbest_size)]
                # print(header)
                # exit()
                writer.writerow(header)
                list_write = []
                flag = 0
                for i, line in enumerate(lines):
                    # print(i)
                    if line[0] == "S":
                        if flag == 1:
                            writer.writerow(list_write)
                        flag = 1
                        list_write = []
                        line = line.replace("=", '')
                        seq_list = line.strip().split()[1:]
                        seq = ' '.join(seq_list)
                        list_write.append(seq)
                    if line[0] == "H":
                        ans_list = line.strip().split()[2:]
                        ans = ' '.join(ans_list)
                        list_write.append(ans)
                        # print(list_write)
                        # exit()
                writer.writerow(list_write)
            print("处理完成！")


def obtain_csv_res1(res_path, csv_path, beam_nbest_size, num1):  # 把束搜索的结果转化为csv结构,
    with open(res_path + f'result{num1}/generate-test.txt', 'r', encoding='utf-8') as f1:
        with open(csv_path + 'generate-test.csv', 'w', encoding='utf-8-sig', newline='') as f5:
            if beam_nbest_size > 240:
                beam_nbest_size = 240
            lines1 = f1.readlines()
            # lines2=f2.readlines()
            # lines3=f3.readlines()
            # lines4=f4.readlines()

            lines = lines1

            writer = csv.writer(f5)
            header = ['seq'] + ["ans" + str(ans) for ans in range(beam_nbest_size)]
            # print(header)
            # exit()
            writer.writerow(header)
            list_write = []
            flag = 0
            for i, line in enumerate(lines):
                # print(i)
                if line[0] == "S":
                    # if flag == 1:
                    #     if int(list_write[0]) <= 10010:
                    writer.writerow(list_write[1:])

                    flag = 1
                    list_write = []
                    line = line.replace("=", '')
                    xuhao = line.strip().split()[0][2:]
                    seq_list = line.strip().split()[1:]
                    seq = ' '.join(seq_list)
                    list_write.append(xuhao)
                    list_write.append(seq)
                if line[0] == "H":
                    ans_list = line.strip().split()[2:]
                    ans = ' '.join(ans_list)
                    list_write.append(ans)
                    # print(list_write)
                    # exit()
            # if int(list_write[0]) <= 10010:
            writer.writerow(list_write[1:])
        print("处理完成！")


def comp_1wan_oeis_acc(path):  # 计算oeis大于35项的前一万条数列的acc
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        count2 = 0
        for i, row in enumerate(reader):
            for j, ch in enumerate(row[0]):
                if ch in [str(num) for num in range(10)]:
                    index = j
                    break
            if int(row[0][index:]) <= 22420:
                # if i<1831:
                # print(row[0][index:])
                # exit()
                if len(row[3].split(", ")) >= 35:
                    count2 += 1

        print("已发现的序列公式中 前一万条正确的数量：", count2)

    return count2 / 10000


def readom_select_train_data(path, rate):
    with open(path + 'train.src', 'r', encoding='utf-8') as f1:
        with open(path + 'train.tgt', 'r', encoding='utf-8') as f2:
            with open(path + f'train_{str(rate)}.src', 'w', encoding='utf-8') as f3:
                with open(path + f'train_{str(rate)}.tgt', 'w', encoding='utf-8') as f4:
                    lines1 = f1.readlines()
                    lines2 = f2.readlines()

                    lis_train = []
                    for i in range(len(lines1)):
                        temp_list = []
                        temp_list.append(lines1[i])
                        temp_list.append(lines2[i])
                        lis_train.append(temp_list)

                    random.shuffle(lis_train)

                    num = int(len(lines1) * rate)

                    for i in range(num):
                        f3.write(lis_train[i][0])
                        f4.write(lis_train[i][1])


def test_1wan_easy_acc(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f1:
        with open(path2, 'r', encoding='utf-8') as f2:
            reader = csv.reader(f1)
            reader2 = csv.reader(f2)
            count_all = 0
            count_pre1 = 0
            count_pre10 = 0
            dict_seq = {}
            for i, row in enumerate(reader2):
                dict_seq[row[0].replace("\ufeff", "")] = row[1:]
            # print(dict_seq)
            for i, row in enumerate(reader):
                if len(row[1][1:].split(',')) >= 35:
                    seq_name = row[0].replace("\ufeff", "")
                    if seq_name in dict_seq:
                        count_all += 1
    print("res:", count_all / 10000)

    return count_all / 10000


def test_findlinearrecurrence():
    with open('/home/skm21/fairseq-0.12.0/data_oeis/wolfram_test/1wan_easy_testdata_25.csv', 'r',
              encoding='utf-8') as f1:
        with open('/home/skm21/fairseq-0.12.0/data_oeis/wolfram_test/findlinarrecurrence_test_res4.2.csv', 'r',
                  encoding='utf-8') as f2:
            # with open('/home/skm21/fairseq-0.12.0/data_oeis/wolfram_test/findsequenceFunction_test_res25.2.csv','w',encoding='utf-8')as f3:
            reader = csv.reader(f1)
            reader2 = csv.reader(f2)
            # writer3=csv.writer(f3)

            count_all = 0
            count_pre1 = 0
            count_pre10 = 0

            dict_data_all = {}
            dict_data_pre1 = {}
            dict_data_pre10 = {}

            count4 = 0
            for row in reader2:
                # if 'DifferenceRoot[Functi' not in row[3]:
                # if row[5]=='success' :
                # if 'DifferenceRoot[Functi' in row[3]:
                if row[5] == 'success' and row[1] == row[2]:
                    # count4+=1
                    # print(count4,row[3],row[5])
                    # writer3.writerow(row)
                    dict_data_all[row[0].replace("\ufeff", "")] = row[1:]

                if row[2] != '':
                    if len(row[1].split(',')) >= 25:
                        ##############################输入25个term ############################################
                        # if row[1].split(',')[:35]==row[2].split(',')[:35]:
                        #     dict_data_pre10[row[0].replace("\ufeff", "")] = row[1:]
                        # if row[1].split(',')[:26]==row[2].split(',')[:26]:
                        #     dict_data_pre1[row[0].replace("\ufeff", "")]= row[1:]

                        ##############################输入15个term ############################################
                        if row[1][1:].split(',')[:25] == row[2][1:].split(',')[:25]:
                            dict_data_pre10[row[0].replace("\ufeff", "")] = row[1:]
                        if row[1][1:].split(',')[:16] == row[2][1:].split(',')[:16]:
                            dict_data_pre1[row[0].replace("\ufeff", "")] = row[1:]
            # print(dict_data_pre1)
            print(len(dict_data_pre1))
            print(len(dict_data_pre10))
            print(len(dict_data_all))
            # exit()
            for i, row in enumerate(reader):
                if i <= 10001:
                    seq_name = row[0].replace("\ufeff", "")

                    if seq_name in dict_data_all:
                        count_all += 1
                        # writer3.writerow(row+dict_data_all[seq_name])
                    if seq_name in dict_data_pre1:
                        count_pre1 += 1
                    if seq_name in dict_data_pre10:
                        count_pre10 += 1

            print(count_pre1)
            print(count_pre10)
            print(count_all)


def test_baseline_pachong():
    with open('/home/skm21/fairseq-0.12.0/data_oeis/wolfram_test/9k_easy_testdata_35.csv', 'r', encoding='utf-8') as f1:
        with open('/home/skm21/fairseq-0.12.0/data_oeis/wolfram_test/easy_seq_35oeis.csv', 'r', encoding='utf-8') as f2:
            reader = csv.reader(f1)
            reader2 = csv.reader(f2)

            count = 0
            count_pre1 = 0
            count_pre10 = 0

            dict_data = {}
            dict_data_pre1 = {}
            dict_data_pre10 = {}

            for row in reader2:
                # if len(row[1].split(','))==25:
                dict_data[row[0].replace("\ufeff", "")] = (row[2][:-1], row[3])
                # else:
                #     print(len(row[1].split(',')),"  ",row[1],row[2],row[3])
            # exit()
            for i, row in enumerate(reader):
                if i <= 10001:
                    seq_name = row[0].replace("\ufeff", "")
                    count += 1
                    if seq_name in dict_data:

                        # print(row[1][1:].split(',')[25:26])
                        # print(dict_data[seq_name][0].split(',')[:1])
                        # print()
                        if row[1][1:].split(',')[25:35] == dict_data[seq_name][0].split(',') or dict_data[seq_name][
                            1] == '0.00%':
                            count_pre10 += 1
                        if row[1][1:].split(',')[25:26] == dict_data[seq_name][0].split(',')[:1] or dict_data[seq_name][
                            1] == '0.00%':
                            count_pre1 += 1

            print(count_pre1, count_pre1 / count)
            print(count_pre10, count_pre10 / count)
            print(len(dict_data))
            print(count)


def find_hard_more_formula(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f1:
        with open(path2 + 'train.csv', 'r', encoding='utf-8') as f2:
            with open(path2 + 'find_hard_more.csv', 'w', encoding='utf-8-sig', newline='') as f3:
                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                writer = csv.writer(f3)
                dict_data = {}
                count = 0
                for row in reader2:
                    if row[0] not in dict_data:
                        dict_data[row[0]] = [row[3], [row[2]]]
                    else:
                        dict_data[row[0]][1].append(row[2])
                writer.writerow(['seq_name', 'formula', 'keyword', 'keyword_all', 'seq_length', 'sequence'])
                for row in reader1:
                    if 'hard' in row[2] or 'more' in row[2]:
                        seq_name = row[0]
                        if seq_name in dict_data:
                            seq_lis = dict_data[seq_name][0]
                            len_seq = len(seq_lis.split(', '))
                            formula_lis = dict_data[seq_name][1]
                            for formula in formula_lis:
                                temp_row = []
                                temp_row.append(seq_name)
                                temp_row.append(formula)
                                if 'hard' in row[2]:
                                    temp_row.append('keyword：hard')
                                else:
                                    temp_row.append('keyword：more')
                                temp_row.append(row[2])
                                temp_row.append(len_seq)
                                temp_row.append(seq_lis)
                                writer.writerow(temp_row)
                            count += 1

    return count


def select_9k_test_set(path, filenum):
    with open(path + f'1wan_easy_testdata_{filenum}.csv', 'r', encoding='utf-8') as f1:
        with open(path + f'easy_seq_{filenum}oeis.csv', 'r', encoding='utf-8') as f2:
            with open(path + f'9k_easy_testdata_{filenum}.csv', 'w', encoding='utf-8') as f3:
                data_list = []
                temp_dict = {}
                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                writer = csv.writer(f3)
                for row in reader2:
                    temp_dict[row[0]] = 1
                print(len(temp_dict))
                for row in reader1:
                    if row[0] in temp_dict:
                        data_list.append(row)
                random.shuffle(data_list)
                for i, row in enumerate(data_list):
                    if i < 9000:
                        writer.writerow(row)


def return_oeis_all_init_term(deep_formula):
    with open('/home/skm21/fairseq-0.12.0_213/data_oeis/stripped', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[4:]
        oeis_dict = {}
        init_term_all_list = []
        init_term_all_dict = {}

        for line in lines:
            seq_name = line[:7]
            seq = line[9:-2]
            init_term = seq.split(',')[:deep_formula]
            init_term = ','.join(init_term)
            init_term_all_dict[init_term] = 1
            # print(seq)
            # print(init_term)
            # exit()
        # print(init_term_all_list)
        flag = 0
        for k, v in init_term_all_dict.items():
            temp_list = k.split(',')
            for term in temp_list[:3]:
                if int(term) > 200:
                    flag = 1
                    break
            if flag != 1:
                init_term_all_list.append(k.split(','))
    return init_term_all_list


def return_oeis_random_init_term(init_term_list, num):
    init_term_random_list = []
    if num > len(init_term_list):
        # print("1222222222222222222222")
        num = len(init_term_list)

    index_list = []
    index_list = random.sample(range(0, len(init_term_list)), num)

    for index in index_list:
        # index = random.randint(1, len(init_term_list))
        init_term_temp = process_oeis_input_seq(init_term_list[index], 25)
        init_term_random_list.append(init_term_temp.split())
    return init_term_random_list


def test_threeDataSet_acc(path1, path2, type, iter_staid, iter_endid):
    """
    测试100轮迭代结果在三种测试集上的acc,
    path1:测试集路径
    path2:训练结果
    type:测试集类型
    """
    with open(path2 + f'/acc_{type}.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(iter_staid, iter_endid):
            path3 = f'{path2}/iter{i}/result/train.csv'
            acc_1w = test_1wan_easy_acc(path1, path3)
            writer.writerow(["iter" + str(i), acc_1w])


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


def trans_formula_INT(env, formula):
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


def return_oeis_pre_n(oeis_dict, seq_name, num):
    """
    返回OEIS中对应序列的前n项
    """
    # with open('/home/skm21/Find_OEIS/model/oeis_dict.json', 'r', encoding='utf-8-sig') as f:
    #     dic=json.load(f)
    # print(dic['A000005'])
    seq = oeis_dict[seq_name]
    seq_pre_num = ','.join([str(term) for term in seq.split(',')[:num]])
    return seq_pre_num


if __name__ == '__main__':
    path = '/home/skm21/fairseq-0.12.0/checkpoints/model0_32/iter51/result/'

    # readom_select_train_data('/home/skm21/fairseq-0.12.0/checkpoints/model0/iter50/result/',0.5)
    # obtain_csv_res1('/home/skm21/fairseq-0.12.0/checkpoints/model0/iter50/','/home/skm21/fairseq-0.12.0/checkpoints/model0/iter50/result/',32,1)

    # test_1wan_easy_acc(35,'/home/skm21/fairseq-0.12.0/checkpoints/model0/iter52/result/')
    # test_findlinearrecurrence()
    # test_baseline_pachong()

    # find_hard_more_formula('/home/skm21/fairseq-0.12.0/process_result/oeis_data_all_keyword.csv',path)
    # select_9k_test_set('/home/skm21/fairseq-0.12.0/data_oeis/wolfram_test/',25)
    # test_threeDataSet_acc("/home/skm21/fairseq-0.12.0_213/data_oeis/1wan_easy_testdata_35.csv",'/home/skm21/fairseq-0.12.0_213/checkpoints/model0_64auto',"easy",1,7)

    formula = 'add mul idiv sqr x_0_1 x_0_2 2 x_0_1'
    # nums=formula_symbolic_nums(formula)
    # print(nums)

    keywords_path = '/home/skm21/fairseq-0.12.0_212/data_oeis/oeis_data_all_keyword.csv'
    path2 = '/home/skm21/fairseq-0.12.0_212/checkpoints/4500w_SL_all_initData/iter31/result/'
    save_path = '/home/skm21/fairseq-0.12.0_212/checkpoints/4500w_SL_all_initData/iter31/result/find_hard_more.csv'
    nums = find_hard_more_formula(keywords_path, path2)
    print("发现hard,more公式数量：", nums)
