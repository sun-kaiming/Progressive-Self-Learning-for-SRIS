import csv

path1 = '/home/skm21/fairseq-0.12.0/data_oeis/All_found_oeis_formuals/4500w_combine_train_9w.csv'
path2 = '/home/skm21/fairseq-0.12.0/data_oeis/All_found_oeis_formuals/4500w_combine_train_36w.csv'
path3 = '/home/skm21/fairseq-0.12.0/data_oeis/All_found_oeis_formuals/4500w_SL_all_initData.csv'
path4 = '/home/skm21/fairseq-0.12.0/data_oeis/All_found_oeis_formuals/model0_32_auto_208.csv'
path5 = '/home/skm21/fairseq-0.12.0/data_oeis/All_found_oeis_formuals/model0_32_newdata2.csv'

with open(path1, 'r') as f1, open(path2, 'r') as f2, open(path3, 'r') as f3, open(path4, 'r') as f4, open(path5,
                                                                                                          'r') as f5:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    reader3 = csv.reader(f3)
    reader4 = csv.reader(f4)
    reader5 = csv.reader(f5)
    dic = {}
    for row in reader1:
        if row[0].strip() not in dic:
            dic[row[0].strip()] = 1
    for row in reader2:
        if row[0].strip() not in dic:
            dic[row[0].strip()] = 1
    for row in reader3:
        if row[0].strip() not in dic:
            dic[row[0].strip()] = 1
    for row in reader4:
        if row[0].strip() not in dic:
            dic[row[0].strip()] = 1
    for row in reader5:
        if row[0].strip() not in dic:
            dic[row[0].strip()] = 1
    print(len(dic))
