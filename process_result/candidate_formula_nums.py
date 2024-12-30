import csv

path = f'/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/'

with open(f'{path}candidate_formula_num.csv', 'w') as f2:
    writer = csv.writer(f2)
    for i in range(1, 51):
        try:
            with open(f'{path}/iter{i}/result/generate-test.csv', 'r') as f:
                lines = csv.reader(f)
                dic = {}
                print("")
                for row in lines:
                    formulas = row[1:]
                    # print(len(formulas))
                    for formula in formulas:
                        dic[formula] = 1
                print(f"iter{i}:", len(dic))
                writer.writerow([f"iter{i}", len(dic)])
        except:
            pass
