import csv


def obtain_csv_res(res_path, csv_path, beam_nbest_size):  # 把束搜索的结果转化为csv结构,
    with open(res_path, 'r', encoding='utf-8') as f1:
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f2:
            if beam_nbest_size > 31:
                beam_nbest_size = 31
            lines = f1.readlines()
            writer = csv.writer(f2)
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


if __name__ == '__main__':
    obtain_csv_res('/home/skm21/fairseq-0.12.0/checkpoints/model0/iter2/result3/generate-test.txt',
                   '/home/skm21/fairseq-0.12.0/checkpoints/model0/iter2/result3/generate-test.csv', 32)
    # obtain_csv_res('/home/skm21/fairseq-0.12.0/checkpoints/model0/iter2.6/result/generate-test.txt','/home/skm21/fairseq-0.12.0/checkpoints/model0/iter2.6/result/generate-test.csv',32)
    # str='H-266	-0.984432578086853	add idiv n x_0_1 mul INT+ 7 add x_0_1 x_0_1'
    # str=str.split()[2:]
    # print(str)
