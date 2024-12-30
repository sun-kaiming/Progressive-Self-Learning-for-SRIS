import csv
import json
def get_data(path,file_type):
    with open(path,'r',encoding='utf-8')as f1:
        with open(f'src-{file_type}.txt','w',encoding='utf-8')as f2:
            with open(f'tgt-{file_type}.txt','w',encoding='utf-8')as f3:
                lines=f1.readlines()
                # print(len(lines))
                # print(lines[124587519])
                # exit()
                # print(lines[0])
                for line in lines:
                    line=json.loads(line)
                    # print(line)
                    # print(type(line))
                    # exit()
                    f2.write(line['x1']+'\n')
                    f3.write(line['x2']+'\n')
    print("数据处理完成！")

def get_max_len(path):
    with open(path, 'r', encoding='utf-8') as f1:
        lines=f1.readlines()
        max_len_x1=0
        max_len_x2=0
        for line in lines:
            line=json.loads(line)
            x1=line['x1'].split(" ")
            x2=line['x2'].split(" ")
            # print(x1)
            # print(x2)
            # exit()

            max_len_x1=max(len(x1),max_len_x1)
            max_len_x2=max(len(x2),max_len_x2)
        print("max_len_x1: ",max_len_x1)
        print("max_len_x2: ",max_len_x2)

# def look_s

def process_matlab_result():
    with open("OEIS/result_test10000_2.csv",'r',encoding='utf-8')as f1:
        with open("OEIS/result_test10000_2_process.csv",'w',encoding='utf-8',newline='')as f2:
            reader=csv.reader(f1)
            header=next(reader)
            writer=csv.writer(f2)
            writer.writerow(header)
            for row in reader:
                print(row)
                list1=row[0].split('_')
                list1=', '.join(list1)
                if row[1]!='falied':
                    list2 = row[1][:-1].split('_')
                    list2 = ', '.join(list2)

                    print(list1)
                    print(list2)
                    writer.writerow([list1,list2,row[2]])
                else:
                    writer.writerow([list1, "failed", ""])
                # exit()



if __name__ == '__main__':
    # get_data('/home/skm21/recur-main/data/5M*16_2/data.prefix','train')
    # get_data('valid/1000.txt','val')
    # get_data('test/10000.txt','test')

    # get_max_len('train/5M.txt')

    # process_matlab_result()
    # get_data('valid/oeis_30000_25.json', 'oeis25')
    # get_data('valid/oeis_30000_35.json', 'oeis35')
    # get_data('test/oeis_all_25.json', 'oeis_all25')
    # get_data('test/oeis_all_7-35.json', 'oeis_all7-35')
    get_data('test/oeis_all_0-1000.json', 'oeis_all0-1000')