import csv,sys
import json
sys.path.append("/home/skm21/OEIS_Sequence_Formula_discovery-main")
from recur.utils.decode import decode
from untils.tool import split_test_data

# with open("/home/skm21/fairseq-0.12.0/data_oeis/oeis_data_all_keyword.csv",'r',encoding='utf-8')as f:
#     with open("/home/skm21/fairseq-0.12.0/data_oeis/oeis_data_all_keyword2.csv",'w',encoding='utf-8')as f2:
#         reader=csv.reader(f)
#         writer=csv.writer(f2)
#         for row in reader:
#             row[1]=''
#             writer.writerow(row)


# with open('/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter30/result/final_res.json', 'r',
#           encoding='utf-8') as f:
#     dic = json.load(f)
#     print(len(dic))

def random_data_to_csv_format(randam_data_path,csv_data_path): ##将随机json格式的数据转换成csv格式的数据
    
    with open(randam_data_path,'r')as fr,open(csv_data_path,'w',encoding='utf-8-sig',newline='')as fw:
      writer=csv.writer(fw)
      for line in fr.readlines():
         line=json.loads(line)
         seq=line['x1']
         print(seq)
         seq2=','.join([str(num) for num in decode(seq.split())])
         print(seq2)
         writer.writerow(['',seq2,''])

if __name__=='__main__':
    randam_data_path='/home/skm21/OEIS_Sequence_Formula_discovery-main/data_oeis/random_data/random_1w_testData_new.json'
    csv_data_path='/home/skm21/OEIS_Sequence_Formula_discovery-main/data_oeis/random_1w_testData_new.csv'
    random_data_to_csv_format(randam_data_path,csv_data_path)
          
    path='/home/skm21/OEIS_Sequence_Formula_discovery-main/data_oeis/random_1w_testData_new.csv'
    num=1
    split_seq_path='/home/skm21/OEIS_Sequence_Formula_discovery-main/data_oeis/random_data'
    small_oeis_testset="False"
    split_test_data(path, num, split_seq_path, small_oeis_testset)

      
         
    