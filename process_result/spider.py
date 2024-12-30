import requests
import json
import csv
import tqdm
from multiprocessing import Pool

# 图片来自bing.com
url = 'https://oeis.org/search?q=id:A000001&fmt=json'


def requests_download(args):  # 存贮json数据
    with open("oeis_data_all_keyword.csv", 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        # print(args)
        url, name = args
        dict_data = requests.get(url, allow_redirects=True).json()

        if 'keyword' in dict_data['results'][0]:
            # if 'easy' in dict_data['results'][0]['keyword']:
            save_main_data(writer, dict_data['query'][3:], dict_data['results'][0]['data'],
                           dict_data['results'][0]['keyword'])
            # count+=1
        # else:
        #     save_main_data(writer, dict_data['query'][3:], dict_data['results'][0]['data'], '')


def save_main_data(writer, seq_nam, seq_data, seq_formula):  # 存储序列关键数据为csv格式
    writer.writerow([seq_nam, seq_data, seq_formula])


def obtain_json_data():  # 循环获取oeis的所有序列的json数据
    # with open("main_oeis_data_1wan_easy2.csv", 'a', encoding='utf-8-sig', newline='') as f:
    #     writer = csv.writer(f)
    #     header = ['seq_name', 'seq_data', 'seq_formula']
    #     writer.writerow(header)
    count = 0
    pool = Pool(40)  # 定义一个进程池，最大进程数48
    for i in range(1, 361990):
        if i % 1000 == 0:
            print(i)
        name = get_six_name(str(i))
        url = 'https://oeis.org/search?q=id:' + name + '&fmt=json'
        # dict_data = requests.get(url, allow_redirects=True).json()
        args = (url, name)
        # print(args)
        # exit()
        pool.apply_async(requests_download, (args,))
        # print(args)
        # exit()
    pool.close()
    # 等待po中所有子进程执行完成，必须放在close语句之后
    pool.join()


def get_six_name(id):  # 补全序号：234=>A000234
    name = id
    if len(name) < 6:
        name = "0" * (6 - len(name)) + name
    return "A" + name


if __name__ == "__main__":
    # requests_download()
    obtain_json_data()
