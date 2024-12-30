import gradio as gr
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import time
from testOEIS_Interface2 import return_oeis_formula
# import queue,re
# import subprocess
from eval_test_acc import eval_test
import pandas as pd
# 创建一个队列来存储训练日志，最多保存1000个值
# log_queue = queue.Queue(maxsize=1000)

import subprocess
import re

def get_free_gpu_ids(threshold=100):
    """
    获取显存占用小于指定阈值（默认100MB）的GPU ID列表。
    
    :param threshold: 显存占用阈值（单位：MB）
    :return: 符合条件的GPU ID列表
    """
    try:
        # 使用 nvidia-smi 查询显存使用情况
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 检查命令是否成功执行
        if result.returncode != 0:
            print(f"Error executing nvidia-smi: {result.stderr}")
            return []

        # 解析 nvidia-smi 输出
        lines = result.stdout.strip().split('\n')
        free_gpus = []
        
        for line in lines:
            parts = line.split(', ')
            if len(parts) == 2:
                index, used_memory = parts
                used_memory = int(used_memory)
                if used_memory < threshold:
                    free_gpus.append(int(index))
        
        return free_gpus

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
def image_id():
    time_now = time.time()
    id = int(round(time_now * 10000))
    return id


def formula_image(real_seq, pred_seq, formula, error_rate, id):
    # print("pred_seq:",pred_seq)
    x = []
    for i in range(len(real_seq)):
        x.append(i + 1)
        real_seq[i] = int(real_seq[i])

    pred_seq = pred_seq[:len(real_seq)]
    x = np.array(x)
    real_seq = np.array(real_seq)
    pred_seq = np.array(pred_seq)

    # print("x:", x)
    # print("real_seq:", real_seq)
    # print("pred_seq:", pred_seq)

    x_new = np.linspace(x.min(), x.max(), 80)  # 300 represents number of points to make between T.min and T.max
    # print("x_new:",x_new)
    y_smooth = make_interp_spline(x, real_seq)(x_new)
    # 散点图
    plt.scatter(x, pred_seq, c='red', s=65)  # alpha:透明度) c:颜色
    # 折线图
    # plt.plot(x, y, linewidth=1)  # 线宽linewidth=1
    # 平滑后的折线图
    plt.plot(x_new, y_smooth, c='black', linewidth=2.0)

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # plt.title(f"序号为{id}的公式产生的数列图示", fontsize=16)  # 标题及字号
    plt.xlabel("n", fontdict={'family': 'Times New Roman', 'size': 16})  # X轴标题及字号
    plt.ylabel("a(n)", fontdict={'family': 'Times New Roman', 'size': 16})  # Y轴标题及字号
    plt.tick_params(axis='both', labelsize=14)  # 刻度大小
    # plt.xlim(x_x)
    # plt.xticks(x)
    # plt.axis([0, 1100, 1, 1100000])#设置坐标轴的取值范围
    image_dic = "image_save"
    if not os.path.exists(image_dic):
        os.makedirs(image_dic)
    image_path = f"{image_dic}/{image_id()}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    # plt.savefig(image_path, dpi=300)
    # plt.show()
    plt.close()
    return image_path


def formula_images_list(real_seq, formula_dic):
    show_image_lis = []
    for error_rate, formula_lis in formula_dic.items():
        id, formula, pred_seq = formula_lis
        image_path = formula_image(real_seq, pred_seq, formula, error_rate, id)
        show_image_lis.append(image_path)
    return show_image_lis


def delete_large_files(directory, max_size_mb=50):
    """
    删除指定目录中超过给定大小的文件。

    :param directory: 要检查的目录路径。
    :param max_size_mb: 文件大小阈值（单位：MB）。默认为 50MB。
    """
    # 将 MB 转换为字节
    max_size_bytes = max_size_mb * 1024 * 1024

    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 获取文件大小
                file_size = os.path.getsize(file_path)

                # 如果文件大小超过阈值，则删除文件
                if file_size > max_size_bytes:
                    print(f"Deleting file: {file_path} (Size: {file_size / (1024 * 1024):.2f} MB)")
                    os.remove(file_path)
            except OSError as e:
                print(f"Error accessing file {file_path}: {e}")

def train_method(beam_size, n_best, max_epoch, model_type,is_combine,random_rate_model1,random_rate_model2,small_oeis_testset,sta_iter_id,end_iter_id):
    gpu_list=get_free_gpu_ids()
    if len(gpu_list)==0:
        print("没有空闲的GPU，无法进行训练")
        return "没有空闲的GPU，无法进行训练"
    else:
        if len(gpu_list)>4:
            gpus_id=','.join([str(id) for id in gpu_list[-4:]])
        train_command=f"python iter_script.py --gpus-id {gpus_id} \
                --beam-size {beam_size} --n-best {n_best} --max-epoch {max_epoch} \
                --model-type {model_type} --is-combine {is_combine} \
                --random-rate-model1 {random_rate_model1} --random-rate-model2 {random_rate_model2} \
                --small-oeis-testset {small_oeis_testset} --sta-iter-id {sta_iter_id} --end-iter-id {end_iter_id}"
        
        os.system(train_command)
        save_best_model_path=f'checkpoints/{model_type}/iter{end_iter_id}/model1/checkpoint_best.pt'
        copy_model_path=f'save/model/{model_type}.pt'
        os.system(f"cp {save_best_model_path} {copy_model_path}")
        directory = f"checkpoints/{model_type}"
        delete_large_files(directory)  ##删除大于50Mb的文件

        print("训练完成")
        return "训练完成"

def test_method(model_name):
    model_path=f'save/model/{model_name}.pt'
    gpu_list=get_free_gpu_ids()
    if len(gpu_list)==0:
        print("没有可用的gpu,无法进行测评")
        return "没有可用的gpu,无法进行测评"
        # eval_test(model_path)
    else:
        eval_test(model_path,gpuid=gpu_list[0])

        print("评测完成")
        return "评测完成"

def get_log(model_type):
    train_log=f"checkpoints/{model_type}/train.log"
    print("train_log:",train_log)
    lines="日志还未刷新。训练前需要加载一个字典文件，需要等待几分钟，再刷新。"
    if os.path.exists(train_log):
        print("2333")
        with open(train_log, 'r', encoding='utf-8')as fr:
            lines_lis =fr.readlines()
            lines_lis=lines_lis[-1000:]
            lines=''.join(lines_lis)


            print("lines:",lines)
    return lines

def get_test_log():
    test_log=f"save/result/test_result.log"
    print("test_log:",test_log)
    lines="请注意，评测一个测试集大约需要10分钟，共有4个测试集，大约需要40分钟。测试日志不会异步刷新，需要点击刷新按钮才可刷新"
    if os.path.exists(test_log):
        print("2333")
        with open(test_log, 'r', encoding='utf-8')as fr:
            lines_lis =fr.readlines()
            lines_lis=lines_lis[-1000:]
            lines=''.join(lines_lis)
            print("lines:",lines)
    return lines

def get_model_list(path="save/model"):
    model_list = []
    for filename in os.listdir(path):
        if filename.endswith('.pt'):
            model_name = filename[:-3]  # 去掉'.pt'后缀
            file_path = os.path.join(path, filename)
            creation_time = os.path.getctime(file_path)
            model_list.append((model_name, creation_time))
    
    # 按照创建时间排序
    model_list.sort(key=lambda x: x[1])
    
    # 提取排序后的模型名称
    sorted_model_names = [model_name for model_name, _ in model_list]
    return sorted_model_names
def update_model_list():
    # 更新模型列表
    new_model_list = get_model_list()
    print("new_model_list:",new_model_list)
    return gr.update(choices=new_model_list)

# 定义一个函数来更新 CSV 文件的数据
# def update_csv_data():
#     return load_csv_data()

# 定义一个函数来读取 CSV 文件并返回数据
# def load_csv_data():
#     csv_file_path = "/home/skm21/OEIS_Sequence_Formula_discovery-main/save/result/test_result.csv"
#     df = pd.read_csv(csv_file_path)
#     return df.values.tolist()

# 定义一个函数来读取 CSV 文件并返回数据
def load_csv_data():
    csv_file_path = "/home/skm21/OEIS_Sequence_Formula_discovery-main/save/result/test_result.csv"
    df = pd.read_csv(csv_file_path)
    
    # 创建一个新的 DataFrame 来存储合并后的数据
    new_data = {
        "model_name": df["model_name"].tolist(),
        "easy_pre_1": df["easy_predict_1"].tolist(),
        "easy_pre_10": df["easy_predict_10"].tolist(),
        "easy_pre_all": df["easy_predict_all"].tolist(),
        "sign_pre_1": df["sign_predict_1"].tolist(),
        "sign_pre_10": df["sign_predict_10"].tolist(),
        "sign_pre_all": df["sign_predict_all"].tolist(),
        "base_pre_1": df["base_predict_1"].tolist(),
        "base_pre_10": df["base_predict_10"].tolist(),
        "base_pre_all": df["base_predict_all"].tolist(),
        "random_pre_1": df["random_predict_1"].tolist(),
        "random_pre_10": df["random_predict_10"].tolist(),
        "random_pre_all": df["random_predict_all"].tolist(),
    }
    
    # 创建一个新的 DataFrame
    new_df = pd.DataFrame(new_data)
    
    # 转换为列表形式
    data = new_df.values.tolist()
    
    # 添加表头和子表头
    headers = [
        ["model_name", "easy", "easy", "easy", "sign", "sign", "sign", "base", "base", "base", "random", "random", "random"],
        ["", "pre_1", "pre_10", "pre_all", "pre_1", "pre_10", "pre_all", "pre_1", "pre_10", "pre_all", "pre_1", "pre_10", "pre_all"]
    ]
    
    # 将表头和数据合并
    data_with_headers = headers + data
    
    return data_with_headers
# 定义一个函数来更新 CSV 文件的数据
def update_csv_data():
    return load_csv_data()
#     return model_list
def inference_method(input_sequence, beam_size, show_formulas_nums, pred_terms_nums,model_name):
    gpu_list=get_free_gpu_ids()
    if len(gpu_list)==0:
        print("没有空闲的GPU,无法进行推理")
        return "没有空闲的GPU,无法进行推理",[]
        # finale_res = return_oeis_formula(input_sequence, beam_size, show_formulas_nums, pred_terms_nums,model_name)
    else:
        finale_res = return_oeis_formula(input_sequence, beam_size, show_formulas_nums, pred_terms_nums,model_name,gpu_list[0])
    input_sequence = input_sequence.replace(" ", "")
    input_sequence = input_sequence.replace("、", ",")
    input_sequence_lis = input_sequence.split(',')
    seq_len = len(input_sequence_lis)
    seq_len = min(25, seq_len)
    print("")
    
    finale_res = finale_res[:show_formulas_nums]

    image_dic = {}

    pred_formulas = "|$$\\textbf{序号}$$| $$\\textbf{候选公式}$$ | $$\\textbf{预测输入数列的后" + str(
        pred_terms_nums) + "项}$$ | $$\\textbf{误差率}$$ | $$\\textbf{OEIS序号}$$ |\n| :----: | :----: | :----: | :----:  |  :----: |\n"

    for i, tuple in enumerate(finale_res):
        if tuple[4] != '--':
            seq_name_link = f"[{tuple[4]}](https://oeis.org/{tuple[4]})"
        else:
            seq_name_link = "--"
        pre_seq_str = ', '.join(str(num) for num in tuple[3][seq_len:])
        pred_formulas = pred_formulas + f"| $${i + 1}$$ |  $$ \large {str(tuple[2])}$$   | <center> <font size=3>{pre_seq_str}</font> | $${tuple[0]}\\\\%$$ | <font size=3> {seq_name_link} </font>  |\n"

        if str(tuple[0]) not in image_dic and len(image_dic) < 5:
            image_dic[str(tuple[0])] = [i + 1, str(tuple[2]), tuple[3]]
    #
    # print("%"*100)
    # print("markDown:", pred_formulas)
    # print("%"*100)

    show_image_lis = formula_images_list(input_sequence_lis, image_dic)
    # show_image_lis = ['D:\PythonWorkspace\\fairseq-main\\test2.png', 'D:\PythonWorkspace\\fairseq-main\\test2.png',
    #                   'D:\PythonWorkspace\\fairseq-main\\test2.png']
    return pred_formulas, show_image_lis


with gr.Blocks() as demo:
    gr.Markdown("""
        # <center>  OEIS整数数列公式发现系统 </center>
    """)
    # 设置输入组件
    with gr.Tab(label="Train"):
        with gr.Row():
            with gr.Column():
                beam_size = gr.Slider(0, 100, value=32, label="束搜索宽度", )
                n_best = gr.Slider(0, 100, value=32, label="每次束搜索所保留的最高概率公式数量")
                max_epoch = gr.Slider(0, 100, value=100, label="训练的epochs")
            with gr.Column():
                model_type = gr.Textbox(label="保存的模型名称", value="test_1201-2")
                is_combine = gr.Checkbox(label="是否采用联合训练", value=False)
                random_rate_model1 = gr.Slider(0.0, 1.0, value=1.0, label="模型1随机选择当前全部OEIS数据的比率")
                random_rate_model2 = gr.Slider(0.0, 1.0, value=0.5, label="模型2随机选择当前全部OEIS数据的比率")
            with gr.Column():
                small_oeis_testset = gr.Checkbox(label="小OEIS测试集", value=False)
                # gpus_id = gr.Textbox(label="GPU ID", value="0,1")
                sta_iter_id = gr.Slider(0, 100, value=1, label="起始迭代ID")
                end_iter_id = gr.Slider(0, 100, value=3, label="结束迭代ID")
        with gr.Row():
            with gr.Column(scale=11):
                train_log = gr.Textbox(label="训练日志",lines=15)
            with gr.Column(scale=1, min_width=1):
                update_log_btn = gr.Button("Update Train Log", scale=1, variant='primary', size='lg')

        train_log_flag = gr.Textbox(label="训练完成标识")
        with gr.Column(scale=4):
            train_btn = gr.Button("Start Train", scale=1, variant='primary', size='lg')
        
        train_btn.click(fn=train_method, inputs=[beam_size, n_best, max_epoch, model_type,is_combine,random_rate_model1,random_rate_model2,small_oeis_testset,sta_iter_id,end_iter_id],
                        outputs=[train_log_flag], )
        # 将按钮点击事件绑定到更新日志的回调函数
        update_log_btn.click(fn=get_log, inputs=[model_type], outputs=[train_log])

        #  添加定时任务来更新日志
        # demo.load(fn=get_log,inputs=[model_type], outputs=[train_log], every=1000)  #
    
    with gr.Tab(label="Test"):
        with gr.Row():
            with gr.Column(scale=9, min_width=1):
                with gr.Row():
                    # model_name= gr.Textbox(label="选择待测试的模型") 
                    model_list=get_model_list()
                    with gr.Column(scale=11):
                        model_name = gr.Dropdown(choices=model_list, label="选择待测试的模型",value="4500w_combine_train_36w_iter50")
                    with gr.Column(scale=1,min_width=4):
                        update_model_list_btn= gr.Button("Update Model List",scale=2, variant='primary', size='lg')    
            with gr.Column(scale=11):
                with gr.Row():
                    with gr.Column(scale=11):
                        test_log = gr.Textbox(label="测试日志",lines=4,value="请注意，评测一个测试集大约需要10分钟，共有4个测试集，大约需要40分钟。测试日志不会异步刷新，需要点击刷新按钮才可刷新")
                    with gr.Column(scale=1, min_width=1):
                        update_test_log_btn = gr.Button("Update Test Log", scale=2, variant='primary', size='lg')
                with gr.Row(): 
                    test_log_flag = gr.Textbox(label="测试完成标识")
        with gr.Row():
            test_btn = gr.Button("Start Test", scale=1, variant='primary', size='lg')
            test_btn.click(fn=test_method, inputs=[model_name],
                                outputs=[test_log_flag], )
            # 将按钮点击事件绑定到更新日志的回调函数
            update_test_log_btn.click(fn=get_test_log, inputs=[], outputs=[test_log])
            update_model_list_btn.click(fn=update_model_list, inputs=[], outputs=[model_name])
        with gr.Row():
            csv_data_df = gr.DataFrame(
                datatype=["str"] * 13, 
                label="CSV Data",
                value=update_csv_data()  # 设置默认值为 update_csv_data 的返回值
            )
        with gr.Row():
            # 添加一个按钮来更新 CSV 数据
            update_csv_btn = gr.Button("Update Test Result", scale=1, variant='primary', size='lg')
            update_csv_btn.click(fn=update_csv_data, outputs=[csv_data_df])

                
    with gr.Tab(label="Inference"):
        with gr.Row():
            with gr.Column(scale=11):
                with gr.Row():
                    # model_name= gr.Textbox(label="选择待测试的模型") 
                    model_list=get_model_list()
                    model_name = gr.Dropdown(choices=model_list,scale=5, label="选择待测试的模型",value="4500w_combine_train_36w_iter50")
                    update_model_list_btn= gr.Button("Update Model List",scale=2, variant='primary', size='lg',min_width=1)   
                with gr.Row():
                    input_sequence = gr.Textbox(label="请输入一条整数数列, 用英文逗号分隔整数, 最佳输入项数为25项",
                                                value='0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610')
                with gr.Row():
                    beam_size = gr.Slider(0, 100, value=32, label="束搜索大小", )
                    show_formulas_nums = gr.Slider(0, 100, value=32, label="候选公式数量")
                    pred_terms_nums = gr.Slider(0, 100, value=10, label="预测的项数")

            with gr.Column(scale=1, min_width=1):
                reset_btn = gr.ClearButton(value='重置', scale=1, size='lg').add([input_sequence])

            with gr.Column(scale=4):
                generate_btn = gr.Button("生成", scale=1, variant='primary', size='lg')
                
        with gr.Row():
            with gr.Column(scale=15):
                formulas_md = gr.Markdown(elem_classes='center')
            with gr.Column(scale=7, min_width=1):
                image_res_lis = gr.Gallery(min_width=1, columns=2)
                # 设置按钮点击事件
        update_model_list_btn.click(fn=update_model_list, inputs=[], outputs=[model_name])
        generate_btn.click(fn=inference_method, inputs=[input_sequence, beam_size, show_formulas_nums, pred_terms_nums,model_name],
                        outputs=[formulas_md, image_res_lis] )

# exit()
demo.launch(server_name='10.1.11.212', server_port=12126)
