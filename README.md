## 0.1 安装向日葵

下载一个向日葵，从谷歌或者百度下载，输入链接码，最好能够直接连接我的（孙凯明）的主机

## 0.2 连接向日葵

## 0.3 打开pycharm

## 1.创建conda虚拟环境

### (1) 安装miniconda

### (2) 创建环境

```angular2html
conda create -n oeis python=3.8
```

### (3) 激活环境

```angular2html
conda activate oeis
```

## 2.安装环境

安装fairseq环境

```angular2html
git clone --branch v0.12.0 https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
```

For faster training install NVIDIA's apex library:

```angular2html
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
--global-option="--deprecated_fused_adam" --global-option="--xentropy" \
--global-option="--fast_multihead_attn" ./
```

## 3. 开始迭代训练

```angular2html
sh script/combine_train_scripts.sh
```

## 4. 测试模型

```angular2html
sh  script/eval_script.sh
```
   