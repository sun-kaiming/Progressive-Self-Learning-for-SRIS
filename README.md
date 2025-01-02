#  Citing Progressive-Self-Learning-for-SRIS

```angular2html
@article{
    anonymous2024progressive,
    title={Progressive Self-Learning for Domain Adaptation on Symbolic      Regression of Integer Sequences},
    author={Yaohui Zhu, Kaiming Sun, Zhengdong Luo, Lingfeng Wang},
    booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
    year={2024},
    url={https://openreview.net/forum?id=MDjQNDuYch}
}
```
# Main Results
不同方法在三种测试集上的acc
| methods       | pred=1  | pred=10 | pred=all | pred=1  | pred=10 | pred=all | pred=1  | pred=10 | pred=all |
|------------|---------|---------|----------|---------|---------|----------|---------|---------|----------|
|            | Easy    |         |          | Sign    |         |          | Base    |         |          |
| FSF        | 14.14%  | 13.9%   | 13.63%   | 7.97%   | 6.98%   | 6.44%    | 1.52%   | 1.34%   | 0.91%    |
| FLR        | 21.88%  | 20.86%  | 19.76%   | 15.15%  | 13.67%  | 12.96%   | 11.21%  | 7.8%    | 5.55%    |
| DSR        | 30.2%   | 20.42%  | 19.16%   | 23.96%  | 12.44%  | 9.26%    | 12.54%  | 3.96%   | 1.73%    |
| SL-A2      | 29.91%  | 22.97%  | 22.86%   | 20.49%  | 9.19%   | 9.00%    | 5.86%   | 1.29%   | 1.11%    |
| SL-A1      | 29.52%  | 24.29%  | 24.23%   | 19.95%  | 10.44%  | 10.4%    | 5.73%   | 1.65%   | 1.51%    |
| SL-All     | 33.54%  | 29.01%  | 28.82%   | 24.56%  | 13.70%  | 13.43%   | 12.19%  | 7.12%   | 6.87%    |
| RSSLA      | 34.37%  | 29.22%  | 28.96%   | 23.7%   | 13.27%  | 13.13%   | 12.02%  | 7.07%   | 6.85%    |
| JTSLS      | 34.31%  | 29.34%  | 29.08%   | 25.21%  | 14.06%  | 13.75%   | 12.12%  | 7.22%   | 6.99%    |
| JTSLA      | 35.22%  | 30.11%  | 29.81%   | 25.09%  | 14.38%  | 13.94%   | 12.75%  | 7.40%   | 7.05%    |

## 1.创建、激活环境

```angular2html
conda create -n psl python=3.8
conda activate psl
```

## 2.安装环境
(1)安装psl环境
```angular2html
git clone https://github.com/sun-kaiming/Progressive-Self-Learning-for-SRIS.git
cd Progressive-Self-Learning-for-SRIS
python script/get-pip.py pip==23.01
pip install --editable ./ -i https://pypi.mirrors.ustc.edu.cn/simple/
```

## 3. 开始迭代训练

```angular2html
sh script/combine_train_scripts.sh
```

## 4. 测试模型

```angular2html
sh  script/eval_script.sh
```
   