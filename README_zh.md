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
## 1. 不同方法在三种测试集上的acc
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

## 2. 发现的公式
### （1）发现的一些有趣的OEIS数列公式
| OEIS 序号 | 本文发现的公式 | 数列部分项 |
|-----------|----------------|------------|
| A036146   | $ a_n = (2a_{n-1}) \% 139 $ | 1, 2, 4, 8, 16, 32, 64, 128, 117, 95, 51, 102 |
| A000037   | $ a_n = 2n - n^2 // (a_{n-1} + 2) $ | 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18 |
| A214993   | $ a_n = a_{n-1}^2 // a_{n-2} + 10 $ | 11, 121, 1341, 14871, 164921, 1829001 |
| A080206   | $ a_n = abs(a_{n-2} + a_{n-3}) - 2a_{n-1} $ | 1, 1, 1, 0, 2, -3, 8, -15, 35, -63, 146, -264 |
| A070602   | $ \begin{cases} a_n = (n-1)^5 \% 18 \\ a_n = (a_{n-9} + 9) \% 18 \end{cases} $ | 0, 1, 14, 9, 16, 11, 0, 13, 8, 9, 10, 5, 0, 7 |
| A147657   | $ a_n = \text{sign}(a_{n-3})(n \% (-2)) - a_{n-2} $ | 1, 2, 1, -2, -2, 2, 3, -2, -4, 2, 5, -2, -6, 2, 7 |
| A301697   | $ a_n = a_{n-1} + a_{n-6} - a_{n-7}, n \geq 9 $ | 1, 5, 10, 16, 22, 27, 32, 37, 42, 48, 54, 59 |

### （2）发现与OEIS中不同的数列公式
| OEIS 序号 | 本文发现的公式 | OEIS 原公式 |
|-----------|----------------|-------------|
| A212342   | $ a_n = (-2) // (a_{n-1} + a_{n-1} + n) $ | $ a_n = 3a_{n-1} - 3a_{n-2} + a_{n-3}, n \geq 6 $ |
| A006257   | $ a_n = 1 + (a_{n-1} + 1) \% (n - 1) $ | $ a_{2n-1} = 2a_n - 1, a_{2n} = 2a_n + 1, n \geq 2 $ |
| A350520   | $ a_n = 2a_{n-2} + a_{n-1} + a_{n-1} \% (4a_{n-3}) $ | $ a_{2n-1} = 4^{n-1} - 2^{n-2}, a_{2n} = 2^{2n-1}, n \geq 2 $ |
| A099197   | $ a_n = a_{n-2} + (20a_{n-1}) // (n - 2) $ | $ a_n = n^2(2n^8 + 120n^6 + 1806n^4 + 7180n^2 + 5067) / 14175 $ |
| A273331   | $ a_n = 1 + 4(a_{n-1} + 1) \% a_{n-3} $ | $ a_n = (53 * 4^{n-3} - 5) / 3, n \geq 4 $ |
| A264041   | $ a_n = n^2 - a_{n-1} + (n + 1) // 6 $ | $ a_n = a_{n-1} + a_{n-2} - a_{n-3} + a_{n-6} - a_{n-7} - a_{n-8} + a_{n-9} $ |


### （3）新发现的OEIS数列公式
| OEIS 序号 | 数列描述 | 本文发现的新公式 |
|-----------|----------|------------------|
| A134342   | 某自适应自动机接受的输入组成的数列 | $ a_n = a_{n-1} + (a_{n-1} + 4) // 2 $ |
| A352178   | 集合 $ S $ 中含有任意 $ n $ 个不同的整数，$ a_n $ 是 $ S $ 中两个数之和为 2 的幂的最大数量 | $ a_n = n + (n^2 - 5) // 19 $ |
| A302930   | 在有 $ n $ 个地雷的无限扫雷网格中，可放入 “6” 的最大数量，要求每个 “6” 都正好与 6 个地雷相邻 | $ a_n = (a_{n-1} + (n - 1)^2) // 4n $ |
| A213685   | 最小尺寸的最大反链 | $ a_n = (3n - 2) // 2 + a_{n-4} + n $ |
| A229803   | 三角形六边形单元格棋盘上的车图 HR(n) 的支配数，车可以沿着相邻单元格的任何一行移动 | $ a_n = (6n + 9) // 13 $ |
| A157795   | 与有关密度 Hales-Jewett 定理的超乐观猜想有关 | $ a_n = n + (n^2 + 1 - n \% 5) // 6 $ |

# Quick start

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
# gradio_web
系统说明见 docs/OEIS整数数列公式发现系统说明书.docx
```angular2html
python system_main.py
```

