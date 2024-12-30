#!/usr/bin/env wolframscript
seq=$ScriptCommandLine[[2]];
seqinit=$ScriptCommandLine[[3]];
len=$ScriptCommandLine[[4]];

seqlist=StringSplit[seq, ","]; #把字符串序列转化为字符串列表
seqlistint=ToExpression[seqlist];# 把字符串列表转化为转化为整型

seqinitlist=StringSplit[seqinit, ","];
seqinitlistint=ToExpression[seqinitlist];


lenint=ToExpression[len]; #把字符串数字转化为整型

formula=FindLinearRecurrence[seqlistint]; #把序列转化为线性递推表达式

Print[formula]
Print[LinearRecurrence[formula, seqinitlistint, lenint]]; # 把线性递推式转为序列，可以指定输出的项数

