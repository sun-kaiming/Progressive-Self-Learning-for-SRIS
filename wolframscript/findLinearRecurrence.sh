#!/usr/bin/env wolframscript


seq=$ScriptCommandLine[[2]];

seqlist=StringSplit[seq, ","]; #把字符串序列转化为字符串列表
seqlistint=ToExpression[seqlist];# 把字符串列表转化为转化为整型
formula=FindLinearRecurrence[seqlistint]; #把序列转化为线性递推表达式

Print[formula]