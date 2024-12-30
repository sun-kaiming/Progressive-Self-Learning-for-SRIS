#!/usr/bin/env wolframscript
formula=$ScriptCommandLine[[2]];
seqinit=$ScriptCommandLine[[3]];
len=$ScriptCommandLine[[4]];

formulalist=StringSplit[formula, ","]; #把字符串序列转化为字符串列表
formulalistint=ToExpression[formulalist];# 把字符串列表转化为转化为整型

seqinitlist=StringSplit[seqinit, ","];
seqinitlistint=ToExpression[seqinitlist];


lenint=ToExpression[len]; #把字符串数字转化为整型

Print[LinearRecurrence[formulalistint, seqinitlistint, lenint]]; # 把线性递推式转为序列，可以指定输出的项数
