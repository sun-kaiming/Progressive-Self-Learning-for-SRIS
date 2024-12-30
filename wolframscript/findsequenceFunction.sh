#!/usr/bin/env wolframscript


seq=$ScriptCommandLine[[2]];
len=$ScriptCommandLine[[3]];

seqlist=StringSplit[seq, ","]; #把字符串序列转化为字符串列表

seqlistint=ToExpression[seqlist];# 把字符串列表转化为转化为整型

lenint=ToExpression[len]; #把字符串数字转化为整型

formula=FindSequenceFunction[seqlistint, n];


Print[formula];

If[Head[formula] === FindSequenceFunction,
    Print[no formula],
    Print[Table[formula, {n, lenint}]]
]

