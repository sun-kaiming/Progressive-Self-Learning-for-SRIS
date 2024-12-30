def process_oeis_input_seq(oeis_seq, max_len):  # 处理oeis整数序列： 例如 "1,-1,1,-2,2,-4567893"=> "+ 1 - 1 1 - 2 2 - 456 7893"
    # oeis_seq='A000087 ,2,1,2,4,10,37,138,628,2972,14903,76994,409594,-2222628,12281570,-68864086,391120036,2246122574,13025720000,76101450042,449105860008,2666126033850,15925105028685,95664343622234,577651490729530,'

    if len(oeis_seq) > max_len:
        oeis_seq = oeis_seq[:max_len]

    # oeis_seq.reverse() #逆序seq

    oeis_seq2 = []
    for num in oeis_seq:
        if len(num) > 4:
            num = process_big_num(num)

        if '-' not in num:
            oeis_seq2.append('+')
            oeis_seq2.extend(num.split())
        else:
            oeis_seq2.append('-')
            oeis_seq2.extend(num[1:].split())
    # print(oeis_seq)
    # print(type(oeis_seq[0]))

    # print(oeis_seq2)
    oeis_seq3 = []
    for n in oeis_seq2:
        if len(n) == 4:
            if n == "0000":
                oeis_seq3.append('0')
            elif n[:3] == "000":
                oeis_seq3.append(n[-1])
            elif n[:2] == "00":
                oeis_seq3.append(n[2:])
            elif n[:1] == "0":
                oeis_seq3.append(n[1:])
            else:
                oeis_seq3.append(n)
        else:
            oeis_seq3.append(n)
    # print(oeis_seq3)
    # exit()
    return ' '.join(oeis_seq3)


def process_big_num(big_num):  # 处理大数，把大数分割为小于1000的数，例如：12345321=> 1234 5321
    count = 0
    big_num_len = len(big_num)
    big_num_split = big_num
    inster_index = []  # 在大数中插入空格下标
    for i in range(1, big_num_len):
        j = 4 * i
        k = big_num_len - j
        if k < 0:
            break
        inster_index.append(k)
    # print(inster_index)
    for id in inster_index:
        big_num_split = str_insert(big_num_split, id, ' ')
    big_num_split = big_num_split.strip()

    # print("bignum:",big_num_split)
    return big_num_split


# 在字符串指定位置插入字符
# str_origin：源字符串  pos：插入位置  str_add：待插入的字符串
#
def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)  # 字符串转list
    str_list.insert(pos, str_add)  # 在指定位置插入字符串
    str_out = ''.join(str_list)  # 空字符连接
    return str_out
