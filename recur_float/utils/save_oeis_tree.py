import pickle
import sys
from recur.utils.node_class import Node

sys.setrecursionlimit(300000)  # 将默认的递归深度修改为300000


def get_data_list():
    with open('/home/skm21/OpenNMT-py-master/data_recur/stripped', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        oeis_list = []

        for i in range(4, len(lines)):
            dict_temp = {}
            if "A" in lines[i]:
                seq_name = lines[i][:7]
                oeis_seq = lines[i][9:-2].strip().split(',')
                dict_temp[seq_name] = oeis_seq
                oeis_list.append(dict_temp)

    return oeis_list


def isinclude_num(node, num):  # 检测某个节点的孩子节点中是否包含数据num
    child_node_list = node.l_child

    if num in child_node_list:
        return True, child_node_list[num]

    return False, None


def save_tree_file(save_path):  # 保存oeis转化成的树，方便快速加载
    res = ''
    # oeis_list=[[1, 1, 1, 1, 2, 3, 6, 11, 23, 47, 106, 235, 551, 1301, 3159, 7741, 19320, 48629, 123867, 317955, 823065, 2144505, 5623756, 14828074, 39299897, 104636890, 279793450, 751065460, 2023443032, 5469566585, 14830871802, 40330829030, 109972410221, 300628862480, 823779631721, 2262366343746, 6226306037178],[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271]]
    oeis_list = get_data_list()
    root = Node("S")
    for seq_dict in oeis_list:
        temp_node = root
        is_final_term = 0
        for seq_name, seq_list in seq_dict.items():
            for i, term in enumerate(seq_list):
                if i == len(seq_list) - 1:
                    is_final_term = 1
                if temp_node.l_child == {}:

                    new_node = Node(term, is_final_term)
                    if is_final_term == 1:
                        new_node.add_seq_name(seq_name)
                    temp_node.add_child(new_node)
                    temp_node = new_node
                else:

                    flag, return_node = isinclude_num(temp_node, term)
                    if flag:
                        temp_node = return_node
                        if is_final_term == 1:
                            temp_node.add_seq_name(seq_name)
                    else:
                        new_node = Node(term, is_final_term)
                        if is_final_term == 1:
                            new_node.add_seq_name(seq_name)
                        temp_node.add_child(new_node)
                        temp_node = new_node

    tree_str = pickle.dumps(root)
    with open(save_path, 'wb') as f:
        f.write(tree_str)
    print("完成")


if __name__ == '__main__':
    save_path = '/home/skm21/fairseq-0.12.0/tree/oeis_tree_save_dict_seqNames.pkl'
    save_tree_file(save_path)
