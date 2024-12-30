# class Node():
#     # 初始化一个节点
#     def __init__(self, val=None, name=None):
#         self.val = val  # 节点值
#         # self.l_child = []    # 子节点列表
#         self.l_child = {}  # 子节点字典
#         self.name = name  # 该节点属于的oeis序列号  在分叉后 再给该变量赋值
#
#     # 添加子节点
#     def add_child(self, node):
#         # self.l_child.append(node)
#         self.l_child[node.val] = node
#
class Node():
    # 初始化一个节点
    def __init__(self, val=None, flag=0):
        self.val = val  # 节点值
        # self.l_child = []    # 子节点列表
        self.l_child = {}  # 子节点字典
        self.name = []  # 该节点属于的oeis序列号  在分叉后 再给该变量赋值
        self.flag = flag  # 这个节点是否为序列的最后一个节点 如果是则设置为1

    # 添加子节点
    def add_child(self, node):
        # self.l_child.append(node)
        self.l_child[node.val] = node

    def add_seq_name(self, seq_name):
        if seq_name not in self.name:
            self.name.append(seq_name)
