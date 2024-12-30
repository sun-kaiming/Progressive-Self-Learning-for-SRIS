def decode(lst):
    if len(lst) == 0:
        return None
    res = []
    if lst[0] in ["+", "-"]:
        curr_group = [lst[0]]
    else:
        return None
    if lst[-1] in ["+", "-"]:
        return None

    for x in lst[1:]:
        if x in ["+", "-"]:
            if len(curr_group) > 1:
                sign = 1 if curr_group[0] == "+" else -1
                value = 0
                for elem in curr_group[1:]:
                    value = value * 10000 + int(elem)
                res.append(sign * value)
                curr_group = [x]
            else:
                return None
        else:
            curr_group.append(x)
    if len(curr_group) > 1:
        sign = 1 if curr_group[0] == "+" else -1
        value = 0
        for elem in curr_group[1:]:
            value = value * 10000 + int(elem)
        res.append(sign * value)
    return res
