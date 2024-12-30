import os


def create_no_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


create_no_path("/home/skm21/fairseq-0.12.0/checkpoints/iter2/result")
