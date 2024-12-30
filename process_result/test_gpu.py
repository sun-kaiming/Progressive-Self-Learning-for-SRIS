import subprocess
import re

def get_free_gpu_ids(threshold=100):
    """
    获取显存占用小于指定阈值（默认100MB）的GPU ID列表。
    
    :param threshold: 显存占用阈值（单位：MB）
    :return: 符合条件的GPU ID列表
    """
    try:
        # 使用 nvidia-smi 查询显存使用情况
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 检查命令是否成功执行
        if result.returncode != 0:
            print(f"Error executing nvidia-smi: {result.stderr}")
            return []

        # 解析 nvidia-smi 输出
        lines = result.stdout.strip().split('\n')
        free_gpus = []
        
        for line in lines:
            parts = line.split(', ')
            if len(parts) == 2:
                index, used_memory = parts
                used_memory = int(used_memory)
                if used_memory < threshold:
                    free_gpus.append(int(index))
        
        return free_gpus

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    free_gpu_ids = get_free_gpu_ids()
    print(f"GPUs with less than 100MB memory usage: {free_gpu_ids}")