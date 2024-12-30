import os
import shutil

def delete_large_files(directory, max_size_mb=50):
    """
    删除指定目录中超过给定大小的文件。

    :param directory: 要检查的目录路径。
    :param max_size_mb: 文件大小阈值（单位：MB）。默认为 50MB。
    """
    # 将 MB 转换为字节
    max_size_bytes = max_size_mb * 1024 * 1024

    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 获取文件大小
                file_size = os.path.getsize(file_path)

                # 如果文件大小超过阈值，则删除文件
                if file_size > max_size_bytes:
                    print(f"Deleting file: {file_path} (Size: {file_size / (1024 * 1024):.2f} MB)")
                    os.remove(file_path)
            except OSError as e:
                print(f"Error accessing file {file_path}: {e}")

if __name__ == "__main__":
    target_directory = "checkpoints/test_1210-13"  # 替换为你要检查的实际目录路径
    delete_large_files(target_directory)
    