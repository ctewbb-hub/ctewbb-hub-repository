"""Desc :
为整个工程提供统一的绝对路径
"""
import os

def get_project_root() -> str:
    """
    获取工程所在的根目录,方便后续路径拼接
    :return: str根目录
    """
    # 当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取工程的根目录,先获取文件所在的文件夹绝对路径
    current_dir = os.path.dirname(current_file)
    # 再获取当前文件所在的文件夹绝对路径,即本次工程的根目录
    project_root = os.path.dirname(current_dir)
    return project_root

def get_abs_path(relative_path: str) -> str:
    """
    传入相对路径,返回绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    return os.path.join(get_project_root(), relative_path)

# if __name__ == '__main__':
#     print(get_abs_path(__file__))
