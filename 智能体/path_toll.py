import os

def get_project_root() -> str:
    current_file = os.path.abspath(__file__)
    # 获取当前文件所在的目录：D:\PythonProject3\智能体
    current_dir = os.path.dirname(current_file)
    # 获取当前文件所在目录的父目录，即项目根目录：D:\PythonProject3
    project_root = os.path.dirname(current_dir)
    # 返回项目根目录的绝对路径
    return project_root
def get_abs_path(relative_path:str)->str:
    project_root = get_project_root()# 获取项目根目录的绝对路径
    return os.path.join(project_root, relative_path)# 返回绝对路径

if __name__ == "__main__":
    print(get_abs_path("智能体/prompt/main_prompt.txt"))

