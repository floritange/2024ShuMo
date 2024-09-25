import os
import logging
from dotenv import dotenv_values
import psutil
import time, requests
from sklearn.preprocessing import MinMaxScaler

PROJECTNAME = "2024ShuMo"


def get_project_root(project_name=PROJECTNAME):
    """
    获取项目的根目录路径。
    """
    # 当前文件的绝对路径
    current_script_path = os.path.abspath(__file__)

    # 查找项目根目录
    project_root = current_script_path
    while os.path.basename(project_root) != project_name:
        project_root = os.path.dirname(project_root)
        if project_root == os.path.dirname(project_root):
            raise Exception(f"项目根目录 {project_name} 未找到。请确保脚本在项目目录内运行。")

    return project_root


def get_absolute_path(relative_path):
    """
    根据传入的相对路径，返回以项目根目录为基础的绝对路径。

    参数:
        relative_path (str): 相对于项目根目录的路径。

    返回:
        str: 绝对路径。
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


def setup_logging(project_name=PROJECTNAME):
    """
    配置全局日志记录器，输出到控制台和日志文件。
    """
    # 创建日志记录器
    logger = logging.getLogger(project_name)
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台日志级别

    # 创建文件处理器
    log_file_path = os.path.join(get_project_root(), "logs", "project_running.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)  # 文件日志级别

    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 配置全局日志记录器
logger = setup_logging()

# 读取 .env 文件并设置全局 CONFIG 变量
CONFIG = dotenv_values(get_absolute_path(".env"))


def wait_for_health_check(url, headers=None, timeout=60):
    """等待服务器健康检查通过."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    return False


def get_process_pid(command_keywords_list):
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info["cmdline"]
        if cmdline:  # 检查 cmdline 是否非空
            cmdline_str = " ".join(cmdline)  # 将命令行参数列表连接成字符串
            # 检查命令行是否包含所有的关键字
            if all(keyword in cmdline_str for keyword in command_keywords_list):
                # 检查是否直接运行 python，而不是通过 /bin/sh -c 调用
                if "/bin/sh" not in cmdline_str:
                    return proc.info["pid"]
    return None  # 如果没有匹配的进程，返回 None


def get_files_by_keywords(directory, keywords):
    # 遍历指定文件夹，根据关键字数组提取文件。
    matched_files = []
    for filename in os.listdir(directory):
        if all(keyword in filename for keyword in keywords):
            matched_files.append(filename)
    return matched_files


def scaler_data(data_to_normalize, columns_to_normalize=["eirp", "seq_time"]):
    # 归一化数值列
    scaler = MinMaxScaler()
    data_scaled = data_to_normalize.copy()
    for column in columns_to_normalize:
        data_scaled[column] = data_scaled[column].astype(float)
    data_scaled.loc[:, columns_to_normalize] = scaler.fit_transform(data_scaled.loc[:, columns_to_normalize])
    return data_scaled
