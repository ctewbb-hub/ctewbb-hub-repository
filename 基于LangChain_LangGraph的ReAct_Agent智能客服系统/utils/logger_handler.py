"""Desc :
记录代码日志
"""
import os
import logging
from datetime import datetime
from utils.path_tool import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def get_logger(
        name : str = "agent",
        console_level : int = logging.INFO,
        file_level : int = logging.DEBUG,
        log_file = None
) -> logging.Logger :
    # 获取一个logger实例，名字默认叫 "agent"
    logger = logging.getLogger(name)
    # logger全局级别设为 DEBUG（所有等级日志都会被它接收，由handler自己过滤）
    logger.setLevel(logging.DEBUG)

    # 判断：如果这个logger已经存在处理器（handlers）
    if logger.handlers:
        # 直接返回现成logger，不再重复添加handler
        return logger

    # console_Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # file_Handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)
    return logger

logger = get_logger()

if __name__ == "__main__":
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")