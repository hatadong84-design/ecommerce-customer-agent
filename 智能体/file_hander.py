import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

import os
import hashlib
from 智能体.logger_hander import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def get_file_md5_hex(filepath: str):# 获取文件md5
    if not os.path.exists(filepath):# 文件不存在
        logger.error(f"[md5计算]文件不存在: {filepath}")#
        return# 文件不存在返回

    if not os.path.isfile(filepath):# 路径不存在或不是文件
        logger.error(f"[md5计算]路径不存在或不是文件: {filepath}")#
        return# 路径不存在或不是文件返回
    md5 = hashlib.md5()# 创建md5对象

    chunk_size = 8192# 分块大小
    try:
        with open(filepath, 'rb') as f:#
            while True:
                chunk = f.read(chunk_size)# 读取分块
                if not chunk:
                    break
                md5.update(chunk)# 更新md5对象
        return md5.hexdigest()# 返回md5值
    except Exception as e:
        logger.error(f"[md5计算]文件读取错误: {filepath}, 错误信息: {e}")#
        return# 文件读取错误返回


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")#
        return ()# 文件夹不存在返回空元组
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
