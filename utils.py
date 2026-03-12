"""
工具函数模块
导入核心工具函数，避免重复代码
"""

from core_utils import *


def rand_unit(n):
    """生成n个随机单位向量"""
    v = np.random.randn(n, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


