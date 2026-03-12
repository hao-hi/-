"""
Wahba问题求解模块
用于从矢量观测确定姿态
"""

import numpy as np


def wahba_attitude(obs_cam, ref_inertial, w=None):
    """
    求解Wahba问题：从观测矢量确定姿态旋转矩阵
    
    参数:
        obs_cam: 相机坐标系中的观测矢量 (N×3)
        ref_inertial: 惯性坐标系中的参考矢量 (N×3)
        w: 权重向量 (可选，默认全1)
    
    返回:
        R: 从惯性系到相机系的旋转矩阵
    """
    assert obs_cam.shape == ref_inertial.shape
    
    if w is None:
        w = np.ones(obs_cam.shape[0])
    
    # 直接按行加权，避免构造 N×N 对角矩阵
    weighted_obs = obs_cam * np.asarray(w, dtype=float)[:, None]
    B = weighted_obs.T @ ref_inertial
    
    # SVD分解
    U, S, Vt = np.linalg.svd(B)
    
    # 确保旋转矩阵行列式为+1
    M = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        M[2, 2] = -1
    
    R = U @ M @ Vt
    return R
