"""
星敏感器模块
模拟星敏感器的观测过程
"""

import numpy as np
from core_utils import rand_unit, R_to_quat


def wahba_attitude(obs_cam, ref_inertial, w=None):
    """
    求解 Wahba 问题：从观测矢量确定姿态旋转矩阵。
    """
    assert obs_cam.shape == ref_inertial.shape

    if w is None:
        w = np.ones(obs_cam.shape[0])

    weighted_obs = obs_cam * np.asarray(w, dtype=float)[:, None]
    B = weighted_obs.T @ ref_inertial

    U, _, Vt = np.linalg.svd(B)

    M = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        M[2, 2] = -1

    return U @ M @ Vt


class StarTracker:
    """星敏感器模型"""
    
    def __init__(self, n_stars_catalog=2000, fov_deg=20, dir_noise_std=1e-3, rng=None):
        """
        初始化星敏感器
        
        参数:
            n_stars_catalog: 星表大小
            fov_deg: 视场角（度）
            dir_noise_std: 方向测量噪声标准差
            rng: 随机数发生器，默认使用 numpy 全局随机源
        """
        self.rng = np.random if rng is None else rng
        # 生成随机星表（单位矢量）
        self.catalog = rand_unit(n_stars_catalog, rng=self.rng)
        self.fov = np.deg2rad(fov_deg)
        self.cos_fov = np.cos(self.fov)
        self.noise = dir_noise_std
    
    def observe(self, R_inertial_to_cam):
        """
        观测过程：根据当前姿态确定可见恒星并返回测量姿态
        
        参数:
            R_inertial_to_cam: 从惯性系到相机系的旋转矩阵
        
        返回:
            q_meas: 测量得到的姿态四元数，如果可见星数不足则返回None
        """
        # 先在惯性系中筛选：只计算相机光轴与星表的点乘，避免全表旋转和 arccos。
        boresight_inertial = R_inertial_to_cam.T[:, 2]
        vis = self.catalog @ boresight_inertial > self.cos_fov

        visible_catalog = self.catalog[vis]

        # 如果可见星数不足，返回None
        if visible_catalog.shape[0] < 8:
            return None

        # 只把入视场的恒星转换到相机坐标系
        V = visible_catalog @ R_inertial_to_cam.T
        
        # 添加测量噪声
        Vn = V + self.noise * self.rng.randn(*V.shape)
        Vn = Vn / np.linalg.norm(Vn, axis=1, keepdims=True)
        
        # 选择最亮的20颗星（按z坐标排序，z越大越接近光轴中心）
        top_n = min(20, Vn.shape[0])
        idx = np.argpartition(-Vn[:, 2], top_n - 1)[:top_n]
        idx = idx[np.argsort(-Vn[idx, 2])]
        obs = Vn[idx]
        ref = visible_catalog[idx]
        
        # 使用Wahba算法确定姿态
        R_meas = wahba_attitude(obs, ref)
        q_meas = R_to_quat(R_meas)
        
        return q_meas
