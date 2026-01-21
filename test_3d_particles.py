#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试3D粒子生成的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from run_simulation import SlopeModelBuilder

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def test_3d_particle_generation():
    """测试3D粒子生成"""
    print("开始测试3D粒子生成...")
    
    # 创建边坡模型构建器
    builder = SlopeModelBuilder()
    
    # 构建参数化边坡模型
    particles = builder.build_parametric_slope(
        slope_height=10.0,
        slope_angle=45.0,
        slope_width=20.0,
        ground_depth=5.0,
        particle_radius=0.5,  # 使用较大的粒子半径以减少粒子数量
        particle_type='soil'
    )
    
    # 检查粒子数量
    print(f"生成的粒子数量: {len(particles)}")
    
    # 检查粒子是否具有3D坐标
    if particles:
        # 获取所有粒子的位置
        positions = np.array([p['position'] for p in particles])
        
        # 检查坐标维度
        print(f"粒子坐标维度: {positions.shape}")
        
        # 检查Z坐标是否有变化
        z_values = positions[:, 2]
        min_z = np.min(z_values)
        max_z = np.max(z_values)
        print(f"Z坐标范围: {min_z:.2f} 到 {max_z:.2f}")
        print(f"Z坐标标准差: {np.std(z_values):.4f}")
        
        if np.std(z_values) > 0.01:
            print("✓ 粒子具有3D分布，Z坐标存在变化")
        else:
            print("✗ 粒子Z坐标几乎没有变化，仍然是平面分布")
        
        # 绘制3D散点图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制粒子
        ax.scatter(
            positions[:, 0], 
            positions[:, 1], 
            positions[:, 2],
            s=20,
            alpha=0.7,
            c='brown',
            marker='o'
        )
        
        ax.set_title('边坡模型3D粒子分布测试')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # 设置坐标轴范围
        ax.set_xlim([-15, 15])
        ax.set_ylim([-5, 15])
        ax.set_zlim([-5, 5])
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig('3d_particle_test.png', dpi=300, bbox_inches='tight')
        print("3D粒子分布图已保存为: 3d_particle_test.png")
        
        # 显示图像（如果在交互环境中）
        # plt.show()
        plt.close()
    
    print("3D粒子生成测试完成!")

if __name__ == "__main__":
    test_3d_particle_generation()
