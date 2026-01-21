#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试主应用程序中的3D模型预览功能
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所需模块
from run_simulation import SlopeModelBuilder

class TestWindow(QMainWindow):
    """测试窗口类"""
    
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        self.slope_builder = SlopeModelBuilder()
        self.particles = []
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("3D模型预览测试")
        self.setGeometry(100, 100, 800, 600)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 垂直布局
        layout = QVBoxLayout(central_widget)
        
        # 标题标签
        title_label = QLabel("3D模型预览测试")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # 测试按钮
        test_button = QPushButton("生成3D模型并预览")
        test_button.clicked.connect(self.test_3d_preview)
        layout.addWidget(test_button)
        
        # 状态标签
        self.status_label = QLabel("点击按钮开始测试")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: blue;")
        layout.addWidget(self.status_label)
        
        # 统计信息标签
        self.stats_label = QLabel("统计信息将显示在这里")
        self.stats_label.setAlignment(Qt.AlignLeft)
        self.stats_label.setStyleSheet("font-size: 12px; margin-top: 10px;")
        layout.addWidget(self.stats_label)
    
    def test_3d_preview(self):
        """测试3D预览功能"""
        self.status_label.setText("正在生成3D模型...")
        
        # 构建边坡模型
        self.particles = self.slope_builder.build_parametric_slope(
            slope_height=10.0,
            slope_angle=45.0,
            slope_width=20.0,
            ground_depth=5.0,
            particle_radius=0.5,
            particle_type='soil'
        )
        
        # 检查3D分布
        self.check_3d_distribution()
        
        # 尝试创建3D预览
        try:
            self.create_3d_preview()
            self.status_label.setText("3D模型预览创建成功")
        except Exception as e:
            self.status_label.setText(f"创建3D预览失败: {str(e)}")
    
    def check_3d_distribution(self):
        """检查3D分布"""
        if not self.particles:
            self.stats_label.setText("未生成任何粒子")
            return
        
        # 获取所有粒子的位置
        positions = np.array([p['position'] for p in self.particles])
        
        # 计算统计信息
        particle_count = len(positions)
        min_z, max_z = np.min(positions[:, 2]), np.max(positions[:, 2])
        avg_z = np.mean(positions[:, 2])
        std_z = np.std(positions[:, 2])
        
        # 显示统计信息
        stats_text = f"粒子数量: {particle_count}\n"
        stats_text += f"Z坐标范围: {min_z:.2f} 到 {max_z:.2f} (m)\n"
        stats_text += f"Z坐标平均值: {avg_z:.2f} (m)\n"
        stats_text += f"Z坐标标准差: {std_z:.4f} (m)\n"
        
        if std_z > 0.01:
            stats_text += "✓ 粒子具有3D分布，Z坐标存在变化\n"
        else:
            stats_text += "✗ 粒子Z坐标几乎没有变化，仍然是平面分布\n"
        
        # 检查X和Y坐标范围
        min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
        min_y, max_y = np.min(positions[:, 1]), np.max(positions[:, 1])
        
        stats_text += f"X坐标范围: {min_x:.2f} 到 {max_x:.2f} (m)\n"
        stats_text += f"Y坐标范围: {min_y:.2f} 到 {max_y:.2f} (m)\n"
        
        self.stats_label.setText(stats_text)
    
    def create_3d_preview(self):
        """创建3D预览"""
        # 这里我们直接调用主应用程序中的update_model_preview函数的核心逻辑
        # 但我们会简化实现，只关注3D粒子生成是否正确
        
        if not self.particles:
            return
        
        # 直接使用matplotlib创建3D图
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 设置Matplotlib支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取粒子位置
        positions = np.array([p['position'] for p in self.particles])
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 10))
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
        
        # 设置标题和标签
        ax.set_title('边坡模型3D粒子分布', fontsize=16)
        ax.set_xlabel('X 坐标 (m)', fontsize=12)
        ax.set_ylabel('Y 坐标 (m)', fontsize=12)
        ax.set_zlabel('Z 坐标 (m)', fontsize=12)
        
        # 设置坐标轴范围
        ax.set_xlim([-15, 15])
        ax.set_ylim([-5, 15])
        ax.set_zlim([-5, 5])
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 添加网格
        ax.grid(True)
        
        # 添加统计信息文本
        stats_text = f"粒子数量: {len(positions)}\n"
        stats_text += f"Z坐标范围: {np.min(positions[:, 2]):.2f} 到 {np.max(positions[:, 2]):.2f} (m)\n"
        stats_text += f"Z坐标标准差: {np.std(positions[:, 2]):.4f} (m)"
        
        # 在图上添加文本
        ax.text2D(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig('3d_preview_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("3D预览图像已保存为: 3d_preview_test.png")
        self.status_label.setText("3D模型预览创建成功，图像已保存")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    
    # 运行测试
    window.test_3d_preview()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
