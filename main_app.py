#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子法岩土边坡仿真软件主程序

核心架构：
1. 核心层：包含粒子系统、邻居搜索、本构模型等核心算法
2. 功能层：包含边坡模型构建、仿真控制、稳定性评估等功能模块
3. UI层：使用PyQt5实现主窗口和各种功能界面
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
                             QPushButton, QLabel, QLineEdit, QComboBox, 
                             QSpinBox, QDoubleSpinBox, QProgressBar, 
                             QTableWidget, QTableWidgetItem, QTextEdit,
                             QFileDialog, QMessageBox, QAction, QMenuBar,
                             QStatusBar, QDockWidget, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap

# 导入核心功能模块
from run_simulation import SlopeModelBuilder, StabilityEvaluator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class ParameterTemplateManager:
    """参数模板管理器类"""
    
    def __init__(self):
        self.templates = self._load_templates()
        
    def _load_templates(self):
        """加载模板"""
        # 内置默认模板
        default_templates = [
            {
                'name': '普通黏性土',
                'lithology': '黏性土',
                'condition': '一般工况',
                'params': {
                    'particle_type': 'soil',
                    'material_params': {
                        'density': 2600.0,
                        'elastic_modulus': 10e6,
                        'poisson_ratio': 0.3,
                        'cohesion': 10e3,
                        'friction_angle': 30.0
                    }
                }
            },
            {
                'name': '砂性土',
                'lithology': '砂性土',
                'condition': '一般工况',
                'params': {
                    'particle_type': 'soil',
                    'material_params': {
                        'density': 2700.0,
                        'elastic_modulus': 15e6,
                        'poisson_ratio': 0.35,
                        'cohesion': 5e3,
                        'friction_angle': 35.0
                    }
                }
            },
            {
                'name': '硬质岩石',
                'lithology': '岩石',
                'condition': '一般工况',
                'params': {
                    'particle_type': 'rock',
                    'material_params': {
                        'density': 2800.0,
                        'elastic_modulus': 50e6,
                        'poisson_ratio': 0.25,
                        'cohesion': 50e3,
                        'friction_angle': 45.0
                    }
                }
            },
            {
                'name': '软质岩石',
                'lithology': '岩石',
                'condition': '风化工况',
                'params': {
                    'particle_type': 'rock',
                    'material_params': {
                        'density': 2700.0,
                        'elastic_modulus': 20e6,
                        'poisson_ratio': 0.3,
                        'cohesion': 20e3,
                        'friction_angle': 38.0
                    }
                }
            }
        ]
        
        # 这里可以添加从文件加载模板的逻辑
        return default_templates
    
    def save_template(self, template):
        """保存模板"""
        # 检查模板名称是否已存在
        for existing_template in self.templates:
            if existing_template['name'] == template['name']:
                # 更新现有模板
                existing_template.update(template)
                return True
        
        # 添加新模板
        self.templates.append(template)
        return True
    
    def load_template(self, template_name):
        """加载模板"""
        for template in self.templates:
            if template['name'] == template_name:
                return template
        return None
    
    def delete_template(self, template_name):
        """删除模板"""
        for i, template in enumerate(self.templates):
            if template['name'] == template_name:
                del self.templates[i]
                return True
        return False
    
    def get_templates(self):
        """获取所有模板"""
        return self.templates.copy()
    
    def get_template_by_lithology(self, lithology):
        """根据岩性获取模板"""
        return [template for template in self.templates if template['lithology'] == lithology]
    
    def export_templates(self, file_path):
        """导出模板到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.templates, f, indent=2)
    
    def import_templates(self, file_path):
        """从文件导入模板"""
        import json
        with open(file_path, 'r') as f:
            imported_templates = json.load(f)
        self.templates.extend(imported_templates)

class BatchSimulationManager:
    """批量仿真管理器类"""
    
    def __init__(self):
        self.batch_tasks = []
        self.results = []
        
    def create_batch_task(self, base_params, param_ranges):
        """创建批量仿真任务"""
        # 生成参数组合
        param_combinations = self._generate_param_combinations(base_params, param_ranges)
        
        # 创建任务列表
        tasks = []
        for i, params in enumerate(param_combinations):
            task = {
                'id': i + 1,
                'params': params,
                'status': 'pending',
                'result': None
            }
            tasks.append(task)
        
        self.batch_tasks = tasks
        return tasks
    
    def _generate_param_combinations(self, base_params, param_ranges):
        """生成参数组合"""
        # 简化实现，生成参数组合
        import itertools
        
        # 提取参数范围
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        # 生成所有组合
        combinations = list(itertools.product(*param_values))
        
        # 构建参数字典
        param_combinations = []
        for combo in combinations:
            params = base_params.copy()
            for name, value in zip(param_names, combo):
                params[name] = value
            param_combinations.append(params)
        
        return param_combinations
    
    def run_batch_simulation(self):
        """运行批量仿真"""
        # 简化实现，模拟批量仿真
        results = []
        for task in self.batch_tasks:
            # 模拟仿真结果
            result = {
                'task_id': task['id'],
                'safety_factor': np.random.uniform(1.0, 2.5),
                'max_displacement': np.random.uniform(0.001, 0.1),
                'status': 'completed',
                'params': task['params']
            }
            results.append(result)
            task['status'] = 'completed'
            task['result'] = result
        
        self.results = results
        return results
    
    def get_batch_tasks(self):
        """获取批量任务列表"""
        return self.batch_tasks.copy()
    
    def get_results(self):
        """获取批量仿真结果"""
        return self.results.copy()
    
    def export_results(self, file_path, format='csv'):
        """导出批量仿真结果"""
        if format == 'csv':
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['Task ID', 'Safety Factor', 'Max Displacement', 'Status'] + 
                              list(self.results[0]['params'].keys()) if self.results else [])
                
                # 写入数据
                for result in self.results:
                    row = [result['task_id'], result['safety_factor'], result['max_displacement'], result['status']]
                    row.extend(list(result['params'].values()))
                    writer.writerow(row)
        elif format == 'json':
            import json
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=2)

class VisualizationModule(QWidget):
    """可视化模块类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化属性
        self.particles = []
        self.current_display_property = 'displacement'  # 默认显示位移
        self.display_mode = 'scatter'  # scatter, contour, surface
        self.colormap = 'viridis'  # 颜色映射
        self.animation_running = False
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.animation = None
        self.particle_data_history = []  # 存储历史粒子数据用于动画
        self.current_frame = 0
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        """初始化可视化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建可视化控制工具栏
        control_layout = QHBoxLayout()
        
        # 显示属性选择
        self.property_combo = QComboBox()
        self.property_combo.addItems(['displacement', 'velocity', 'stress', 'strain'])
        self.property_combo.currentTextChanged.connect(self._change_display_property)
        control_layout.addWidget(QLabel("显示属性:"))
        control_layout.addWidget(self.property_combo)
        
        # 显示模式选择
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['scatter', 'contour', 'surface'])
        self.mode_combo.currentTextChanged.connect(self._change_display_mode)
        control_layout.addWidget(QLabel("显示模式:"))
        control_layout.addWidget(self.mode_combo)
        
        # 颜色映射选择
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                                      'jet', 'coolwarm', 'RdYlBu', 'GnBu', 'OrRd'])
        self.colormap_combo.currentTextChanged.connect(self._change_colormap)
        control_layout.addWidget(QLabel("颜色映射:"))
        control_layout.addWidget(self.colormap_combo)
        
        # 动画控制按钮
        self.play_button = QPushButton("播放动画")
        self.play_button.clicked.connect(self._toggle_animation)
        control_layout.addWidget(self.play_button)
        
        # 截图按钮
        self.screenshot_button = QPushButton("保存截图")
        self.screenshot_button.clicked.connect(self._save_screenshot)
        control_layout.addWidget(self.screenshot_button)
        
        # 视频导出按钮
        self.video_button = QPushButton("导出视频")
        self.video_button.clicked.connect(self._export_video)
        control_layout.addWidget(self.video_button)
        
        # 清除按钮
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self._clear_visualization)
        control_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(control_layout)
        
        # 创建Matplotlib画布
        self.fig, self.ax = plt.subplots(figsize=(10, 8), dpi=100, subplot_kw={'projection': '3d'})
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        # 设置布局
        self.setLayout(main_layout)
        
    def _change_display_property(self, property_name):
        """改变显示属性"""
        self.current_display_property = property_name
        self._update_visualization()
    
    def _change_display_mode(self, mode):
        """改变显示模式"""
        self.display_mode = mode
        self._update_visualization()
    
    def _change_colormap(self, colormap):
        """改变颜色映射"""
        self.colormap = colormap
        self._update_visualization()
    
    def _update_visualization(self):
        """更新可视化显示"""
        if not self.particles:
            return
        
        self.ax.clear()
        
        # 提取数据
        positions = np.array([p['position'] for p in self.particles])
        
        # 根据选择的属性获取数据
        if self.current_display_property == 'displacement':
            # 计算位移
            if 'displacement' in self.particles[0]:
                values = np.array([np.linalg.norm(p['displacement']) for p in self.particles])
            else:
                # 没有位移数据，使用位置信息
                values = positions[:, 1]  # 使用y坐标
            title = '位移场分布'
            cbar_label = '位移 (m)'
        elif self.current_display_property == 'velocity':
            values = np.array([np.linalg.norm(p['velocity']) for p in self.particles])
            title = '速度场分布'
            cbar_label = '速度 (m/s)'
        elif self.current_display_property == 'stress':
            if 'stress' in self.particles[0]:
                # 计算最大主应力
                values = []
                for p in self.particles:
                    sigma = p['stress']
                    eigenvalues, _ = np.linalg.eigh(sigma)
                    max_principal_stress = np.max(eigenvalues)
                    values.append(max_principal_stress)
                values = np.array(values)
            else:
                values = positions[:, 1]
            title = '应力场分布'
            cbar_label = '最大主应力 (Pa)'
        elif self.current_display_property == 'strain':
            if 'strain' in self.particles[0]:
                # 计算应变张量的第二不变量
                values = []
                for p in self.particles:
                    strain = p['strain']
                    # 计算应变第二不变量
                    J2 = 0.5 * (np.trace(strain)**2 - np.trace(np.dot(strain, strain)))
                    values.append(np.sqrt(J2))
                values = np.array(values)
            else:
                values = positions[:, 1]
            title = '应变场分布'
            cbar_label = '应变第二不变量'
        else:
            values = positions[:, 1]
            title = '粒子分布'
            cbar_label = 'Y坐标'
        
        # 设置坐标轴范围
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        
        # 添加边距
        x_padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        z_padding = (z_max - z_min) * 0.1 if z_max != z_min else 1.0
        
        # 根据显示模式绘制
        if self.display_mode == 'scatter':
            scatter = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                                     c=values, cmap=self.colormap, s=10, alpha=0.8)
            cbar = self.fig.colorbar(scatter, ax=self.ax, label=cbar_label)
        elif self.display_mode == 'contour':
            try:
                # 使用插值方法构建网格数据
                from scipy.interpolate import griddata
                
                # 创建网格
                grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                
                # 插值
                grid_z = griddata(positions[:, :2], values, (grid_x, grid_y), method='cubic')
                
                # 绘制等值线图
                contour = self.ax.contourf(grid_x, grid_y, grid_z, cmap=self.colormap, alpha=0.8, levels=20)
                cbar = self.fig.colorbar(contour, ax=self.ax, label=cbar_label)
            except Exception:
                # 插值失败时回退到散点图
                scatter = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                                         c=values, cmap=self.colormap, s=10, alpha=0.8)
                cbar = self.fig.colorbar(scatter, ax=self.ax, label=cbar_label)
        elif self.display_mode == 'surface':
            try:
                # 使用插值方法构建网格数据
                from scipy.interpolate import griddata
                
                # 创建网格
                grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                
                # 插值
                grid_z = griddata(positions[:, :2], values, (grid_x, grid_y), method='cubic')
                
                # 绘制曲面图
                surface = self.ax.plot_surface(grid_x, grid_y, grid_z, cmap=self.colormap, alpha=0.8, rcount=100, ccount=100)
                cbar = self.fig.colorbar(surface, ax=self.ax, label=cbar_label)
            except Exception:
                # 插值失败时回退到散点图
                scatter = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                                         c=values, cmap=self.colormap, s=10, alpha=0.8)
                cbar = self.fig.colorbar(scatter, ax=self.ax, label=cbar_label)
        
        # 设置标题和标签
        self.ax.set_title(title)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 设置坐标轴范围
        self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
        self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
        self.ax.set_zlim(z_min - z_padding, z_max + z_padding)
        
        # 设置视角
        self.ax.view_init(elev=30, azim=45)
        
        # 调整布局，确保图形完整显示
        self.fig.tight_layout()
        
        # 更新画布
        self.canvas.draw()
        """更新可视化显示"""
        if not self.particles:
            return
        
        self.ax.clear()
        
        # 提取数据
        positions = np.array([p['position'] for p in self.particles])
        
        # 根据选择的属性获取数据
        if self.current_display_property == 'displacement':
            # 计算位移
            if 'displacement' in self.particles[0]:
                values = np.array([np.linalg.norm(p['displacement']) for p in self.particles])
            else:
                # 没有位移数据，使用位置信息
                values = positions[:, 1]  # 使用y坐标
            title = '位移场分布'
            cbar_label = '位移 (m)'
        elif self.current_display_property == 'velocity':
            values = np.array([np.linalg.norm(p['velocity']) for p in self.particles])
            title = '速度场分布'
            cbar_label = '速度 (m/s)'
        elif self.current_display_property == 'stress':
            if 'stress' in self.particles[0]:
                # 计算最大主应力
                values = []
                for p in self.particles:
                    sigma = p['stress']
                    eigenvalues, _ = np.linalg.eigh(sigma)
                    max_principal_stress = np.max(eigenvalues)
                    values.append(max_principal_stress)
                values = np.array(values)
            else:
                values = positions[:, 1]
            title = '应力场分布'
            cbar_label = '最大主应力 (Pa)'
        elif self.current_display_property == 'strain':
            if 'strain' in self.particles[0]:
                # 计算应变张量的第二不变量
                values = []
                for p in self.particles:
                    strain = p['strain']
                    # 计算应变第二不变量
                    J2 = 0.5 * (np.trace(strain)**2 - np.trace(np.dot(strain, strain)))
                    values.append(np.sqrt(J2))
                values = np.array(values)
            else:
                values = positions[:, 1]
            title = '应变场分布'
            cbar_label = '应变第二不变量'
        else:
            values = positions[:, 1]
            title = '粒子分布'
            cbar_label = 'Y坐标'
        
        # 根据显示模式绘制
        if self.display_mode == 'scatter':
            scatter = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                                     c=values, cmap=self.colormap, s=10, alpha=0.8)
            cbar = self.fig.colorbar(scatter, ax=self.ax, label=cbar_label)
        elif self.display_mode == 'contour':
            # 简化的等值线图（2D投影）
            self.ax.contourf(positions[:, 0], positions[:, 1], values.reshape(-1, int(np.sqrt(len(values)))), 
                           cmap=self.colormap, alpha=0.8)
        elif self.display_mode == 'surface':
            # 简化的曲面图（2D投影）
            self.ax.plot_surface(positions[:, 0].reshape(-1, int(np.sqrt(len(values)))), 
                                positions[:, 1].reshape(-1, int(np.sqrt(len(values)))),
                                values.reshape(-1, int(np.sqrt(len(values)))),
                                cmap=self.colormap, alpha=0.8)
        
        # 设置标题和标签
        self.ax.set_title(title)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 设置视角
        self.ax.view_init(elev=30, azim=45)
        
        # 更新画布
        self.canvas.draw()
    
    def update_particles(self, particles):
        """更新粒子数据"""
        self.particles = particles
        self.particle_data_history.append(particles.copy())  # 保存到历史记录
        self._update_visualization()
    
    def _toggle_animation(self):
        """切换动画播放状态"""
        if self.animation_running:
            self._stop_animation()
        else:
            self._start_animation()
    
    def _start_animation(self):
        """开始动画"""
        if not self.particle_data_history:
            return
        
        self.animation_running = True
        self.play_button.setText("暂停动画")
        
        # 创建动画
        self.animation = FuncAnimation(self.fig, self._animate_frame, 
                                      frames=len(self.particle_data_history),
                                      interval=100, blit=False, repeat=True)
        
        self.canvas.draw()
    
    def _stop_animation(self):
        """停止动画"""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        
        self.animation_running = False
        self.play_button.setText("播放动画")
    
    def _animate_frame(self, frame):
        """动画帧更新"""
        self.current_frame = frame
        self.particles = self.particle_data_history[frame].copy()
        self._update_visualization()
    
    def _save_screenshot(self):
        """保存截图"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存截图", "visualization_screenshot.png", "PNG Files (*.png)")
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
    
    def _export_video(self):
        """导出视频"""
        # 简化实现，显示提示
        QMessageBox.information(self, "提示", "视频导出功能将在后续版本中实现！")
    
    def _clear_visualization(self):
        """清除可视化"""
        self.ax.clear()
        self.particles = []
        self.particle_data_history = []
        self.canvas.draw()
    
    def clear_data(self):
        """清除所有数据"""
        self._stop_animation()
        self._clear_visualization()
        self.particle_data_history.clear()

class SimulationThread(QThread):
    """仿真计算线程类"""
    
    # 定义信号
    progress_update = pyqtSignal(int)  # 进度更新信号
    simulation_finished = pyqtSignal(dict)  # 仿真完成信号
    status_update = pyqtSignal(str)  # 状态更新信号
    particle_update = pyqtSignal(list)  # 粒子状态更新信号（用于实时可视化）
    
    def __init__(self, particles, simulation_params):
        super().__init__()
        self.particles = particles
        self.simulation_params = simulation_params
        self.is_running = True
        self.is_paused = False
        
        # 检查是否使用GPU加速
        self.use_gpu = simulation_params.get('use_gpu', False)
        self.xp = np  # 默认使用NumPy
        
        # 尝试导入CuPy库
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
                self.status_update.emit("已启用GPU加速")
            except ImportError:
                self.status_update.emit("未找到CuPy库，将使用CPU计算")
                self.use_gpu = False
        
    def run(self):
        """线程运行函数"""
        self.status_update.emit("开始仿真计算...")
        
        try:
            # 初始化仿真参数
            total_steps = self.simulation_params.get('total_steps', 100)
            dt = self.simulation_params.get('time_step', 0.01)
            constitutive_model = self.simulation_params.get('constitutive_model', '弹性模型')
            
            # 复制初始粒子状态
            current_particles = [p.copy() for p in self.particles]
            initial_particles = [p.copy() for p in self.particles]
            
            # 计算初始应力和应变
            self._initialize_stress_strain(current_particles)
            
            # 主仿真循环
            for step in range(total_steps):
                if not self.is_running:
                    break
                
                if self.is_paused:
                    self.status_update.emit(f"仿真已暂停，当前进度: {int(step/total_steps*100)}%")
                    while self.is_paused and self.is_running:
                        self.msleep(100)
                    if not self.is_running:
                        break
                
                # 更新粒子状态
                self._update_particles(current_particles, dt, constitutive_model)
                
                # 更新进度
                progress = int((step + 1) / total_steps * 100)
                self.progress_update.emit(progress)
                
                # 每10步更新一次可视化
                if step % 10 == 0:
                    self.particle_update.emit(current_particles.copy())
                    
                # 更新状态信息
                self.status_update.emit(f"仿真进行中，当前进度: {progress}%，时间步: {step+1}/{total_steps}")
                
                # 模拟计算耗时
                self.msleep(30)
            
            if self.is_running:
                # 最后一次更新可视化
                self.particle_update.emit(current_particles.copy())
                
                # 评估稳定性
                self.status_update.emit("评估边坡稳定性...")
                stability_evaluator = StabilityEvaluator()
                indices = stability_evaluator.evaluate_stability(current_particles, initial_particles)
                
                self.status_update.emit("仿真计算完成！")
                self.simulation_finished.emit({
                    'deformed_particles': current_particles,
                    'stability_indices': indices,
                    'simulation_params': self.simulation_params
                })
        except Exception as e:
            self.status_update.emit(f"仿真计算出错: {str(e)}")
            self.simulation_finished.emit({'error': str(e)})
    
    def _initialize_stress_strain(self, particles):
        """初始化粒子的应力和应变状态"""
        for p in particles:
            # 根据材料属性初始化应力和应变
            if 'material_params' in p:
                # 简单的初始应力计算（自重应力）
                depth = p['position'][1]  # 简化的深度计算
                if depth > 0:  # 只对边坡区域应用初始应力
                    p['stress'][1][1] = -p['material_params']['density'] * 9.81 * depth  # 竖向自重应力
                    p['stress'][0][0] = p['stress'][1][1] * p['material_params'].get('poisson_ratio', 0.3)  # 横向自重应力
                    p['stress'][2][2] = p['stress'][0][0]  # 三维情况
    
    def _update_particles(self, particles, dt, constitutive_model):
        """更新粒子状态"""
        # 计算每个粒子的受力
        for p in particles:
            # 重力
            gravity = np.array([0.0, -9.81, 0.0]) * p['mass']
            
            # 简化的接触力（基于应力梯度）
            contact_force = self._compute_contact_force(p, particles)
            
            # 总受力
            total_force = gravity + contact_force
            
            # 更新加速度
            p['acceleration'] = total_force / p['mass']
        
        # 更新速度和位置（显式欧拉法）
        for p in particles:
            # 更新速度
            p['velocity'] += p['acceleration'] * dt
            
            # 应用阻尼
            damping = 0.99  # 简化的阻尼系数
            p['velocity'] *= damping
            
            # 更新位置
            p['position'] += p['velocity'] * dt
        
        # 更新应力应变（基于本构模型）
        for p in particles:
            self._update_stress_strain(p, dt, constitutive_model)
    
    def _compute_contact_force(self, particle, particles):
        """计算接触力"""
        contact_force = self.xp.zeros(3)
        
        # 查找邻近粒子
        search_radius = particle['radius'] * 2.0
        
        for other in particles:
            if particle['id'] == other['id']:
                continue
            
            # 计算粒子间距离和相对位置
            delta_pos = other['position'] - particle['position']
            distance = self.xp.linalg.norm(delta_pos)
            
            if distance < search_radius and distance > 1e-10:
                # 归一化相对位置向量
                unit_vec = delta_pos / distance
                
                # 计算重叠量
                overlap = particle['radius'] + other['radius'] - distance
                
                if overlap > 0:
                    # 基于岩土材料的接触力模型
                    # 1. 法向接触力（弹簧-阻尼模型）
                    # 弹簧刚度根据材料属性动态计算
                    E1 = particle.get('material_params', {}).get('elastic_modulus', 10e6)
                    E2 = other.get('material_params', {}).get('elastic_modulus', 10e6)
                    nu1 = particle.get('material_params', {}).get('poisson_ratio', 0.3)
                    nu2 = other.get('material_params', {}).get('poisson_ratio', 0.3)
                    
                    # 等效弹性模量
                    E_eq = (E1 * E2) / (E1 + E2)
                    
                    # 弹簧刚度（考虑粒子大小）
                    spring_k = E_eq * self.xp.sqrt(overlap * (particle['radius'] * other['radius']))
                    
                    # 法向弹簧力
                    normal_spring_force = spring_k * overlap * unit_vec
                    
                    # 法向阻尼力
                    relative_vel = other['velocity'] - particle['velocity']
                    normal_vel = self.xp.dot(relative_vel, unit_vec)
                    damping_c = 0.1 * spring_k  # 阻尼系数与刚度成正比
                    normal_damping_force = damping_c * normal_vel * unit_vec
                    
                    # 2. 切向接触力（库仑摩擦模型）
                    # 计算切向相对速度
                    tangential_vel = relative_vel - normal_vel * unit_vec
                    
                    # 切向弹簧力（基于剪切模量）
                    G1 = E1 / (2 * (1 + nu1))
                    G2 = E2 / (2 * (1 + nu2))
                    G_eq = (G1 * G2) / (G1 + G2)
                    
                    # 切向刚度
                    tangential_k = G_eq * self.xp.sqrt(overlap * (particle['radius'] * other['radius']))
                    
                    # 切向弹簧力
                    tangential_spring_force = tangential_k * tangential_vel * 0.01  # 小系数避免过大切向力
                    
                    # 3. 摩擦约束（库仑摩擦定律）
                    mu = min(particle.get('material_params', {}).get('friction_angle', 30.0), 
                            other.get('material_params', {}).get('friction_angle', 30.0))
                    mu = self.xp.tan(self.xp.radians(mu))  # 转换为摩擦系数
                    
                    normal_force_magnitude = self.xp.linalg.norm(normal_spring_force + normal_damping_force)
                    max_tangential_force = mu * normal_force_magnitude
                    
                    tangential_force_magnitude = self.xp.linalg.norm(tangential_spring_force)
                    if tangential_force_magnitude > max_tangential_force:
                        # 发生滑动，应用库仑摩擦约束
                        tangential_spring_force = (tangential_spring_force / tangential_force_magnitude) * max_tangential_force
                    
                    # 总接触力
                    contact_force += normal_spring_force + normal_damping_force + tangential_spring_force
        
        # 如果使用GPU，将结果转换回CPU
        if self.use_gpu:
            contact_force = self.xp.asnumpy(contact_force)
        
        return contact_force
    
    def _update_stress_strain(self, particle, dt, constitutive_model):
        """更新应力应变"""
        # 更准确的应变率计算
        strain_rate = np.zeros((3, 3))
        
        # 1. 基于粒子运动的应变率计算
        v = particle['velocity']
        
        # 2. 考虑邻近粒子影响的应变率计算（更准确）
        # 获取粒子加速度
        a = particle['acceleration']
        
        # 计算速度梯度张量（简化实现，考虑粒子自身运动）
        velocity_gradient = np.zeros((3, 3))
        
        # 基于速度变化率的速度梯度
        # 假设应变率与速度变化率成正比
        for i in range(3):
            for j in range(3):
                if i == j:
                    # 法向应变率
                    velocity_gradient[i][j] = a[i] * dt  # 基于加速度的速度变化
                else:
                    # 切向应变率
                    velocity_gradient[i][j] = 0.5 * (v[i] + v[j]) * dt
        
        # 应变率张量是速度梯度张量的对称部分
        strain_rate = 0.5 * (velocity_gradient + velocity_gradient.T)
        
        # 更新应变
        particle['strain'] += strain_rate * dt
        
        # 3. 基于塑性变形的修正
        # 如果是塑性模型，考虑塑性应变
        if constitutive_model == "Mohr-Coulomb模型":
            # 记录初始应力用于屈服判断
            initial_stress = particle['stress'].copy()
        
        # 根据本构模型更新应力
        if constitutive_model == "弹性模型":
            self._update_stress_elastic(particle, strain_rate, dt)
        elif constitutive_model == "Mohr-Coulomb模型":
            self._update_stress_mohr_coulomb(particle, strain_rate, dt)
        elif constitutive_model == "黏弹性模型":
            self._update_stress_viscoelastic(particle, strain_rate, dt)
        elif constitutive_model == "Drucker-Prager模型":
            self._update_stress_drucker_prager(particle, strain_rate, dt)
        elif constitutive_model == "Hyperelastic模型":
            self._update_stress_hyperelastic(particle, strain_rate, dt)
    
    def _update_stress_elastic(self, particle, strain_rate, dt):
        """弹性模型应力更新（高分辨率实现）"""
        if 'material_params' not in particle:
            return
        
        E = particle['material_params'].get('elastic_modulus', 10e6)
        nu = particle['material_params'].get('poisson_ratio', 0.3)
        
        # 计算弹性刚度张量
        K = E / (3 * (1 - 2 * nu))  # 体积模量
        G = E / (2 * (1 + nu))  # 剪切模量
        
        # 计算应变增量
        strain_increment = strain_rate * dt
        
        # 高精度应力增量计算
        # 1. 计算应变增量的球张量和偏张量
        strain_incr_spherical = (np.trace(strain_increment) / 3) * np.eye(3)
        strain_incr_deviatoric = strain_increment - strain_incr_spherical
        
        # 2. 计算应力增量
        # 体积应力增量（基于体积模量）
        stress_incr_spherical = 3 * K * strain_incr_spherical
        
        # 偏应力增量（基于剪切模量）
        stress_incr_deviatoric = 2 * G * strain_incr_deviatoric
        
        # 总应力增量
        stress_increment = stress_incr_spherical + stress_incr_deviatoric
        
        # 3. 数值稳定化处理
        # 添加小应变正则化，提高数值稳定性
        regularization_factor = 1.0 + 1e-8
        stress_increment *= regularization_factor
        
        # 4. 更新应力
        particle['stress'] += stress_increment
        
        # 5. 应力状态记录（用于调试和分析）
        particle['stress_state'] = {
            'elastic_strain': particle['strain'].copy(),
            'elastic_stress': particle['stress'].copy(),
            'E': E,
            'nu': nu,
            'K': K,
            'G': G
        }
    
    def _update_stress_mohr_coulomb(self, particle, strain_rate, dt):
        """Mohr-Coulomb模型应力更新"""
        if 'material_params' not in particle:
            # 回退到弹性模型
            self._update_stress_elastic(particle, strain_rate, dt)
            return
        
        # 简化的Mohr-Coulomb模型实现
        # 先计算弹性应力增量
        self._update_stress_elastic(particle, strain_rate, dt)
        
        # 检查屈服条件
        sigma = particle['stress']
        
        # 计算主应力
        eigenvalues, _ = np.linalg.eigh(sigma)
        sigma1, sigma2, sigma3 = eigenvalues[::-1]  # 主应力排序
        
        # 计算屈服函数
        c = particle['material_params'].get('cohesion', 10e3)
        phi = np.radians(particle['material_params'].get('friction_angle', 30.0))
        
        # Mohr-Coulomb屈服准则
        yield_function = (sigma1 - sigma3) - (sigma1 + sigma3) * np.sin(phi) - 2 * c * np.cos(phi)
        
        if yield_function > 0:
            # 发生屈服，进行塑性修正
            # 简化的塑性流动规则
            plastic_multiplier = yield_function / (2 * (1 + np.sin(phi)))
            
            # 更新应力
            delta_sigma = plastic_multiplier * np.array([
                [1 - np.sin(phi), 0, 0],
                [0, -np.sin(phi), 0],
                [0, 0, -1 - np.sin(phi)]
            ])
            
            # 应用塑性修正
            particle['stress'] += delta_sigma
    
    def _update_stress_viscoelastic(self, particle, strain_rate, dt):
        """黏弹性模型应力更新"""
        if 'material_params' not in particle:
            # 回退到弹性模型
            self._update_stress_elastic(particle, strain_rate, dt)
            return
        
        # 简化的Maxwell黏弹性模型
        E = particle['material_params'].get('elastic_modulus', 10e6)
        nu = particle['material_params'].get('poisson_ratio', 0.3)
        viscosity = particle['material_params'].get('viscosity', 1e9)  # 简化的黏度参数
        
        # 体积模量和剪切模量
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        
        # 应变增量
        strain_increment = strain_rate * dt
        
        # 应力更新（黏弹性模型）
        dev_strain_incr = strain_increment - (np.trace(strain_increment) / 3) * np.eye(3)
        
        # 弹性部分
        elastic_stress_incr = 2 * G * dev_strain_incr + K * np.trace(strain_increment) * np.eye(3)
        
        # 黏性部分（简化实现）
        viscous_stress_incr = -particle['stress'] * dt / (viscosity / G)
        
        # 总应力增量
        stress_increment = elastic_stress_incr + viscous_stress_incr
        
        # 更新应力
        particle['stress'] += stress_increment
    
    def _update_stress_drucker_prager(self, particle, strain_rate, dt):
        """Drucker-Prager模型应力更新"""
        if 'material_params' not in particle:
            # 回退到弹性模型
            self._update_stress_elastic(particle, strain_rate, dt)
            return
        
        # 先计算弹性应力增量
        self._update_stress_elastic(particle, strain_rate, dt)
        
        # Drucker-Prager屈服准则
        sigma = particle['stress']
        
        # 计算主应力
        eigenvalues, _ = np.linalg.eigh(sigma)
        sigma1, sigma2, sigma3 = eigenvalues[::-1]  # 主应力排序
        
        # 计算平均应力和等效剪应力
        p = (sigma1 + sigma2 + sigma3) / 3.0  # 平均应力
        q = np.sqrt(0.5 * ((sigma1 - sigma2)**2 + (sigma2 - sigma3)**2 + (sigma3 - sigma1)**2))  # 等效剪应力
        
        # 材料参数
        c = particle['material_params'].get('cohesion', 10e3)
        phi = np.radians(particle['material_params'].get('friction_angle', 30.0))
        psi = np.radians(particle['material_params'].get('dilation_angle', 10.0))  # 剪胀角
        
        # Drucker-Prager屈服函数
        alpha = np.sin(phi) / np.sqrt(3) / np.sqrt(3 + np.sin(phi)**2)
        k = 2 * c * np.cos(phi) / np.sqrt(3) / np.sqrt(3 + np.sin(phi)**2)
        yield_function = q - alpha * p - k
        
        if yield_function > 0:
            # 发生屈服，进行塑性修正
            # 简化的塑性流动规则
            plastic_multiplier = yield_function / (1 + alpha)
            
            # 更新应力
            delta_sigma = plastic_multiplier * np.array([
                [1 - alpha, 0, 0],
                [0, 1 - alpha, 0],
                [0, 0, 1 - alpha]
            ])
            
            particle['stress'] += delta_sigma
    
    def _update_stress_hyperelastic(self, particle, strain_rate, dt):
        """Hyperelastic模型应力更新"""
        if 'material_params' not in particle:
            # 回退到弹性模型
            self._update_stress_elastic(particle, strain_rate, dt)
            return
        
        # 简化的超弹性模型（基于Saint-Venant Kirchhoff模型）
        E = particle['material_params'].get('elastic_modulus', 10e6)
        nu = particle['material_params'].get('poisson_ratio', 0.3)
        
        # 体积模量和剪切模量
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        
        # 计算应变张量
        strain = particle['strain']
        
        # 计算Green-Lagrange应变张量（超弹性模型常用）
        E_green = 0.5 * (strain + strain.T + np.dot(strain.T, strain))
        
        # 计算第二Piola-Kirchhoff应力张量
        S = 2 * G * E_green + K * np.trace(E_green) * np.eye(3)
        
        # 将第二Piola-Kirchhoff应力转换为Cauchy应力
        # 简化实现，假设小变形
        particle['stress'] = S
    
    def pause(self):
        """暂停仿真"""
        self.is_paused = True
    
    def resume(self):
        """恢复仿真"""
        self.is_paused = False
    
    def stop(self):
        """停止仿真"""
        self.is_running = False
        self.is_paused = False

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化核心对象
        self.slope_builder = SlopeModelBuilder()
        self.stability_evaluator = StabilityEvaluator()
        
        # 初始化数据
        self.particles = []
        self.initial_particles = []
        self.deformed_particles = []
        self.stability_indices = {}
        
        # 初始化UI
        self.init_ui()
        
        # 初始化仿真线程
        self.simulation_thread = None
        
    def init_ui(self):
        """初始化UI界面"""
        # 设置窗口标题和大小
        self.setWindowTitle("粒子法岩土边坡仿真软件")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        
        # 设置应用程序图标
        # self.setWindowIcon(QIcon("icon.png"))  # 可以添加软件图标
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabShape(QTabWidget.Rounded)
        self.tab_widget.setElideMode(Qt.ElideNone)  # 不省略文字
        self.tab_widget.setUsesScrollButtons(True)
        self.tab_widget.setDocumentMode(True)
        
        # 设置选项卡标签字体和大小
        font = self.tab_widget.font()
        font.setPointSize(12)  # 设置合适的字体大小
        self.tab_widget.setFont(font)
        
        # 创建各个功能页面
        self.model_page = self.create_model_page()
        self.simulation_page = self.create_simulation_page()
        self.result_page = self.create_result_page()
        self.extension_page = self.create_extension_page()
        
        # 添加页面到选项卡
        self.tab_widget.addTab(self.model_page, "边坡建模")
        self.tab_widget.addTab(self.simulation_page, "仿真控制")
        self.tab_widget.addTab(self.result_page, "结果分析")
        self.tab_widget.addTab(self.extension_page, "拓展功能")
        
        # 添加选项卡到主布局
        main_layout.addWidget(self.tab_widget)
        
        # 设置主控件
        self.setCentralWidget(main_widget)
        
        # 初始化主题
        self.is_dark_theme = False
        self.theme_action.setChecked(False)
        self.apply_light_theme()
    
    def apply_light_theme(self):
        """应用浅色主题"""
        self.is_dark_theme = False
        self.setStyleSheet("""
            /* 主窗口样式 */
            QMainWindow {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            
            /* 选项卡控件 */
            QTabWidget {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 5px;
            }
            
            /* 选项卡标签 */
            QTabBar::tab {
                background-color: #e9ecef;
                color: #495057;
                padding: 10px 20px;
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
                font-size: 14px;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-color: #dee2e6;
                border-bottom-color: transparent;
                font-weight: 600;
                color: #007bff;
            }
            
            QTabBar::tab:hover:not(:selected) {
                background-color: #dee2e6;
            }
            
            /* 分组框样式 */
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 20px;
            }
            
            QGroupBox::title {
                color: #495057;
                font-size: 14px;
                font-weight: 600;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: #ffffff;
            }
            
            /* 按钮样式 */
            QPushButton {
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 600;
                min-height: 38px;
                transition: all 0.2s ease-in-out;
                box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
            }
            
            QPushButton:hover {
                background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
            }
            
            QPushButton:pressed {
                background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
                transform: translateY(0);
            }
            
            QPushButton:disabled {
                background-color: #6c757d;
                color: #ffffff;
            }
            
            /* 标签样式 */
            QLabel {
                color: #495057;
                font-size: 14px;
                font-weight: 500;
            }
            
            /* 输入控件样式 */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #ffffff;
                color: #1e293b;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px 14px;
                font-size: 14px;
                min-height: 38px;
                transition: all 0.2s ease-in-out;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #2563eb;
                outline: none;
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }
            
            /* 文本编辑框样式 */
            QTextEdit {
                background-color: #ffffff;
                color: #495057;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            
            /* 表格样式 */
            QTableWidget {
                background-color: #ffffff;
                color: #495057;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }
            
            QTableWidget::header {
                background-color: #e9ecef;
                color: #495057;
                font-weight: 600;
                border: none;
                border-bottom: 2px solid #dee2e6;
            }
            
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            QTableWidget::item:selected {
                background-color: #007bff;
                color: #ffffff;
            }
            
            /* 进度条样式 */
            QProgressBar {
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
            
            /* 状态栏样式 */
            QStatusBar {
                background-color: #e9ecef;
                color: #495057;
                border-top: 1px solid #dee2e6;
                font-size: 13px;
            }
            
            /* 分割线样式 */
            QSplitter::handle {
                background-color: #dee2e6;
                width: 4px;
                height: 4px;
            }
            
            QSplitter::handle:hover {
                background-color: #adb5bd;
            }
            
            /* 菜单样式 */
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #dee2e6;
                font-size: 14px;
            }
            
            QMenuBar::item {
                background-color: transparent;
                color: #495057;
                padding: 8px 12px;
            }
            
            QMenuBar::item:selected {
                background-color: #e9ecef;
                color: #007bff;
            }
            
            QMenu {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 0;
            }
            
            QMenu::item {
                padding: 6px 20px;
                color: #495057;
                font-size: 13px;
            }
            
            QMenu::item:selected {
                background-color: #007bff;
                color: #ffffff;
            }
        """)
    
    def apply_dark_theme(self):
        """应用深色主题"""
        self.is_dark_theme = True
        self.setStyleSheet("""
            /* 主窗口样式 */
            QMainWindow {
                background-color: #212529;
                border: 1px solid #343a40;
                border-radius: 4px;
            }
            
            /* 选项卡控件 */
            QTabWidget {
                background-color: #2c3035;
                border: 1px solid #343a40;
                border-radius: 6px;
                padding: 5px;
            }
            
            /* 选项卡标签 */
            QTabBar::tab {
                background-color: #343a40;
                color: #e9ecef;
                padding: 10px 20px;
                border: 1px solid #495057;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
                font-size: 14px;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background-color: #2c3035;
                border-color: #495057;
                border-bottom-color: transparent;
                font-weight: 600;
                color: #4dabf7;
            }
            
            QTabBar::tab:hover:not(:selected) {
                background-color: #3a3f44;
            }
            
            /* 分组框样式 */
            QGroupBox {
                background-color: #2c3035;
                border: 1px solid #343a40;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 20px;
            }
            
            QGroupBox::title {
                color: #e9ecef;
                font-size: 14px;
                font-weight: 600;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: #2c3035;
            }
            
            /* 按钮样式 */
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 600;
                min-height: 36px;
            }
            
            QPushButton:hover {
                background-color: #0056b3;
            }
            
            QPushButton:pressed {
                background-color: #004085;
            }
            
            QPushButton:disabled {
                background-color: #495057;
                color: #adb5bd;
            }
            
            /* 标签样式 */
            QLabel {
                color: #e9ecef;
                font-size: 14px;
                font-weight: 500;
            }
            
            /* 输入控件样式 */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #343a40;
                color: #e9ecef;
                border: 1px solid #495057;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 14px;
                min-height: 32px;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #4dabf7;
                outline: none;
                box-shadow: 0 0 0 0.2rem rgba(77, 171, 247, 0.25);
            }
            
            /* 文本编辑框样式 */
            QTextEdit {
                background-color: #343a40;
                color: #e9ecef;
                border: 1px solid #495057;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            
            /* 表格样式 */
            QTableWidget {
                background-color: #343a40;
                color: #e9ecef;
                border: 1px solid #495057;
                border-radius: 4px;
                font-size: 14px;
            }
            
            QTableWidget::header {
                background-color: #495057;
                color: #e9ecef;
                font-weight: 600;
                border: none;
                border-bottom: 2px solid #6c757d;
            }
            
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #495057;
            }
            
            QTableWidget::item:selected {
                background-color: #007bff;
                color: #ffffff;
            }
            
            /* 进度条样式 */
            QProgressBar {
                background-color: #343a40;
                border: 1px solid #495057;
                border-radius: 4px;
                text-align: center;
                color: #e9ecef;
                height: 20px;
            }
            
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
            
            /* 状态栏样式 */
            QStatusBar {
                background-color: #343a40;
                color: #e9ecef;
                border-top: 1px solid #495057;
                font-size: 13px;
            }
            
            /* 分割线样式 */
            QSplitter::handle {
                background-color: #495057;
                width: 4px;
                height: 4px;
            }
            
            QSplitter::handle:hover {
                background-color: #6c757d;
            }
            
            /* 菜单样式 */
            QMenuBar {
                background-color: #343a40;
                border-bottom: 1px solid #495057;
                font-size: 14px;
            }
            
            QMenuBar::item {
                background-color: transparent;
                color: #e9ecef;
                padding: 8px 12px;
            }
            
            QMenuBar::item:selected {
                background-color: #495057;
                color: #4dabf7;
            }
            
            QMenu {
                background-color: #343a40;
                border: 1px solid #495057;
                border-radius: 4px;
                padding: 4px 0;
            }
            
            QMenu::item {
                padding: 6px 20px;
                color: #e9ecef;
                font-size: 13px;
            }
            
            QMenu::item:selected {
                background-color: #007bff;
                color: #ffffff;
            }
        """)
    
    def toggle_theme(self):
        """切换主题"""
        if self.theme_action.isChecked():
            # 切换到深色主题
            self.apply_dark_theme()
        else:
            # 切换到浅色主题
            self.apply_light_theme()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        # 创建菜单栏
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 新建项目
        new_action = QAction("新建项目", self)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        # 打开项目
        open_action = QAction("打开项目", self)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        # 保存项目
        save_action = QAction("保存项目", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        # 保存项目为
        save_as_action = QAction("保存项目为", self)
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # 导出模型
        export_model_action = QAction("导出模型", self)
        export_model_action.triggered.connect(self.export_model)
        file_menu.addAction(export_model_action)
        
        # 导入模型
        import_model_action = QAction("导入模型", self)
        import_model_action.triggered.connect(self.import_model)
        file_menu.addAction(import_model_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        
        # 撤销
        undo_action = QAction("撤销", self)
        edit_menu.addAction(undo_action)
        
        # 重做
        redo_action = QAction("重做", self)
        edit_menu.addAction(redo_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        # 主题切换
        self.theme_action = QAction("切换深色主题", self)
        self.theme_action.setCheckable(True)
        self.theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 使用帮助
        help_action = QAction("使用帮助", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_model_page(self):
        """创建边坡建模页面"""
        page = QWidget()
        layout = QHBoxLayout(page)
        
        # 创建左侧参数设置区域
        param_group = QGroupBox("模型参数")
        param_layout = QVBoxLayout(param_group)
        
        # 边坡高度
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("边坡高度 (m):"))
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(1.0, 100.0)
        self.height_input.setValue(10.0)
        height_layout.addWidget(self.height_input)
        param_layout.addLayout(height_layout)
        
        # 边坡坡角
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("边坡坡角 (°):"))
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(10.0, 80.0)
        self.angle_input.setValue(45.0)
        angle_layout.addWidget(self.angle_input)
        param_layout.addLayout(angle_layout)
        
        # 边坡宽度
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("边坡宽度 (m):"))
        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(5.0, 100.0)
        self.width_input.setValue(20.0)
        width_layout.addWidget(self.width_input)
        param_layout.addLayout(width_layout)
        
        # 地基深度
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("地基深度 (m):"))
        self.depth_input = QDoubleSpinBox()
        self.depth_input.setRange(1.0, 20.0)
        self.depth_input.setValue(5.0)
        depth_layout.addWidget(self.depth_input)
        param_layout.addLayout(depth_layout)
        
        # 粒子半径
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("粒子半径 (m):"))
        self.radius_input = QDoubleSpinBox()
        self.radius_input.setRange(0.05, 1.0)
        self.radius_input.setValue(0.2)
        self.radius_input.setSingleStep(0.05)
        radius_layout.addWidget(self.radius_input)
        param_layout.addLayout(radius_layout)
        
        # 粒子类型
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("粒子类型:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["soil", "rock", "water", "boundary"])
        type_layout.addWidget(self.type_combo)
        param_layout.addLayout(type_layout)
        
        # 地貌类型
        landform_layout = QHBoxLayout()
        landform_layout.addWidget(QLabel("地貌类型:"))
        self.landform_combo = QComboBox()
        self.landform_combo.addItems(["flat", "undulating", "rocky", "sandy"])
        landform_layout.addWidget(self.landform_combo)
        param_layout.addLayout(landform_layout)
        
        # 天气条件
        weather_layout = QHBoxLayout()
        weather_layout.addWidget(QLabel("天气条件:"))
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(["dry", "wet", "rainy", "frozen"])
        weather_layout.addWidget(self.weather_combo)
        param_layout.addLayout(weather_layout)
        
        # 地下水位
        groundwater_layout = QHBoxLayout()
        groundwater_layout.addWidget(QLabel("地下水位 (m):"))
        self.groundwater_input = QDoubleSpinBox()
        self.groundwater_input.setRange(-5.0, 5.0)
        self.groundwater_input.setValue(0.0)
        self.groundwater_input.setSingleStep(0.5)
        groundwater_layout.addWidget(self.groundwater_input)
        param_layout.addLayout(groundwater_layout)
        
        # 植被密度
        vegetation_layout = QHBoxLayout()
        vegetation_layout.addWidget(QLabel("植被密度:"))
        self.vegetation_input = QDoubleSpinBox()
        self.vegetation_input.setRange(0.0, 1.0)
        self.vegetation_input.setValue(0.0)
        self.vegetation_input.setSingleStep(0.1)
        vegetation_layout.addWidget(self.vegetation_input)
        param_layout.addLayout(vegetation_layout)
        
        # 构建模型按钮
        self.build_button = QPushButton("构建边坡模型")
        self.build_button.clicked.connect(self.build_slope_model)
        param_layout.addWidget(self.build_button)
        
        # 模型信息显示
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setPlaceholderText("模型信息将显示在这里...")
        param_layout.addWidget(self.model_info)
        
        # 创建右侧模型预览区域
        preview_group = QGroupBox("模型预览")
        preview_layout = QVBoxLayout(preview_group)
        
        # 模型预览区域
        self.model_preview = QWidget()
        self.model_preview.setStyleSheet("background-color: #f0f0f0;")
        self.model_preview.setLayout(QVBoxLayout())  # 为预览窗口部件设置布局
        preview_layout.addWidget(self.model_preview)
        
        # 预览控制按钮
        preview_control_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("放大")
        self.zoom_out_button = QPushButton("缩小")
        self.reset_view_button = QPushButton("重置视图")
        
        preview_control_layout.addWidget(self.zoom_in_button)
        preview_control_layout.addWidget(self.zoom_out_button)
        preview_control_layout.addWidget(self.reset_view_button)
        preview_layout.addLayout(preview_control_layout)
        
        # 将左右区域添加到主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(param_group)
        splitter.addWidget(preview_group)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter)
        
        return page
    
    def create_simulation_page(self):
        """创建仿真控制页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 仿真参数设置
        param_group = QGroupBox("仿真参数设置")
        param_layout = QVBoxLayout(param_group)
        
        # 时间步长
        dt_layout = QHBoxLayout()
        dt_layout.addWidget(QLabel("时间步长 (s):"))
        self.dt_input = QDoubleSpinBox()
        self.dt_input.setRange(0.001, 0.1)
        self.dt_input.setValue(0.01)
        self.dt_input.setSingleStep(0.001)
        dt_layout.addWidget(self.dt_input)
        param_layout.addLayout(dt_layout)
        
        # 总时间步数
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("总时间步数:"))
        self.steps_input = QSpinBox()
        self.steps_input.setRange(10, 10000)
        self.steps_input.setValue(100)
        self.steps_input.setSingleStep(10)
        steps_layout.addWidget(self.steps_input)
        param_layout.addLayout(steps_layout)
        
        # 本构模型选择
        constitutive_layout = QHBoxLayout()
        constitutive_layout.addWidget(QLabel("本构模型:"))
        self.constitutive_combo = QComboBox()
        self.constitutive_combo.addItems(["弹性模型", "Mohr-Coulomb模型", "黏弹性模型", "Drucker-Prager模型", "Hyperelastic模型"])
        constitutive_layout.addWidget(self.constitutive_combo)
        param_layout.addLayout(constitutive_layout)
        
        # GPU加速选项
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("使用GPU加速:"))
        self.gpu_checkbox = QCheckBox()
        self.gpu_checkbox.setLayoutDirection(Qt.RightToLeft)  # 右对齐
        gpu_layout.addWidget(self.gpu_checkbox)
        param_layout.addLayout(gpu_layout)
        
        # 材料参数设置（展开选项）
        material_params_group = QGroupBox("材料参数")
        material_params_layout = QVBoxLayout(material_params_group)
        
        # 密度设置
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("密度 (kg/m³):"))
        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(1000.0, 3000.0)
        self.density_input.setValue(2600.0)
        self.density_input.setSingleStep(100.0)
        density_layout.addWidget(self.density_input)
        material_params_layout.addLayout(density_layout)
        
        # 弹性模量设置
        elastic_modulus_layout = QHBoxLayout()
        elastic_modulus_layout.addWidget(QLabel("弹性模量 (Pa):"))
        self.elastic_modulus_input = QDoubleSpinBox()
        self.elastic_modulus_input.setRange(1e6, 100e6)
        self.elastic_modulus_input.setValue(10e6)
        self.elastic_modulus_input.setSingleStep(1e6)
        self.elastic_modulus_input.setSuffix(" Pa")
        elastic_modulus_layout.addWidget(self.elastic_modulus_input)
        material_params_layout.addLayout(elastic_modulus_layout)
        
        # 泊松比设置
        poisson_ratio_layout = QHBoxLayout()
        poisson_ratio_layout.addWidget(QLabel("泊松比:"))
        self.poisson_ratio_input = QDoubleSpinBox()
        self.poisson_ratio_input.setRange(0.0, 0.5)
        self.poisson_ratio_input.setValue(0.3)
        self.poisson_ratio_input.setSingleStep(0.05)
        poisson_ratio_layout.addWidget(self.poisson_ratio_input)
        material_params_layout.addLayout(poisson_ratio_layout)
        
        # 黏聚力设置
        cohesion_layout = QHBoxLayout()
        cohesion_layout.addWidget(QLabel("黏聚力 (Pa):"))
        self.cohesion_input = QDoubleSpinBox()
        self.cohesion_input.setRange(0.0, 100e3)
        self.cohesion_input.setValue(10e3)
        self.cohesion_input.setSingleStep(1e3)
        self.cohesion_input.setSuffix(" Pa")
        cohesion_layout.addWidget(self.cohesion_input)
        material_params_layout.addLayout(cohesion_layout)
        
        # 内摩擦角设置
        friction_angle_layout = QHBoxLayout()
        friction_angle_layout.addWidget(QLabel("内摩擦角 (°):"))
        self.friction_angle_input = QDoubleSpinBox()
        self.friction_angle_input.setRange(0.0, 90.0)
        self.friction_angle_input.setValue(30.0)
        self.friction_angle_input.setSingleStep(1.0)
        self.friction_angle_input.setSuffix(" °")
        friction_angle_layout.addWidget(self.friction_angle_input)
        material_params_layout.addLayout(friction_angle_layout)
        
        param_layout.addWidget(material_params_group)
        
        # 仿真控制按钮
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("开始仿真")
        self.pause_button = QPushButton("暂停仿真")
        self.stop_button = QPushButton("停止仿真")
        
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        
        # 初始状态设置
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        param_layout.addLayout(control_layout)
        
        layout.addWidget(param_group)
        
        # 仿真进度和状态
        progress_group = QGroupBox("仿真进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 仿真监控区域
        monitor_group = QGroupBox("仿真监控")
        monitor_layout = QVBoxLayout(monitor_group)
        
        # 监控内容（占位符）
        self.monitor_text = QTextEdit()
        self.monitor_text.setReadOnly(True)
        self.monitor_text.setPlaceholderText("仿真过程监控信息将显示在这里...")
        monitor_layout.addWidget(self.monitor_text)
        
        layout.addWidget(monitor_group)
        
        return page
    
    def create_result_page(self):
        """创建结果分析页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 结果显示选项卡
        self.result_tab = QTabWidget()
        
        # 稳定性指标页面
        indices_page = QWidget()
        indices_layout = QVBoxLayout(indices_page)
        
        # 指标表格
        self.indices_table = QTableWidget(0, 2)
        self.indices_table.setHorizontalHeaderLabels(["指标名称", "数值"])
        self.indices_table.horizontalHeader().setStretchLastSection(True)
        indices_layout.addWidget(self.indices_table)
        
        # 稳定性评估报告
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        indices_layout.addWidget(self.report_text)
        
        self.result_tab.addTab(indices_page, "稳定性指标")
        
        # 动态可视化页面
        visualization_page = QWidget()
        visualization_layout = QVBoxLayout(visualization_page)
        
        # 创建可视化模块
        self.visualization_module = VisualizationModule(self)
        visualization_layout.addWidget(self.visualization_module)
        
        self.result_tab.addTab(visualization_page, "动态可视化")
        
        # 位移场分布页面
        displacement_page = QWidget()
        displacement_layout = QVBoxLayout(displacement_page)
        
        # 位移场预览区域
        self.displacement_preview = QWidget()
        self.displacement_preview.setStyleSheet("background-color: #f0f0f0;")
        self.displacement_preview.setLayout(QVBoxLayout())  # 为预览窗口部件设置布局
        displacement_layout.addWidget(self.displacement_preview)
        
        self.result_tab.addTab(displacement_page, "位移场分布")
        
        # 应力应变场页面
        stress_page = QWidget()
        stress_layout = QVBoxLayout(stress_page)
        
        # 应力场预览区域
        self.stress_preview = QWidget()
        self.stress_preview.setStyleSheet("background-color: #f0f0f0;")
        self.stress_preview.setLayout(QVBoxLayout())  # 为预览窗口部件设置布局
        stress_layout.addWidget(self.stress_preview)
        
        self.result_tab.addTab(stress_page, "应力应变场")
        
        layout.addWidget(self.result_tab)
        
        # 结果导出按钮
        export_layout = QHBoxLayout()
        self.export_result_button = QPushButton("导出结果")
        self.export_result_button.clicked.connect(self.export_results)
        export_layout.addWidget(self.export_result_button)
        
        self.export_report_button = QPushButton("导出评估报告")
        self.export_report_button.clicked.connect(self.export_report)
        export_layout.addWidget(self.export_report_button)
        
        self.export_video_button = QPushButton("导出动画")
        self.export_video_button.clicked.connect(self.export_video)
        export_layout.addWidget(self.export_video_button)
        
        layout.addLayout(export_layout)
        
        return page
    
    def create_extension_page(self):
        """创建拓展功能页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 初始化模板管理器
        self.template_manager = ParameterTemplateManager()
        self.batch_manager = BatchSimulationManager()
        
        # 参数模板库
        template_group = QGroupBox("参数模板库")
        template_layout = QVBoxLayout(template_group)
        
        # 模板列表
        self.template_list = QTableWidget(0, 3)
        self.template_list.setHorizontalHeaderLabels(["模板名称", "岩性", "工况"])
        template_layout.addWidget(self.template_list)
        
        # 模板操作按钮
        template_button_layout = QHBoxLayout()
        self.save_template_button = QPushButton("保存当前参数为模板")
        self.save_template_button.clicked.connect(self.save_current_template)
        
        self.load_template_button = QPushButton("加载模板")
        self.load_template_button.clicked.connect(self.load_selected_template)
        
        self.delete_template_button = QPushButton("删除模板")
        self.delete_template_button.clicked.connect(self.delete_selected_template)
        
        template_button_layout.addWidget(self.save_template_button)
        template_button_layout.addWidget(self.load_template_button)
        template_button_layout.addWidget(self.delete_template_button)
        
        template_layout.addLayout(template_button_layout)
        
        layout.addWidget(template_group)
        
        # 批量仿真
        batch_group = QGroupBox("批量仿真")
        batch_layout = QVBoxLayout(batch_group)
        
        # 批量仿真参数设置
        batch_param_layout = QHBoxLayout()
        
        # 参数范围设置
        param_range_layout = QVBoxLayout()
        param_range_layout.addWidget(QLabel("参数范围设置:"))
        
        # 示例参数范围设置
        self.batch_params = {
            'slope_angle': QDoubleSpinBox(),
            'particle_radius': QDoubleSpinBox()
        }
        
        # 坡角范围
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("坡角范围 (°):"))
        self.batch_params['slope_angle'].setRange(20.0, 80.0)
        self.batch_params['slope_angle'].setValue(45.0)
        angle_layout.addWidget(self.batch_params['slope_angle'])
        param_range_layout.addLayout(angle_layout)
        
        # 粒子半径范围
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("粒子半径范围 (m):"))
        self.batch_params['particle_radius'].setRange(0.05, 0.5)
        self.batch_params['particle_radius'].setValue(0.2)
        self.batch_params['particle_radius'].setSingleStep(0.05)
        radius_layout.addWidget(self.batch_params['particle_radius'])
        param_range_layout.addLayout(radius_layout)
        
        batch_param_layout.addLayout(param_range_layout)
        
        # 批量设置说明
        batch_info = QTextEdit()
        batch_info.setReadOnly(True)
        batch_info.setPlainText("批量仿真说明:\n1. 设置基础参数\n2. 设置参数范围\n3. 点击创建批量任务\n4. 点击运行批量仿真\n5. 查看批量结果")
        batch_param_layout.addWidget(batch_info)
        
        batch_layout.addLayout(batch_param_layout)
        
        # 批量仿真控制按钮
        batch_button_layout = QHBoxLayout()
        self.create_batch_button = QPushButton("创建批量任务")
        self.create_batch_button.clicked.connect(self.create_batch_task)
        
        self.run_batch_button = QPushButton("运行批量仿真")
        self.run_batch_button.clicked.connect(self.run_batch_simulation)
        
        self.view_batch_result_button = QPushButton("查看批量结果")
        self.view_batch_result_button.clicked.connect(self.view_batch_results)
        
        self.export_batch_result_button = QPushButton("导出批量结果")
        self.export_batch_result_button.clicked.connect(self.export_batch_results)
        
        batch_button_layout.addWidget(self.create_batch_button)
        batch_button_layout.addWidget(self.run_batch_button)
        batch_button_layout.addWidget(self.view_batch_result_button)
        batch_button_layout.addWidget(self.export_batch_result_button)
        
        batch_layout.addLayout(batch_button_layout)
        
        layout.addWidget(batch_group)
        
        # 批量结果显示
        self.batch_result_table = QTableWidget(0, 4)
        self.batch_result_table.setHorizontalHeaderLabels(["任务ID", "安全系数", "最大位移 (m)", "状态"])
        self.batch_result_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_result_table)
        
        # 加载初始模板数据
        self._load_template_list()
        
        return page
    
    def build_slope_model(self):
        """构建边坡模型"""
        try:
            # 获取参数
            slope_height = self.height_input.value()
            slope_angle = self.angle_input.value()
            slope_width = self.width_input.value()
            ground_depth = self.depth_input.value()
            particle_radius = self.radius_input.value()
            particle_type = self.type_combo.currentText()
            landform_type = self.landform_combo.currentText()
            weather_condition = self.weather_combo.currentText()
            groundwater_level = self.groundwater_input.value()
            vegetation_density = self.vegetation_input.value()
            
            # 参数校验
            if not self._validate_model_params(slope_height, slope_angle, slope_width, ground_depth, particle_radius):
                return
            
            # 构建模型
            self.particles = self.slope_builder.build_parametric_slope(
                slope_height=slope_height,
                slope_angle=slope_angle,
                slope_width=slope_width,
                ground_depth=ground_depth,
                particle_radius=particle_radius,
                particle_type=particle_type,
                landform_type=landform_type,
                weather_condition=weather_condition,
                groundwater_level=groundwater_level,
                vegetation_density=vegetation_density
            )
            
            self.initial_particles = self.particles.copy()
            
            # 更新模型信息
            model_info = f"""边坡模型构建成功！
模型参数：
- 边坡高度: {slope_height} m
- 边坡坡角: {slope_angle} °
- 边坡宽度: {slope_width} m
- 地基深度: {ground_depth} m
- 粒子半径: {particle_radius} m
- 粒子类型: {particle_type}
- 地貌类型: {landform_type}
- 天气条件: {weather_condition}
- 地下水位: {groundwater_level} m
- 植被密度: {vegetation_density}
- 粒子数量: {len(self.particles)}
"""
            
            self.model_info.setText(model_info)
            self.statusBar.showMessage(f"边坡模型构建成功，粒子数量: {len(self.particles)}")
            
            # 更新预览区域（简化实现）
            self.update_model_preview()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"构建模型失败: {str(e)}")
            self.statusBar.showMessage("构建模型失败")
    
    def _validate_model_params(self, height, angle, width, depth, radius):
        """参数校验"""
        errors = []
        
        if height <= 0 or height > 100:
            errors.append("边坡高度必须在0到100米之间")
        
        if angle <= 0 or angle >= 90:
            errors.append("边坡坡角必须在0到90度之间")
        
        if width <= 0 or width > 200:
            errors.append("边坡宽度必须在0到200米之间")
        
        if depth <= 0 or depth > 50:
            errors.append("地基深度必须在0到50米之间")
        
        if radius <= 0 or radius > 1.0:
            errors.append("粒子半径必须在0到1.0米之间")
        
        if errors:
            QMessageBox.warning(self, "参数错误", "\n".join(errors))
            return False
        
        return True
    
    def _validate_simulation_params(self, time_step, total_steps):
        """仿真参数校验"""
        errors = []
        
        if time_step <= 0 or time_step > 1.0:
            errors.append("时间步长必须在0到1.0秒之间")
        
        if total_steps <= 0 or total_steps > 10000:
            errors.append("总时间步数必须在0到10000之间")
        
        if errors:
            QMessageBox.warning(self, "参数错误", "\n".join(errors))
            return False
        
        return True
    
    def update_model_preview(self):
        """更新模型预览"""
        if not hasattr(self, 'preview_canvas'):
            # 创建预览画布
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            self.preview_fig = Figure(figsize=(5, 4), dpi=100)
            self.preview_ax = self.preview_fig.add_subplot(111, projection='3d')  # 创建3D坐标轴
            self.preview_canvas = FigureCanvas(self.preview_fig)
            
            # 移除之前的占位符
            for i in reversed(range(self.model_preview.layout().count())):
                widget = self.model_preview.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            
            # 添加画布到布局
            self.model_preview.layout().addWidget(self.preview_canvas)
        
        # 清除现有图形
        self.preview_ax.clear()
        
        if self.particles:
            # 绘制边坡模型
            positions = np.array([p['position'] for p in self.particles])  # 获取所有三个坐标
            self.preview_ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=5, alpha=0.8, c='brown')
            
            self.preview_ax.set_title('边坡模型三维预览')
            self.preview_ax.set_xlabel('X (m)')
            self.preview_ax.set_ylabel('Y (m)')
            self.preview_ax.set_zlabel('Z (m)')
            
            # 设置合适的视角
            self.preview_ax.view_init(elev=30, azim=45)
            self.preview_ax.grid(True)
        else:
            self.preview_ax.text(0.5, 0.5, 0.5, '无模型数据', ha='center', va='center', transform=self.preview_ax.transAxes)
        
        # 刷新画布
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()
    
    def start_simulation(self):
        """开始仿真"""
        if not self.particles:
            QMessageBox.warning(self, "警告", "请先构建边坡模型！")
            return
        
        # 清除之前的可视化数据
        self.clear_visualization()
        
        # 获取材料参数
        material_params = {
            'density': self.density_input.value(),
            'elastic_modulus': self.elastic_modulus_input.value(),
            'poisson_ratio': self.poisson_ratio_input.value(),
            'cohesion': self.cohesion_input.value(),
            'friction_angle': self.friction_angle_input.value()
        }
        
        # 更新粒子的材料参数
        updated_particles = []
        for p in self.particles:
            updated_p = p.copy()
            updated_p['material_params'].update(material_params)
            updated_p['density'] = material_params['density']
            updated_p['mass'] = (4/3)*np.pi*updated_p['radius']**3 * material_params['density']
            updated_particles.append(updated_p)
        
        # 设置仿真参数
        simulation_params = {
            'total_steps': self.steps_input.value(),
            'time_step': self.dt_input.value(),
            'constitutive_model': self.constitutive_combo.currentText(),
            'material_params': material_params,
            'use_gpu': self.gpu_checkbox.isChecked()  # 添加GPU加速选项
        }
        
        # 创建并启动仿真线程
        self.simulation_thread = SimulationThread(updated_particles, simulation_params)
        self.simulation_thread.progress_update.connect(self.update_progress)
        self.simulation_thread.simulation_finished.connect(self.simulation_finished)
        self.simulation_thread.status_update.connect(self.update_status)
        
        # 连接粒子更新信号到可视化模块
        self.simulation_thread.particle_update.connect(self.visualization_module.update_particles)
        
        # 启动线程
        self.simulation_thread.start()
        
        # 更新按钮状态
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("暂停仿真")
        self.stop_button.setEnabled(True)
        
        # 更新状态栏
        if self.gpu_checkbox.isChecked():
            self.statusBar.showMessage("仿真进行中... (使用GPU加速)")
        else:
            self.statusBar.showMessage("仿真进行中...")
    
    def pause_simulation(self):
        """暂停仿真"""
        if self.simulation_thread:
            self.simulation_thread.pause()
            
            # 更新按钮状态
            self.pause_button.setText("继续仿真")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.resume_simulation)
    
    def resume_simulation(self):
        """继续仿真"""
        if self.simulation_thread:
            self.simulation_thread.resume()
            
            # 更新按钮状态
            self.pause_button.setText("暂停仿真")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_simulation)
    
    def stop_simulation(self):
        """停止仿真"""
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()
            self.simulation_thread = None
            
            # 重置按钮状态
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("暂停仿真")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_simulation)
            self.stop_button.setEnabled(False)
    
    def update_progress(self, progress):
        """更新仿真进度"""
        self.progress_bar.setValue(progress)
    
    def update_status(self, status):
        """更新仿真状态"""
        self.status_label.setText(status)
        self.monitor_text.append(status)
        self.statusBar.showMessage(status)
    
    def simulation_finished(self, result):
        """仿真完成处理"""
        if 'error' in result:
            QMessageBox.critical(self, "仿真错误", result['error'])
        else:
            # 保存结果
            self.deformed_particles = result['deformed_particles']
            self.stability_indices = result['stability_indices']
            
            # 更新结果显示
            self.update_result_display()
            
            QMessageBox.information(self, "仿真完成", "边坡仿真计算已完成！")
        
        # 重置按钮状态
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停仿真")
        self.pause_button.clicked.disconnect()
        self.pause_button.clicked.connect(self.pause_simulation)
        self.stop_button.setEnabled(False)
    
    def update_result_display(self):
        """更新结果显示"""
        # 更新指标表格
        self.indices_table.setRowCount(0)
        
        indices = {
            "安全系数": self.stability_indices.get('safety_factor', 0.0),
            "最大位移 (m)": self.stability_indices.get('max_displacement', 0.0),
            "平均位移 (m)": self.stability_indices.get('average_displacement', 0.0),
            "失稳预警等级": self.stability_indices.get('instability_warning_level', 0)
        }
        
        for i, (name, value) in enumerate(indices.items()):
            self.indices_table.insertRow(i)
            self.indices_table.setItem(i, 0, QTableWidgetItem(name))
            self.indices_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
        
        # 更新报告
        with open('slope_stability_report.txt', 'r', encoding='utf-8') as f:
            report_content = f.read()
            self.report_text.setText(report_content)
        
        # 更新位移场预览
        self._update_displacement_preview()
        
        # 更新应力场预览
        self._update_stress_preview()
    
    def _update_displacement_preview(self):
        """更新位移场预览"""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        # 确保有变形后的粒子数据
        if not hasattr(self, 'deformed_particles') or not self.deformed_particles:
            return
        
        # 检查是否已有画布
        if not hasattr(self, 'displacement_canvas'):
            # 创建画布
            self.displacement_fig = Figure(figsize=(8, 6), dpi=100)
            self.displacement_ax = self.displacement_fig.add_subplot(111)
            self.displacement_canvas = FigureCanvas(self.displacement_fig)
            
            # 移除之前的占位符
            for i in reversed(range(self.displacement_preview.layout().count())):
                widget = self.displacement_preview.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            
            # 添加画布到布局
            self.displacement_preview.layout().addWidget(self.displacement_canvas)
        
        # 清除现有图形
        self.displacement_ax.clear()
        
        # 提取数据
        positions = np.array([p['position'] for p in self.deformed_particles])
        
        # 计算位移
        if hasattr(self, 'initial_particles') and self.initial_particles:
            initial_positions = np.array([p['position'] for p in self.initial_particles])
            displacements = positions - initial_positions
            displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        else:
            # 没有初始粒子数据时，使用位置的y坐标作为替代
            displacement_magnitudes = positions[:, 1]
        
        # 绘制位移场
        scatter = self.displacement_ax.scatter(positions[:, 0], positions[:, 1], 
                                              c=displacement_magnitudes, cmap='viridis', s=10, alpha=0.8)
        cbar = self.displacement_fig.colorbar(scatter, ax=self.displacement_ax, label='位移 (m)')
        
        # 设置标题和标签
        self.displacement_ax.set_title('位移场分布')
        self.displacement_ax.set_xlabel('X (m)')
        self.displacement_ax.set_ylabel('Y (m)')
        self.displacement_ax.set_aspect('equal')
        self.displacement_ax.grid(True)
        
        # 调整布局
        self.displacement_fig.tight_layout()
        
        # 更新画布
        self.displacement_canvas.draw()
    
    def _update_stress_preview(self):
        """更新应力场预览"""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        # 确保有变形后的粒子数据
        if not hasattr(self, 'deformed_particles') or not self.deformed_particles:
            return
        
        # 检查是否已有画布
        if not hasattr(self, 'stress_canvas'):
            # 创建画布
            self.stress_fig = Figure(figsize=(8, 6), dpi=100)
            self.stress_ax = self.stress_fig.add_subplot(111)
            self.stress_canvas = FigureCanvas(self.stress_fig)
            
            # 移除之前的占位符
            for i in reversed(range(self.stress_preview.layout().count())):
                widget = self.stress_preview.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            
            # 添加画布到布局
            self.stress_preview.layout().addWidget(self.stress_canvas)
        
        # 清除现有图形
        self.stress_ax.clear()
        
        # 提取数据
        positions = np.array([p['position'] for p in self.deformed_particles])
        
        # 计算应力
        if 'stress' in self.deformed_particles[0]:
            # 计算最大主应力
            max_stresses = []
            for p in self.deformed_particles:
                sigma = p['stress']
                eigenvalues, _ = np.linalg.eigh(sigma)
                max_principal_stress = np.max(eigenvalues)
                max_stresses.append(max_principal_stress)
            max_stresses = np.array(max_stresses)
        else:
            # 没有应力数据时，使用位置的y坐标作为替代
            max_stresses = positions[:, 1]
        
        # 绘制应力场
        scatter = self.stress_ax.scatter(positions[:, 0], positions[:, 1], 
                                        c=max_stresses, cmap='plasma', s=10, alpha=0.8)
        cbar = self.stress_fig.colorbar(scatter, ax=self.stress_ax, label='最大主应力 (Pa)')
        
        # 设置标题和标签
        self.stress_ax.set_title('应力场分布')
        self.stress_ax.set_xlabel('X (m)')
        self.stress_ax.set_ylabel('Y (m)')
        self.stress_ax.set_aspect('equal')
        self.stress_ax.grid(True)
        
        # 调整布局
        self.stress_fig.tight_layout()
        
        # 更新画布
        self.stress_canvas.draw()
    
    def export_results(self):
        """导出结果"""
        # 简化实现
        QMessageBox.information(self, "导出结果", "结果导出功能将在后续版本中实现！")
    
    def export_report(self):
        """导出评估报告"""
        # 简化实现
        QMessageBox.information(self, "导出报告", "报告导出功能将在后续版本中实现！")
    
    def export_video(self):
        """导出动画"""
        # 调用可视化模块的导出视频功能
        self.visualization_module._export_video()
    
    def clear_visualization(self):
        """清除可视化数据"""
        if hasattr(self, 'visualization_module'):
            self.visualization_module.clear_data()
    
    def new_project(self):
        """新建项目"""
        # 简化实现
        QMessageBox.information(self, "新建项目", "新建项目功能将在后续版本中实现！")
    
    def open_project(self):
        """打开项目"""
        # 简化实现
        QMessageBox.information(self, "打开项目", "打开项目功能将在后续版本中实现！")
    
    def save_project(self):
        """保存项目"""
        # 简化实现
        QMessageBox.information(self, "保存项目", "保存项目功能将在后续版本中实现！")
    
    def save_project_as(self):
        """保存项目为"""
        # 简化实现
        QMessageBox.information(self, "保存项目为", "保存项目为功能将在后续版本中实现！")
    
    def export_model(self):
        """导出模型"""
        # 简化实现
        QMessageBox.information(self, "导出模型", "导出模型功能将在后续版本中实现！")
    
    def import_model(self):
        """导入模型"""
        # 简化实现
        QMessageBox.information(self, "导入模型", "导入模型功能将在后续版本中实现！")
    
    def toggle_theme(self):
        """切换主题"""
        # 简化实现
        if self.theme_action.isChecked():
            # 切换到深色主题
            self.setStyleSheet("background-color: #333333; color: #ffffff;")
        else:
            # 切换到浅色主题
            self.setStyleSheet("")
    
    def show_help(self):
        """显示帮助"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout
        
        # 创建帮助对话框
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("使用帮助")
        help_dialog.setMinimumSize(800, 600)
        
        # 创建主布局
        main_layout = QVBoxLayout(help_dialog)
        
        # 创建文本浏览器
        text_browser = QTextBrowser()
        
        # 帮助内容
        help_content = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; font-size: 14px; }
                h1 { color: #2563eb; font-size: 18px; margin-top: 20px; margin-bottom: 10px; }
                h2 { color: #1d4ed8; font-size: 16px; margin-top: 15px; margin-bottom: 8px; }
                p { margin: 5px 0; }
                ul { margin: 5px 0; padding-left: 20px; }
                li { margin: 3px 0; }
                .term { font-weight: bold; color: #1e40af; }
                .section { margin-bottom: 15px; }
            </style>
        </head>
        <body>
            <h1>粒子法岩土边坡仿真软件使用帮助</h1>
            
            <div class="section">
                <h2>一、软件简介</h2>
                <p>本软件基于粒子法（SPH/PFC）实现岩土边坡稳定性仿真分析，可用于边坡工程设计、稳定性评估和灾害预警。</p>
            </div>
            
            <div class="section">
                <h2>二、主要功能</h2>
                <ul>
                    <li><strong>边坡建模</strong>：参数化构建边坡模型，支持不同地貌和天气条件</li>
                    <li><strong>仿真控制</strong>：设置仿真参数，支持CPU/GPU加速</li>
                    <li><strong>结果分析</strong>：查看稳定性指标、位移场和应力应变场</li>
                    <li><strong>拓展功能</strong>：参数模板管理、批量仿真</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>三、使用流程</h2>
                <ol>
                    <li>在<strong>边坡建模</strong>页面设置模型参数，点击"构建边坡模型"</li>
                    <li>在<strong>仿真控制</strong>页面设置仿真参数，点击"开始仿真"</li>
                    <li>在<strong>结果分析</strong>页面查看仿真结果</li>
                    <li>可选：在<strong>拓展功能</strong>页面保存参数模板或进行批量仿真</li>
                </ol>
            </div>
            
            <div class="section">
                <h2>四、关键术语解释</h2>
                <ul>
                    <li><span class="term">粒子法（Particle Method）</span>：一种无网格数值方法，将连续介质离散为大量粒子，通过粒子间的相互作用模拟物理现象</li>
                    <li><span class="term">SPH（Smoothed Particle Hydrodynamics）</span>：光滑粒子流体动力学，常用于模拟流体和可变形固体</li>
                    <li><span class="term">PFC（Particle Flow Code）</span>：粒子流代码，常用于模拟颗粒材料的力学行为</li>
                    <li><span class="term">安全系数</span>：边坡抗滑力与下滑力的比值，用于评估边坡稳定性，一般要求大于1.2</li>
                    <li><span class="term">本构模型</span>：描述材料应力-应变关系的数学模型，本软件支持弹性、Mohr-Coulomb、黏弹性等多种模型</li>
                    <li><span class="term">位移场</span>：边坡中各点的位移分布情况，用于分析边坡变形特征</li>
                    <li><span class="term">应力应变场</span>：边坡中各点的应力和应变分布情况，用于分析边坡受力状态</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>五、常见问题</h2>
                <ul>
                    <li><strong>问题1：模型预览不显示？</strong><br>解决方案：请确保已点击"构建边坡模型"按钮，生成了模型数据。</li>
                    <li><strong>问题2：仿真动画显示不全？</strong><br>解决方案：软件会自动调整坐标轴范围，确保所有粒子都在视图范围内。</li>
                    <li><strong>问题3：页面放缩时字体显示不全？</strong><br>解决方案：软件已支持高DPI缩放，可在系统设置中调整显示缩放比例。</li>
                    <li><strong>问题4：GPU加速不起作用？</strong><br>解决方案：请确保已安装CuPy库，软件会在缺少CuPy时自动回退到CPU计算。</li>
                    <li><strong>问题5：图表文字乱码？</strong><br>解决方案：软件已设置支持中文的字体，如仍有问题，请检查系统字体设置。</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>六、软件操作说明</h2>
                
                <h3>6.1 边坡建模</h3>
                <ul>
                    <li><strong>边坡高度</strong>：边坡的垂直高度，单位为米</li>
                    <li><strong>边坡坡角</strong>：边坡的倾斜角度，单位为度</li>
                    <li><strong>边坡宽度</strong>：边坡的水平宽度，单位为米</li>
                    <li><strong>地基深度</strong>：边坡下方地基的深度，单位为米</li>
                    <li><strong>粒子半径</strong>：粒子的半径，影响模型精度和计算效率</li>
                    <li><strong>粒子类型</strong>：可选土壤、岩石、水或边界</li>
                    <li><strong>地貌类型</strong>：可选平坦、起伏、岩石或砂质</li>
                    <li><strong>天气条件</strong>：可选干燥、湿润、雨天或冻结</li>
                </ul>
                
                <h3>6.2 仿真控制</h3>
                <ul>
                    <li><strong>时间步长</strong>：仿真的时间步长，单位为秒，一般取0.001-0.01</li>
                    <li><strong>总时间步数</strong>：仿真的总步数，影响仿真时长和精度</li>
                    <li><strong>本构模型</strong>：选择材料的本构模型</li>
                    <li><strong>材料参数</strong>：设置材料的密度、弹性模量、泊松比、黏聚力和内摩擦角</li>
                    <li><strong>使用GPU加速</strong>：勾选可启用GPU加速，提高计算速度</li>
                </ul>
                
                <h3>6.3 结果分析</h3>
                <ul>
                    <li><strong>稳定性指标</strong>：显示安全系数、最大位移、平均位移和失稳预警等级</li>
                    <li><strong>动态可视化</strong>：实时显示仿真过程，可选择显示位移、速度、应力或应变</li>
                    <li><strong>位移场分布</strong>：显示边坡的位移分布情况</li>
                    <li><strong>应力应变场</strong>：显示边坡的应力和应变分布情况</li>
                </ul>
                
                <h3>6.4 拓展功能</h3>
                <ul>
                    <li><strong>参数模板库</strong>：保存和加载常用参数组合</li>
                    <li><strong>批量仿真</strong>：生成多组参数组合，批量进行仿真计算</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>七、技术支持</h2>
                <p>如果您在使用过程中遇到问题或有改进建议，欢迎联系技术支持团队。</p>
            </div>
        </body>
        </html>
        """
        
        text_browser.setHtml(help_content)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(help_dialog.accept)
        button_layout.addWidget(close_button)
        
        # 添加到主布局
        main_layout.addWidget(text_browser)
        main_layout.addLayout(button_layout)
        
        # 显示对话框
        help_dialog.exec_()
    
    def show_about(self):
        """显示关于"""
        QMessageBox.information(self, "关于", "粒子法岩土边坡仿真软件\n版本: 1.0.0\n© 2026 粒子法仿真软件团队")
    
    def _load_template_list(self):
        """加载模板列表"""
        templates = self.template_manager.get_templates()
        self.template_list.setRowCount(len(templates))
        
        for i, template in enumerate(templates):
            self.template_list.setItem(i, 0, QTableWidgetItem(template['name']))
            self.template_list.setItem(i, 1, QTableWidgetItem(template['lithology']))
            self.template_list.setItem(i, 2, QTableWidgetItem(template['condition']))
    
    def save_current_template(self):
        """保存当前参数为模板"""
        # 获取当前参数
        current_params = {
            'slope_height': self.height_input.value(),
            'slope_angle': self.angle_input.value(),
            'slope_width': self.width_input.value(),
            'ground_depth': self.depth_input.value(),
            'particle_radius': self.radius_input.value(),
            'particle_type': self.type_combo.currentText()
        }
        
        # 弹出对话框输入模板信息
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        
        # 获取模板名称
        template_name, ok = QInputDialog.getText(self, "保存模板", "请输入模板名称:")
        if not ok or not template_name:
            return
        
        # 获取岩性
        lithology, ok = QInputDialog.getText(self, "保存模板", "请输入岩性:")
        if not ok:
            return
        
        # 获取工况
        condition, ok = QInputDialog.getText(self, "保存模板", "请输入工况:")
        if not ok:
            return
        
        # 创建模板
        template = {
            'name': template_name,
            'lithology': lithology,
            'condition': condition,
            'params': current_params
        }
        
        # 保存模板
        success = self.template_manager.save_template(template)
        if success:
            QMessageBox.information(self, "成功", f"模板 '{template_name}' 保存成功！")
            self._load_template_list()
        else:
            QMessageBox.error(self, "错误", f"保存模板失败！")
    
    def load_selected_template(self):
        """加载选中的模板"""
        selected_items = self.template_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个模板！")
            return
        
        # 获取选中模板的名称
        template_name = selected_items[0].text()
        
        # 加载模板
        template = self.template_manager.load_template(template_name)
        if template:
            # 设置参数
            params = template['params']
            if 'slope_height' in params:
                self.height_input.setValue(params['slope_height'])
            if 'slope_angle' in params:
                self.angle_input.setValue(params['slope_angle'])
            if 'slope_width' in params:
                self.width_input.setValue(params['slope_width'])
            if 'ground_depth' in params:
                self.depth_input.setValue(params['ground_depth'])
            if 'particle_radius' in params:
                self.radius_input.setValue(params['particle_radius'])
            if 'particle_type' in params:
                self.type_combo.setCurrentText(params['particle_type'])
            
            QMessageBox.information(self, "成功", f"模板 '{template_name}' 加载成功！")
        else:
            QMessageBox.error(self, "错误", f"加载模板失败！")
    
    def delete_selected_template(self):
        """删除选中的模板"""
        selected_items = self.template_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个模板！")
            return
        
        # 获取选中模板的名称
        template_name = selected_items[0].text()
        
        # 确认删除
        reply = QMessageBox.question(self, "确认删除", f"确定要删除模板 '{template_name}' 吗？",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            success = self.template_manager.delete_template(template_name)
            if success:
                QMessageBox.information(self, "成功", f"模板 '{template_name}' 删除成功！")
                self._load_template_list()
            else:
                QMessageBox.error(self, "错误", f"删除模板失败！")
    
    def create_batch_task(self):
        """创建批量任务"""
        # 简化实现，创建批量任务
        base_params = {
            'slope_height': 10.0,
            'slope_width': 20.0,
            'ground_depth': 5.0,
            'particle_type': 'soil'
        }
        
        # 参数范围
        param_ranges = {
            'slope_angle': [30.0, 45.0, 60.0],
            'particle_radius': [0.1, 0.2, 0.3]
        }
        
        tasks = self.batch_manager.create_batch_task(base_params, param_ranges)
        QMessageBox.information(self, "成功", f"已创建 {len(tasks)} 个批量仿真任务！")
    
    def run_batch_simulation(self):
        """运行批量仿真"""
        results = self.batch_manager.run_batch_simulation()
        QMessageBox.information(self, "成功", f"批量仿真完成，共 {len(results)} 个任务！")
        self._update_batch_result_table(results)
    
    def view_batch_results(self):
        """查看批量结果"""
        results = self.batch_manager.get_results()
        if not results:
            QMessageBox.warning(self, "警告", "没有批量仿真结果！")
            return
        
        self._update_batch_result_table(results)
    
    def _update_batch_result_table(self, results):
        """更新批量结果表格"""
        self.batch_result_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            self.batch_result_table.setItem(i, 0, QTableWidgetItem(str(result['task_id'])))
            self.batch_result_table.setItem(i, 1, QTableWidgetItem(f"{result['safety_factor']:.3f}"))
            self.batch_result_table.setItem(i, 2, QTableWidgetItem(f"{result['max_displacement']:.6f}"))
            self.batch_result_table.setItem(i, 3, QTableWidgetItem(result['status']))
    
    def export_batch_results(self):
        """导出批量结果"""
        results = self.batch_manager.get_results()
        if not results:
            QMessageBox.warning(self, "警告", "没有批量仿真结果可导出！")
            return
        
        # 保存文件
        file_path, _ = QFileDialog.getSaveFileName(self, "导出批量结果", "batch_simulation_results.csv", "CSV Files (*.csv);;JSON Files (*.json)")
        if file_path:
            format = 'csv' if file_path.endswith('.csv') else 'json'
            self.batch_manager.export_results(file_path, format)
            QMessageBox.information(self, "成功", f"批量结果已导出到 {file_path}！")
    
    def resizeEvent(self, event):
        """窗口大小变化事件处理"""
        super().resizeEvent(event)
        
        # 当窗口大小变化时，更新模型预览
        if hasattr(self, 'particles') and self.particles:
            self.update_model_preview()
        
        # 确保可视化模块适应新的窗口大小
        if hasattr(self, 'visualization_module'):
            self.visualization_module._update_visualization()

if __name__ == "__main__":
    # 添加高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # 设置应用程序风格
    app.setStyle("Fusion")
    
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec_())
