#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子法岩土边坡仿真软件演示脚本

这个脚本演示了粒子法岩土边坡仿真软件的核心功能，包括：
1. 边坡模型构建
2. 稳定性评估
3. 结果可视化

使用Python实现，不需要编译C++代码
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from mpl_toolkits.mplot3d import Axes3D
import os

class SlopeModelBuilder:
    """边坡模型构建器类"""
    
    def __init__(self):
        self.particles = []
        self.initial_particles = []
        self.model_info = {}
    
    def build_parametric_slope(self, slope_height=10.0, slope_angle=45.0, 
                              slope_width=20.0, ground_depth=5.0, particle_radius=0.2,
                              particle_type='soil', material_params=None, 
                              landform_type='flat', weather_condition='dry',
                              groundwater_level=0.0, vegetation_density=0.0):
        """
        参数化构建边坡模型
        
        参数：
        - slope_height: 边坡高度 (m)
        - slope_angle: 边坡坡角 (度)
        - slope_width: 边坡宽度 (m)
        - ground_depth: 地基深度 (m)
        - particle_radius: 粒子半径 (m)
        - particle_type: 粒子类型 (soil, rock, water, boundary)
        - material_params: 材料参数字典
        - landform_type: 地貌类型 (flat, undulating, rocky, sandy)
        - weather_condition: 天气条件 (dry, wet, rainy, frozen)
        - groundwater_level: 地下水位 (m)
        - vegetation_density: 植被密度 (0.0-1.0)
        
        返回：
        - 粒子列表，每个粒子是一个包含位置、速度、密度等属性的字典
        """
        
        print(f"构建边坡模型：高度={slope_height}m, 坡角={slope_angle}度, 宽度={slope_width}m")
        
        # 计算边坡边界
        slope_rad = np.radians(slope_angle)
        slope_ratio = np.tan(slope_rad)
        
        # 生成网格点
        spacing = particle_radius * 2.0
        min_x = -slope_width / 2.0
        max_x = slope_width / 2.0
        min_y = -ground_depth
        max_y = slope_height
        
        # 添加Z轴范围，生成真正的3D模型
        min_z = -slope_width / 4.0
        max_z = slope_width / 4.0
        
        # 生成3D粒子
        particles = []
        particle_id = 0
        
        # 设置默认材料参数
        default_params = {
            'soil': {'density': 2600.0, 'elastic_modulus': 10e6, 'poisson_ratio': 0.3, 'cohesion': 10e3, 'friction_angle': 30.0},
            'rock': {'density': 2800.0, 'elastic_modulus': 50e6, 'poisson_ratio': 0.25, 'cohesion': 50e3, 'friction_angle': 45.0},
            'water': {'density': 1000.0, 'elastic_modulus': 2.2e9, 'poisson_ratio': 0.5},
            'boundary': {'density': 2800.0, 'elastic_modulus': 100e6, 'poisson_ratio': 0.2}
        }
        
        material_param = default_params.get(particle_type, default_params['soil'])
        if material_params:
            material_param.update(material_params)
        
        # 3D粒子生成循环
        for x in np.arange(min_x, max_x, spacing):
            for y in np.arange(min_y, max_y, spacing):
                for z in np.arange(min_z, max_z, spacing):
                    # 检查是否在边坡区域内
                    if y < 0.0 or y > slope_height:
                        continue
                    
                    if x > 0.0:
                        slope_y = slope_height - x * slope_ratio
                        if y > slope_y:
                            continue
                    
                    # 创建粒子
                    particle = {
                        'id': particle_id,
                        'position': np.array([x, y, z]),
                        'velocity': np.array([0.0, 0.0, 0.0]),
                        'acceleration': np.array([0.0, -9.81, 0.0]),
                        'mass': (4/3)*np.pi*particle_radius**3 * material_param['density'],
                        'density': material_param['density'],
                        'radius': particle_radius,
                        'type': particle_type,
                        'stress': np.zeros((3, 3)),
                        'strain': np.zeros((3, 3)),
                        'displacement': np.zeros(3),
                        'material_params': material_param.copy()
                    }
                    
                    particles.append(particle)
                    particle_id += 1
        
        self.particles = particles.copy()
        self.initial_particles = particles.copy()
        
        # 保存模型信息
        self.model_info = {
            'model_type': 'parametric',
            'slope_height': slope_height,
            'slope_angle': slope_angle,
            'slope_width': slope_width,
            'ground_depth': ground_depth,
            'particle_radius': particle_radius,
            'particle_type': particle_type,
            'particle_count': len(particles),
            'material_params': material_param,
            'landform_type': landform_type,
            'weather_condition': weather_condition,
            'groundwater_level': groundwater_level,
            'vegetation_density': vegetation_density
        }
        
        print(f"边坡模型构建完成，粒子数量: {len(particles)}")
        return particles
    
    def import_external_model(self, file_path, particle_radius=0.2, particle_type='soil'):
        """
        导入外部模型（简化实现）
        
        参数：
        - file_path: 外部模型文件路径
        - particle_radius: 粒子半径 (m)
        - particle_type: 粒子类型
        
        返回：
        - 粒子列表，每个粒子是一个包含位置、速度、密度等属性的字典
        """
        
        print(f"导入外部模型：{file_path}")
        
        # 简化实现：仅支持简单的文本格式
        # 实际应用中应支持CAD/BIM格式（如DXF、STL、IFC等）
        
        particles = []
        particle_id = 0
        
        try:
            # 尝试读取文本格式模型文件
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析坐标数据，格式：x y z
                coords = list(map(float, line.split()))
                if len(coords) < 2:
                    continue
                
                # 补全为3D坐标
                x, y = coords[0], coords[1]
                z = coords[2] if len(coords) >= 3 else 0.0
                
                # 创建粒子
                particle = {
                    'id': particle_id,
                    'position': np.array([x, y, z]),
                    'velocity': np.array([0.0, 0.0, 0.0]),
                    'acceleration': np.array([0.0, -9.81, 0.0]),
                    'mass': 1.0,
                    'density': 2600.0,
                    'radius': particle_radius,
                    'type': particle_type,
                    'stress': np.zeros((3, 3)),
                    'strain': np.zeros((3, 3)),
                    'displacement': np.zeros(3)
                }
                
                particles.append(particle)
                particle_id += 1
            
            self.particles = particles.copy()
            self.initial_particles = particles.copy()
            
            # 保存模型信息
            self.model_info = {
                'model_type': 'external',
                'file_path': file_path,
                'particle_radius': particle_radius,
                'particle_type': particle_type,
                'particle_count': len(particles)
            }
            
            print(f"外部模型导入完成，粒子数量: {len(particles)}")
            return particles
            
        except Exception as e:
            print(f"导入外部模型失败: {str(e)}")
            return []
    
    def export_model(self, file_path, format='txt'):
        """
        导出模型
        
        参数：
        - file_path: 导出文件路径
        - format: 导出格式（txt, csv）
        
        返回：
        - bool: 导出是否成功
        """
        
        print(f"导出模型到：{file_path}，格式：{format}")
        
        try:
            with open(file_path, 'w') as f:
                if format == 'txt':
                    # TXT格式：每行一个粒子，包含坐标信息
                    f.write("# 粒子模型数据\n")
                    f.write("# 格式：x y z\n")
                    for particle in self.particles:
                        pos = particle['position']
                        f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")
                elif format == 'csv':
                    # CSV格式：包含粒子的完整属性
                    f.write("id,x,y,z,velocity_x,velocity_y,velocity_z,density,radius,type\n")
                    for particle in self.particles:
                        pos = particle['position']
                        vel = particle['velocity']
                        f.write(f"{particle['id']},{pos[0]},{pos[1]},{pos[2]},{vel[0]},{vel[1]},{vel[2]},{particle['density']},{particle['radius']},{particle['type']}\n")
            
            print("模型导出完成")
            return True
        except Exception as e:
            print(f"导出模型失败: {str(e)}")
            return False
    
    def refine_local(self, center, radius, new_radius):
        """
        局部细化模型
        
        参数：
        - center: 细化区域中心坐标
        - radius: 细化区域半径
        - new_radius: 细化后的粒子半径
        
        返回：
        - 细化后的粒子列表
        """
        
        print(f"局部细化：中心={center}, 半径={radius}, 新粒子半径={new_radius}")
        
        if not self.particles:
            print("未构建模型，无法进行局部细化")
            return self.particles.copy()
        
        # 计算细化区域边界
        min_x, max_x = center[0] - radius, center[0] + radius
        min_y, max_y = center[1] - radius, center[1] + radius
        min_z, max_z = center[2] - radius, center[2] + radius
        
        # 移除细化区域内的旧粒子
        refined_particles = [p for p in self.particles if 
                            (p['position'][0] < min_x or p['position'][0] > max_x) or
                            (p['position'][1] < min_y or p['position'][1] > max_y) or
                            (p['position'][2] < min_z or p['position'][2] > max_z)]
        
        # 在细化区域生成新的小粒子
        spacing = new_radius * 2.0
        particle_id = len(refined_particles)
        
        for x in np.arange(min_x, max_x, spacing):
            for y in np.arange(min_y, max_y, spacing):
                for z in np.arange(min_z, max_z, spacing):
                    # 检查是否在细化区域内
                    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                    if distance > radius:
                        continue
                    
                    # 创建新粒子
                    new_particle = {
                        'id': particle_id,
                        'position': np.array([x, y, z]),
                        'velocity': np.array([0.0, 0.0, 0.0]),
                        'acceleration': np.array([0.0, -9.81, 0.0]),
                        'mass': 1.0,
                        'density': 2600.0,
                        'radius': new_radius,
                        'type': 'soil',
                        'stress': np.zeros((3, 3)),
                        'strain': np.zeros((3, 3)),
                        'displacement': np.zeros(3)
                    }
                    
                    refined_particles.append(new_particle)
                    particle_id += 1
        
        # 更新粒子列表
        self.particles = refined_particles.copy()
        
        print(f"局部细化完成，粒子数量从{len(self.initial_particles)}增加到{len(self.particles)}")
        return refined_particles
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回：
        - 模型信息字典
        """
        return self.model_info.copy()
    
    def clear_model(self):
        """
        清除当前模型
        """
        self.particles = []
        self.initial_particles = []
        self.model_info = {}
    
    def plot_slope_model(self, particles=None, show_displacement=False):
        """绘制边坡模型"""
        if particles is None:
            particles = self.particles
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        positions = np.array([p['position'][:2] for p in particles])
        
        if show_displacement:
            displacements = np.array([p['displacement'][:2] for p in particles])
            displacement_magnitudes = np.linalg.norm(displacements, axis=1)
            scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                               c=displacement_magnitudes, cmap='viridis', 
                               s=10, alpha=0.8)
            plt.colorbar(scatter, ax=ax, label='位移 (m)')
        else:
            ax.scatter(positions[:, 0], positions[:, 1], s=10, alpha=0.8, c='brown')
        
        ax.set_title('岩土边坡粒子模型')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('slope_model.png', dpi=300, bbox_inches='tight')
        print("边坡模型图像已保存为: slope_model.png")
        plt.close()

class StabilityEvaluator:
    """稳定性评估器类"""
    
    def __init__(self):
        self.safety_factor = 1.0
        self.max_displacement = 0.0
        self.average_displacement = 0.0
        self.instability_warning_level = 0
        self.stress_analysis_results = {}
        self.potential_slide_surfaces = []
    
    def evaluate_stability(self, particles, initial_particles):
        """
        评估边坡稳定性
        
        参数：
        - particles: 当前粒子列表
        - initial_particles: 初始粒子列表
        
        返回：
        - 稳定性指标字典
        """
        
        print("评估边坡稳定性...")
        
        # 计算位移
        displacements = []
        for p, p0 in zip(particles, initial_particles):
            disp = p['position'] - p0['position']
            displacements.append(disp)
            p['displacement'] = disp
        
        displacements = np.array(displacements)
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        # 计算位移统计指标
        self.max_displacement = np.max(displacement_magnitudes)
        self.average_displacement = np.mean(displacement_magnitudes)
        self.median_displacement = np.median(displacement_magnitudes)
        self.std_displacement = np.std(displacement_magnitudes)
        
        # 计算应力应变统计指标
        self._analyze_stress_strain(particles)
        
        # 计算安全系数（多种方法结合）
        self._compute_safety_factor(particles)
        
        # 检测潜在滑动面
        self.potential_slide_surfaces = self._detect_slide_surfaces(particles)
        
        # 评估失稳预警等级
        self.instability_warning_level = self._evaluate_warning_level()
        
        # 生成评估报告
        self._generate_report()
        
        return {
            'safety_factor': self.safety_factor,
            'max_displacement': self.max_displacement,
            'average_displacement': self.average_displacement,
            'median_displacement': self.median_displacement,
            'std_displacement': self.std_displacement,
            'stress_analysis': self.stress_analysis_results,
            'instability_warning_level': self.instability_warning_level,
            'potential_slide_surfaces': self.potential_slide_surfaces,
            'warning_level': self.instability_warning_level,
            'warning_description': self._get_warning_description()
        }
    
    def _analyze_stress_strain(self, particles):
        """分析应力应变场"""
        # 计算主应力和剪应力
        principal_stresses = []
        shear_stresses = []
        normal_stresses = []
        
        for p in particles:
            if 'stress' in p:
                sigma = p['stress']
                
                # 计算主应力
                eigenvalues, _ = np.linalg.eigh(sigma)
                principal_stresses.append(eigenvalues)
                
                # 计算剪应力和法向应力
                sigma1, sigma2, sigma3 = eigenvalues
                max_shear_stress = (sigma1 - sigma3) / 2.0
                shear_stresses.append(max_shear_stress)
                
                # 平均法向应力
                mean_normal_stress = (sigma1 + sigma2 + sigma3) / 3.0
                normal_stresses.append(mean_normal_stress)
        
        principal_stresses = np.array(principal_stresses)
        shear_stresses = np.array(shear_stresses)
        normal_stresses = np.array(normal_stresses)
        
        # 计算应力统计指标
        self.stress_analysis_results = {
            'max_principal_stress': np.max(principal_stresses[:, 0]),
            'min_principal_stress': np.min(principal_stresses[:, 2]),
            'avg_max_shear_stress': np.mean(shear_stresses),
            'max_max_shear_stress': np.max(shear_stresses),
            'avg_normal_stress': np.mean(normal_stresses),
            'stress_variance': np.var(principal_stresses),
            'principal_stresses': principal_stresses,
            'shear_stresses': shear_stresses,
            'normal_stresses': normal_stresses
        }
    
    def _compute_safety_factor(self, particles):
        """计算安全系数"""
        # 基于应力的安全系数计算（简化的Bishop法）
        stress_sf = self._compute_stress_based_safety_factor(particles)
        
        # 基于位移的安全系数计算
        displacement_sf = self._compute_displacement_based_safety_factor()
        
        # 基于强度折减法的安全系数计算（简化实现）
        strength_sf = self._compute_strength_reduction_safety_factor(particles)
        
        # 综合安全系数（加权平均）
        weights = [0.4, 0.3, 0.3]  # 应力法权重最高，其次是强度折减法和位移法
        self.safety_factor = (stress_sf * weights[0] + 
                             strength_sf * weights[1] + 
                             displacement_sf * weights[2])
        
        # 确保安全系数在合理范围内
        self.safety_factor = max(0.5, min(3.0, self.safety_factor))
    
    def _compute_stress_based_safety_factor(self, particles):
        """基于应力的安全系数计算（简化的Bishop法）"""
        if not particles or 'stress' not in particles[0]:
            return 1.5  # 默认值
        
        # 简化的Bishop法实现
        total_moment_resisting = 0.0
        total_moment_driving = 0.0
        
        # 假设潜在滑动中心在边坡顶部
        slip_center = np.array([0.0, max(p['position'][1] for p in particles), 0.0])
        
        for p in particles:
            # 计算粒子到滑动中心的距离和角度
            r_vector = p['position'] - slip_center
            r = np.linalg.norm(r_vector)
            if r < 0.1:  # 避免除以零
                continue
            
            # 计算法向应力和剪应力
            sigma = p['stress']
            eigenvalues, _ = np.linalg.eigh(sigma)
            sigma1, sigma2, sigma3 = eigenvalues
            max_shear_stress = (sigma1 - sigma3) / 2.0
            mean_normal_stress = (sigma1 + sigma2 + sigma3) / 3.0
            
            # 计算抗剪强度（基于Mohr-Coulomb准则）
            cohesion = p.get('material_params', {}).get('cohesion', 10e3)
            friction_angle = np.radians(p.get('material_params', {}).get('friction_angle', 30.0))
            
            # 有效法向应力（简化实现，不考虑孔隙水压力）
            effective_normal_stress = abs(mean_normal_stress)  # 假设压应力为正
            
            # 抗剪强度
            shear_strength = cohesion + effective_normal_stress * np.tan(friction_angle)
            
            # 计算力矩
            driving_moment = max_shear_stress * p['mass'] * r
            resisting_moment = shear_strength * p['mass'] * r
            
            total_moment_driving += driving_moment
            total_moment_resisting += resisting_moment
        
        if total_moment_driving < 1e-10:
            return 2.0  # 无驱动力矩，安全系数高
        
        return total_moment_resisting / total_moment_driving
    
    def _compute_displacement_based_safety_factor(self):
        """基于位移的安全系数计算"""
        if self.max_displacement < 1.0e-5:
            return 2.5  # 位移很小，安全系数高
        elif self.max_displacement < 1.0e-4:
            return 2.0
        elif self.max_displacement < 0.001:
            return 1.8
        elif self.max_displacement < 0.01:
            return 1.5
        elif self.max_displacement < 0.05:
            return 1.2
        elif self.max_displacement < 0.1:
            return 1.0
        else:
            return 0.8  # 位移很大，安全系数低
    
    def _compute_strength_reduction_safety_factor(self, particles):
        """基于强度折减法的安全系数计算（简化实现）"""
        # 简化的强度折减法：基于应力水平与强度的比值
        strength_ratios = []
        
        for p in particles:
            if 'stress' in p and 'material_params' in p:
                sigma = p['stress']
                params = p['material_params']
                
                # 计算最大剪应力
                eigenvalues, _ = np.linalg.eigh(sigma)
                sigma1, sigma2, sigma3 = eigenvalues
                max_shear_stress = (sigma1 - sigma3) / 2.0
                
                # 计算抗剪强度
                cohesion = params.get('cohesion', 10e3)
                friction_angle = np.radians(params.get('friction_angle', 30.0))
                mean_normal_stress = (sigma1 + sigma2 + sigma3) / 3.0
                effective_normal_stress = abs(mean_normal_stress)
                
                shear_strength = cohesion + effective_normal_stress * np.tan(friction_angle)
                
                if shear_strength > 0:
                    strength_ratio = shear_strength / (max_shear_stress + 1e-10)
                    strength_ratios.append(strength_ratio)
        
        if not strength_ratios:
            return 1.5  # 默认值
        
        # 安全系数为强度比值的最小值
        return np.min(strength_ratios)
    
    def _detect_slide_surfaces(self, particles):
        """检测潜在滑动面"""
        # 基于位移和应力的滑动面检测
        positions = np.array([p['position'] for p in particles])
        displacement_magnitudes = np.array([np.linalg.norm(p['displacement']) for p in particles])
        
        # 方法1：基于位移梯度检测
        slide_surfaces = []
        
        # 检测位移突变区域
        high_displacement_indices = np.where(displacement_magnitudes > self.average_displacement * 2)[0]
        
        if len(high_displacement_indices) > 0:
            # 获取高位移区域的粒子
            high_displacement_particles = positions[high_displacement_indices]
            
            # 按x坐标排序
            sorted_indices = np.argsort(high_displacement_particles[:, 0])
            sorted_high_disp = high_displacement_particles[sorted_indices]
            
            # 提取滑动面
            slide_surface = []
            prev_x = None
            for pos in sorted_high_disp:
                x, y, z = pos
                if prev_x is None or abs(x - prev_x) > 0.1:  # 避免重复点
                    slide_surface.append([x, y, z])
                    prev_x = x
            
            if len(slide_surface) > 5:  # 至少需要5个点才能构成滑动面
                slide_surfaces.append(np.array(slide_surface))
        
        # 方法2：基于应力检测
        if 'stress' in particles[0]:
            shear_stresses = []
            for p in particles:
                sigma = p['stress']
                eigenvalues, _ = np.linalg.eigh(sigma)
                max_shear = (eigenvalues[0] - eigenvalues[2]) / 2.0
                shear_stresses.append(max_shear)
            
            shear_stresses = np.array(shear_stresses)
            high_shear_indices = np.where(shear_stresses > np.mean(shear_stresses) * 2)[0]
            
            if len(high_shear_indices) > 0:
                high_shear_particles = positions[high_shear_indices]
                sorted_indices = np.argsort(high_shear_particles[:, 0])
                sorted_high_shear = high_shear_particles[sorted_indices]
                
                slide_surface = []
                prev_x = None
                for pos in sorted_high_shear:
                    x, y, z = pos
                    if prev_x is None or abs(x - prev_x) > 0.1:
                        slide_surface.append([x, y, z])
                        prev_x = x
                
                if len(slide_surface) > 5:
                    slide_surfaces.append(np.array(slide_surface))
        
        # 如果没有检测到滑动面，使用默认的边坡轮廓线
        if not slide_surfaces:
            # 简化实现：返回边坡线
            sorted_indices = np.argsort(positions[:, 0])
            sorted_x = positions[sorted_indices, 0]
            sorted_y = positions[sorted_indices, 1]
            
            unique_x = np.unique(sorted_x)
            slide_surface = []
            
            for x in unique_x:
                mask = (sorted_x == x)
                max_y = np.max(sorted_y[mask])
                slide_surface.append([x, max_y, 0.0])
            
            slide_surfaces.append(np.array(slide_surface))
        
        return slide_surfaces
    
    def _evaluate_warning_level(self):
        """评估失稳预警等级"""
        warning_level = 0
        
        # 基于安全系数
        if self.safety_factor < 1.0:
            warning_level += 40
        elif self.safety_factor < 1.2:
            warning_level += 20
        elif self.safety_factor < 1.5:
            warning_level += 10
        
        # 基于位移
        critical_displacement = 0.1
        if self.max_displacement > critical_displacement:
            warning_level += 30
        elif self.max_displacement > critical_displacement * 0.5:
            warning_level += 20
        elif self.max_displacement > critical_displacement * 0.2:
            warning_level += 10
        
        # 基于应力状态
        if 'max_max_shear_stress' in self.stress_analysis_results:
            max_shear = self.stress_analysis_results['max_max_shear_stress']
            avg_shear = self.stress_analysis_results['avg_max_shear_stress']
            if max_shear > avg_shear * 3:
                warning_level += 20
            elif max_shear > avg_shear * 2:
                warning_level += 10
        
        # 确保预警等级在0-100范围内
        return max(0, min(100, warning_level))
    
    def _get_warning_description(self):
        """获取预警等级描述"""
        if self.instability_warning_level <= 20:
            return "稳定 - 边坡处于稳定状态，无明显失稳风险"
        elif self.instability_warning_level <= 40:
            return "基本稳定 - 边坡基本稳定，需定期监测"
        elif self.instability_warning_level <= 60:
            return "轻度预警 - 边坡出现轻微失稳迹象，需加强监测"
        elif self.instability_warning_level <= 80:
            return "中度预警 - 边坡出现明显失稳迹象，需采取防护措施"
        else:
            return "重度预警 - 边坡处于危险状态，极可能发生失稳"
    
    def _generate_report(self):
        """生成稳定性评估报告"""
        report_path = 'slope_stability_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("========================================\n")
            f.write("          岩土边坡稳定性评估报告          \n")
            f.write("========================================\n\n")
            
            f.write("1. 基本信息\n")
            f.write("----------------------------------------\n")
            f.write(f"评估时间: {np.datetime64('now')}\n")
            f.write(f"分析方法: 粒子法数值模拟\n")
            f.write(f"安全系数计算方法: 应力法 + 强度折减法 + 位移法综合评估\n\n")
            
            f.write("2. 稳定性指标\n")
            f.write("----------------------------------------\n")
            f.write(f"安全系数: {self.safety_factor:.3f}\n")
            f.write(f"最大位移: {self.max_displacement:.6f} m\n")
            f.write(f"平均位移: {self.average_displacement:.6f} m\n")
            f.write(f"位移中位数: {self.median_displacement:.6f} m\n")
            f.write(f"位移标准差: {self.std_displacement:.6f} m\n\n")
            
            f.write("3. 应力应变分析\n")
            f.write("----------------------------------------\n")
            f.write(f"最大主应力: {self.stress_analysis_results.get('max_principal_stress', 0):.2f} Pa\n")
            f.write(f"最小主应力: {self.stress_analysis_results.get('min_principal_stress', 0):.2f} Pa\n")
            f.write(f"平均最大剪应力: {self.stress_analysis_results.get('avg_max_shear_stress', 0):.2f} Pa\n")
            f.write(f"最大剪应力: {self.stress_analysis_results.get('max_max_shear_stress', 0):.2f} Pa\n\n")
            
            f.write("4. 失稳预警评估\n")
            f.write("----------------------------------------\n")
            f.write(f"失稳预警等级: {self.instability_warning_level} (0-100)\n")
            f.write(f"预警等级说明: {self._get_warning_description()}\n\n")
            
            f.write("5. 潜在滑动面分析\n")
            f.write("----------------------------------------\n")
            f.write(f"检测到的潜在滑动面数量: {len(self.potential_slide_surfaces)}\n")
            for i, surface in enumerate(self.potential_slide_surfaces):
                f.write(f"滑动面 {i+1}: 包含 {len(surface)} 个特征点\n")
            f.write("\n")
            
            f.write("6. 安全建议\n")
            f.write("----------------------------------------\n")
            if self.instability_warning_level <= 20:
                f.write("- 边坡处于稳定状态，建议定期进行常规监测\n")
                f.write("- 关注降雨量变化对边坡的影响\n")
            elif self.instability_warning_level <= 40:
                f.write("- 边坡基本稳定，建议增加监测频率\n")
                f.write("- 对关键区域进行重点监测\n")
                f.write("- 考虑进行长期变形监测\n")
            elif self.instability_warning_level <= 60:
                f.write("- 边坡出现轻微失稳迹象，建议加强监测力度\n")
                f.write("- 对潜在滑动面区域进行加密监测\n")
                f.write("- 考虑采取适当的加固措施\n")
            elif self.instability_warning_level <= 80:
                f.write("- 边坡出现明显失稳迹象，建议立即采取防护措施\n")
                f.write("- 对危险区域进行封闭，禁止人员靠近\n")
                f.write("- 尽快制定加固方案并实施\n")
            else:
                f.write("- 边坡处于危险状态，极可能发生失稳\n")
                f.write("- 立即启动应急预案\n")
                f.write("- 疏散危险区域人员和设备\n")
                f.write("- 组织专家进行紧急评估并制定抢险方案\n")
            
            f.write("\n")
            f.write("7. 监测建议\n")
            f.write("----------------------------------------\n")
            f.write("- 位移监测：采用全站仪、GPS或位移计进行定期监测\n")
            f.write("- 应力监测：在关键区域安装应力计\n")
            f.write("- 地下水监测：监测地下水位变化\n")
            f.write("- 降雨监测：关注降雨量和降雨强度\n")
            f.write("- 建立自动化监测系统，实现实时预警\n")
            
        print(f"稳定性评估报告已生成: {report_path}")
    
    def _get_warning_description(self):
        """获取预警等级描述"""
        if self.instability_warning_level <= 20:
            return "稳定 - 边坡处于稳定状态，无明显失稳风险"
        elif self.instability_warning_level <= 40:
            return "基本稳定 - 边坡基本稳定，需定期监测"
        elif self.instability_warning_level <= 60:
            return "轻度预警 - 边坡出现轻微失稳迹象，需加强监测"
        elif self.instability_warning_level <= 80:
            return "中度预警 - 边坡出现明显失稳迹象，需采取防护措施"
        else:
            return "重度预警 - 边坡处于危险状态，极可能发生失稳"
    
    def visualize_stability_results(self, particles, slide_surfaces=None):
        """可视化稳定性评估结果"""
        if slide_surfaces is None:
            slide_surfaces = self.potential_slide_surfaces
        
        # 创建3x2布局的图表
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 位移场分布
        ax1 = fig.add_subplot(2, 3, 1)
        positions = np.array([p['position'][:2] for p in particles])
        displacements = np.array([p['displacement'][:2] for p in particles])
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                             c=displacement_magnitudes, cmap='viridis', 
                             s=10, alpha=0.8)
        plt.colorbar(scatter, ax=ax1, label='位移 (m)')
        
        # 绘制潜在滑动面
        for i, surface in enumerate(slide_surfaces):
            ax1.plot(surface[:, 0], surface[:, 1], 
                    '--', linewidth=2, label=f'滑动面 {i+1}')
        ax1.legend()
        ax1.set_title('位移场分布与潜在滑动面')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_aspect('equal')
        ax1.grid(True)
        
        # 2. 安全系数及位移指标
        ax2 = fig.add_subplot(2, 3, 2)
        metrics = ['安全系数', '最大位移 (m)', '平均位移 (m)', '失稳预警等级']
        values = [self.safety_factor, self.max_displacement, 
                 self.average_displacement, self.instability_warning_level]
        
        colors = ['green' if v >= 1.5 else 'yellow' if v >= 1.0 else 'red' for v in values[:1]] + ['blue', 'orange', 'red']
        ax2.bar(metrics, values, color=colors)
        ax2.set_title('边坡稳定性指标')
        ax2.set_ylabel('数值')
        ax2.grid(True, axis='y')
        
        # 显示数值
        for i, v in enumerate(values):
            ax2.text(i, v + 0.05 * max(values), f'{v:.2f}', ha='center', va='bottom')
        
        # 3. 应力分布
        ax3 = fig.add_subplot(2, 3, 3)
        if 'shear_stresses' in self.stress_analysis_results:
            shear_stresses = self.stress_analysis_results['shear_stresses']
            scatter = ax3.scatter(positions[:, 0], positions[:, 1], 
                                 c=shear_stresses, cmap='jet', 
                                 s=10, alpha=0.8)
            plt.colorbar(scatter, ax=ax3, label='剪应力 (Pa)')
            ax3.set_title('剪应力分布')
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_aspect('equal')
            ax3.grid(True)
        
        # 4. 位移统计分布
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.hist(displacement_magnitudes, bins=50, alpha=0.7, color='blue')
        ax4.axvline(self.max_displacement, color='red', linestyle='--', label=f'最大位移: {self.max_displacement:.6f} m')
        ax4.axvline(self.average_displacement, color='green', linestyle='--', label=f'平均位移: {self.average_displacement:.6f} m')
        ax4.set_title('位移分布直方图')
        ax4.set_xlabel('位移 (m)')
        ax4.set_ylabel('粒子数量')
        ax4.legend()
        ax4.grid(True)
        
        # 5. 主应力关系
        ax5 = fig.add_subplot(2, 3, 5)
        if 'principal_stresses' in self.stress_analysis_results:
            principal_stresses = self.stress_analysis_results['principal_stresses']
            sigma1 = principal_stresses[:, 0]
            sigma3 = principal_stresses[:, 2]
            ax5.scatter(sigma3, sigma1, alpha=0.6, s=5, color='purple')
            ax5.set_title('主应力关系 (σ3 vs σ1)')
            ax5.set_xlabel('最小主应力 σ3 (Pa)')
            ax5.set_ylabel('最大主应力 σ1 (Pa)')
            ax5.grid(True)
        
        # 6. 安全系数与位移关系
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.scatter(displacement_magnitudes, np.full_like(displacement_magnitudes, self.safety_factor), 
                   alpha=0.6, s=5, color='orange')
        ax6.axhline(self.safety_factor, color='red', linestyle='--', label=f'安全系数: {self.safety_factor:.3f}')
        ax6.set_title('安全系数与位移关系')
        ax6.set_xlabel('位移 (m)')
        ax6.set_ylabel('安全系数')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig('stability_results.png', dpi=300, bbox_inches='tight')
        print("稳定性评估结果图像已保存为: stability_results.png")
        plt.close()
    
    def generate_comparison_report(self, previous_results, current_results):
        """
        生成对比报告
        
        参数：
        - previous_results: 上次评估结果
        - current_results: 当前评估结果
        
        返回：
        - 对比报告文本
        """
        report = """========================================
          岩土边坡稳定性对比评估报告          
========================================\n\n"""
        
        report += "1. 安全系数对比\n"
        report += "----------------------------------------\n"
        report += f"上次评估: {previous_results.get('safety_factor', 0):.3f}\n"
        report += f"当前评估: {current_results.get('safety_factor', 0):.3f}\n"
        report += f"变化量: {current_results.get('safety_factor', 0) - previous_results.get('safety_factor', 0):.3f}\n"
        report += f"变化率: {((current_results.get('safety_factor', 0) - previous_results.get('safety_factor', 0)) / (previous_results.get('safety_factor', 1) + 1e-10)) * 100:.2f}%\n\n"
        
        report += "2. 位移对比\n"
        report += "----------------------------------------\n"
        report += f"上次最大位移: {previous_results.get('max_displacement', 0):.6f} m\n"
        report += f"当前最大位移: {current_results.get('max_displacement', 0):.6f} m\n"
        report += f"位移变化: {current_results.get('max_displacement', 0) - previous_results.get('max_displacement', 0):.6f} m\n"
        report += f"位移增长率: {((current_results.get('max_displacement', 0) - previous_results.get('max_displacement', 0)) / (previous_results.get('max_displacement', 1e-10) + 1e-10)) * 100:.2f}%\n\n"
        
        report += "3. 预警等级对比\n"
        report += "----------------------------------------\n"
        report += f"上次预警等级: {previous_results.get('instability_warning_level', 0)} ({previous_results.get('warning_description', '')})\n"
        report += f"当前预警等级: {current_results.get('instability_warning_level', 0)} ({current_results.get('warning_description', '')})\n"
        report += f"预警等级变化: {current_results.get('instability_warning_level', 0) - previous_results.get('instability_warning_level', 0)}\n\n"
        
        report += "4. 评估结论与建议\n"
        report += "----------------------------------------\n"
        if current_results.get('safety_factor', 0) < previous_results.get('safety_factor', 0):
            report += "- 安全系数有所降低，边坡稳定性下降\n"
        elif current_results.get('safety_factor', 0) > previous_results.get('safety_factor', 0):
            report += "- 安全系数有所提高，边坡稳定性改善\n"
        else:
            report += "- 安全系数基本保持不变\n"
        
        if current_results.get('max_displacement', 0) > previous_results.get('max_displacement', 0) * 1.5:
            report += "- 位移增长较快，需密切关注\n"
        
        if current_results.get('instability_warning_level', 0) > previous_results.get('instability_warning_level', 0) + 20:
            report += "- 预警等级明显升高，建议立即采取措施\n"
        
        report += "\n"
        report += "建议根据评估结果调整监测频率和防护措施，确保边坡安全。\n"
        
        # 保存对比报告
        with open('slope_stability_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("稳定性对比评估报告已生成: slope_stability_comparison_report.txt")
        return report

def main():
    """主函数"""
    print("========================================")
    print("        粒子法岩土边坡仿真软件           ")
    print("========================================\n")
    
    # 创建输出目录
    if not os.path.exists('output'):
        os.makedirs('output')
    os.chdir('output')
    
    # 1. 构建边坡模型
    slope_builder = SlopeModelBuilder()
    particles = slope_builder.build_parametric_slope(
        slope_height=10.0,
        slope_angle=45.0,
        slope_width=20.0,
        ground_depth=5.0,
        particle_radius=0.2
    )
    
    # 2. 绘制初始边坡模型
    slope_builder.plot_slope_model()
    
    # 3. 模拟边坡变形（简化实现）
    print("\n模拟边坡变形...")
    deformed_particles = []
    for p in particles:
        # 模拟重力引起的变形
        # 简化实现：x>0的区域，y坐标随x增大而减小
        x = p['position'][0]
        y = p['position'][1]
        
        # 模拟位移
        if x > 0 and y > 0:
            # 边坡区域，产生更大的位移
            displacement = np.array([0.01 * x, -0.005 * x, 0.0])
        else:
            # 其他区域，位移较小
            displacement = np.array([0.0, -0.001, 0.0])
        
        deformed_particle = p.copy()
        deformed_particle['position'] += displacement
        deformed_particles.append(deformed_particle)
    
    # 4. 绘制变形后的边坡模型
    slope_builder.plot_slope_model(deformed_particles, show_displacement=True)
    
    # 5. 评估边坡稳定性
    stability_evaluator = StabilityEvaluator()
    indices = stability_evaluator.evaluate_stability(deformed_particles, particles)
    
    # 6. 可视化稳定性评估结果
    stability_evaluator.visualize_stability_results(deformed_particles)
    
    print("\n========================================")
    print("           仿真程序执行完成              ")
    print("========================================")
    print(f"安全系数: {indices['safety_factor']:.2f}")
    print(f"失稳预警等级: {indices['instability_warning_level']} (0-100)")
    print(f"最大位移: {indices['max_displacement']:.6f} m")
    print("\n查看输出文件:")
    print("- slope_model.png: 边坡模型图像")
    print("- slope_stability_report.txt: 稳定性评估报告")
    print("- stability_results.png: 稳定性评估结果图像")

if __name__ == "__main__":
    main()
