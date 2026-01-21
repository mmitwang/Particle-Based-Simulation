// ParticleGenerator.h
// 粒子生成与初始化模块
// 支持参数化生成岩土边坡粒子模型

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "Particle.h"

namespace particle_simulation {

// 粒子生成器类
class ParticleGenerator {
public:
    // 构造函数
    ParticleGenerator();
    
    // 参数化生成边坡粒子模型
    // 参数：
    // - slope_height: 边坡高度
    // - slope_angle: 边坡坡角（度数）
    // - particle_radius: 粒子半径
    // - soil_params: 土体参数（弹性模量、泊松比、内摩擦角、黏聚力等）
    // - moisture_content: 含水率
    // 返回：生成的粒子列表
    std::vector<Particle> generateSlopeParticles(
        double slope_height,
        double slope_angle,
        double particle_radius,
        const Eigen::VectorXd& soil_params,
        double moisture_content = 0.0
    );
    
    // 从外部文件导入粒子模型
    // 参数：
    // - file_path: 模型文件路径
    // - particle_radius: 粒子半径
    // 返回：生成的粒子列表
    std::vector<Particle> importFromFile(
        const std::string& file_path,
        double particle_radius
    );
    
    // 生成边界粒子
    // 参数：
    // - particles: 现有粒子列表
    // - boundary_thickness: 边界厚度
    // - particle_radius: 粒子半径
    // 返回：生成的边界粒子列表
    std::vector<Particle> generateBoundaryParticles(
        const std::vector<Particle>& particles,
        double boundary_thickness,
        double particle_radius
    );
    
    // 设置粒子初始速度
    // 参数：
    // - particles: 粒子列表
    // - initial_velocity: 初始速度场
    void setInitialVelocity(
        std::vector<Particle>& particles,
        const Eigen::Vector3d& initial_velocity
    );
    
    // 设置重力场
    // 参数：
    // - gravity: 重力加速度向量
    void setGravity(const Eigen::Vector3d& gravity);
    
private:
    // 检查点是否在边坡区域内
    bool isPointInSlope(
        const Eigen::Vector3d& point,
        double slope_height,
        double slope_angle
    ) const;
    
    // 生成网格点
    std::vector<Eigen::Vector3d> generateGridPoints(
        double min_x,
        double max_x,
        double min_y,
        double max_y,
        double min_z,
        double max_z,
        double spacing
    ) const;
    
    Eigen::Vector3d gravity_;  // 重力加速度
    double particle_radius_;   // 粒子半径
};

} // namespace particle_simulation
