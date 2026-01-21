// ParticleGenerator.cpp
// 粒子生成与初始化模块实现

#include "ParticleGenerator.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

namespace particle_simulation {

// 构造函数
ParticleGenerator::ParticleGenerator() 
    : gravity_(0.0, -9.81, 0.0), // 默认重力方向向下
      particle_radius_(0.1) {
}

// 参数化生成边坡粒子模型
std::vector<Particle> ParticleGenerator::generateSlopeParticles(
    double slope_height,
    double slope_angle,
    double particle_radius,
    const Eigen::VectorXd& soil_params,
    double moisture_content
) {
    std::vector<Particle> particles;
    particle_radius_ = particle_radius;
    
    // 计算边坡的几何参数
    double slope_rad = slope_angle * M_PI / 180.0;
    double slope_ratio = tan(slope_rad);
    double slope_length = slope_height / sin(slope_rad);
    
    // 确定生成区域的边界
    double min_x = -slope_length * 0.5;
    double max_x = slope_length * 1.5;
    double min_y = 0.0;
    double max_y = slope_height;
    double min_z = -slope_length * 0.2;
    double max_z = slope_length * 0.2;
    
    // 生成网格点
    double spacing = particle_radius * 2.0; // 粒子间距为2倍半径
    std::vector<Eigen::Vector3d> grid_points = generateGridPoints(
        min_x, max_x, min_y, max_y, min_z, max_z, spacing
    );
    
    // 筛选出在边坡区域内的点并生成粒子
    for (const auto& point : grid_points) {
        if (isPointInSlope(point, slope_height, slope_angle)) {
            Particle particle(point, particle_radius * particle_radius * particle_radius * 4.0 / 3.0 * M_PI * 2600.0, ParticleType::SOIL);
            
            // 设置粒子参数
            particle.setRadius(particle_radius);
            particle.setDensity(2600.0); // 土体默认密度
            particle.setLithologyParams(soil_params);
            particle.setMoistureContent(moisture_content);
            
            particles.push_back(particle);
        }
    }
    
    std::cout << "Generated " << particles.size() << " slope particles." << std::endl;
    return particles;
}

// 从外部文件导入粒子模型
std::vector<Particle> ParticleGenerator::importFromFile(
    const std::string& file_path,
    double particle_radius
) {
    std::vector<Particle> particles;
    particle_radius_ = particle_radius;
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return particles;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释行
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        double x, y, z, mass;
        int type;
        
        if (iss >> x >> y >> z >> mass >> type) {
            Eigen::Vector3d position(x, y, z);
            ParticleType particle_type = static_cast<ParticleType>(type);
            
            Particle particle(position, mass, particle_type);
            particle.setRadius(particle_radius);
            particles.push_back(particle);
        }
    }
    
    file.close();
    std::cout << "Imported " << particles.size() << " particles from file." << std::endl;
    return particles;
}

// 生成边界粒子
std::vector<Particle> ParticleGenerator::generateBoundaryParticles(
    const std::vector<Particle>& particles,
    double boundary_thickness,
    double particle_radius
) {
    std::vector<Particle> boundary_particles;
    
    // 确定现有粒子的边界范围
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::lowest();
    
    for (const auto& particle : particles) {
        const Eigen::Vector3d& pos = particle.getPosition();
        min_x = std::min(min_x, pos.x());
        max_x = std::max(max_x, pos.x());
        min_y = std::min(min_y, pos.y());
        max_y = std::max(max_y, pos.y());
        min_z = std::min(min_z, pos.z());
        max_z = std::max(max_z, pos.z());
    }
    
    // 扩展边界范围
    min_x -= boundary_thickness;
    max_x += boundary_thickness;
    min_y -= boundary_thickness;
    max_y += boundary_thickness;
    min_z -= boundary_thickness;
    max_z += boundary_thickness;
    
    // 生成边界粒子
    double spacing = particle_radius * 2.0;
    
    // 生成底部边界
    for (double x = min_x; x <= max_x; x += spacing) {
        for (double z = min_z; z <= max_z; z += spacing) {
            Eigen::Vector3d pos(x, min_y, z);
            Particle particle(pos, particle_radius * particle_radius * particle_radius * 4.0 / 3.0 * M_PI * 3000.0, ParticleType::BOUNDARY);
            particle.setRadius(particle_radius);
            particle.setDensity(3000.0);
            boundary_particles.push_back(particle);
        }
    }
    
    // 生成左右边界
    for (double y = min_y; y <= max_y; y += spacing) {
        for (double z = min_z; z <= max_z; z += spacing) {
            // 左边界
            Eigen::Vector3d pos(min_x, y, z);
            Particle particle(pos, particle_radius * particle_radius * particle_radius * 4.0 / 3.0 * M_PI * 3000.0, ParticleType::BOUNDARY);
            particle.setRadius(particle_radius);
            particle.setDensity(3000.0);
            boundary_particles.push_back(particle);
            
            // 右边界
            pos.x() = max_x;
            particle.setPosition(pos);
            boundary_particles.push_back(particle);
        }
    }
    
    // 生成前后边界
    for (double x = min_x; x <= max_x; x += spacing) {
        for (double y = min_y; y <= max_y; y += spacing) {
            // 前边界
            Eigen::Vector3d pos(x, y, min_z);
            Particle particle(pos, particle_radius * particle_radius * particle_radius * 4.0 / 3.0 * M_PI * 3000.0, ParticleType::BOUNDARY);
            particle.setRadius(particle_radius);
            particle.setDensity(3000.0);
            boundary_particles.push_back(particle);
            
            // 后边界
            pos.z() = max_z;
            particle.setPosition(pos);
            boundary_particles.push_back(particle);
        }
    }
    
    std::cout << "Generated " << boundary_particles.size() << " boundary particles." << std::endl;
    return boundary_particles;
}

// 设置粒子初始速度
void ParticleGenerator::setInitialVelocity(
    std::vector<Particle>& particles,
    const Eigen::Vector3d& initial_velocity
) {
    for (auto& particle : particles) {
        particle.setVelocity(initial_velocity);
    }
}

// 设置重力场
void ParticleGenerator::setGravity(const Eigen::Vector3d& gravity) {
    gravity_ = gravity;
}

// 检查点是否在边坡区域内
bool ParticleGenerator::isPointInSlope(
    const Eigen::Vector3d& point,
    double slope_height,
    double slope_angle
) const {
    double slope_rad = slope_angle * M_PI / 180.0;
    double slope_ratio = tan(slope_rad);
    
    // 边坡的斜边方程：y = slope_height - (x - 0) * slope_ratio
    // 点在边坡内的条件：
    // 1. y >= 0
    // 2. y <= slope_height
    // 3. 如果x >= 0，则y <= slope_height - x * slope_ratio
    
    if (point.y() < 0.0 || point.y() > slope_height) {
        return false;
    }
    
    if (point.x() >= 0.0) {
        double slope_y = slope_height - point.x() * slope_ratio;
        return point.y() <= slope_y;
    }
    
    return true;
}

// 生成网格点
std::vector<Eigen::Vector3d> ParticleGenerator::generateGridPoints(
    double min_x,
    double max_x,
    double min_y,
    double max_y,
    double min_z,
    double max_z,
    double spacing
) const {
    std::vector<Eigen::Vector3d> points;
    
    for (double x = min_x; x <= max_x; x += spacing) {
        for (double y = min_y; y <= max_y; y += spacing) {
            for (double z = min_z; z <= max_z; z += spacing) {
                points.emplace_back(x, y, z);
            }
        }
    }
    
    return points;
}

} // namespace particle_simulation
