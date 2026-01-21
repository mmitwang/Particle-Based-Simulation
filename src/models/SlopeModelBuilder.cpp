// SlopeModelBuilder.cpp
// 岩土边坡模型构建模块实现

#include "SlopeModelBuilder.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace particle_simulation {

// 构造函数
SlopeModelBuilder::SlopeModelBuilder()
    : model_type_(SlopeModelType::PARAMETRIC) {
}

// 参数化构建边坡模型
std::vector<Particle> SlopeModelBuilder::buildParametricSlope(
    const SlopeGeometryParams& geometry_params,
    const GeomaterialParams& geomaterial_params,
    double particle_radius,
    bool is_3d
) {
    std::vector<Particle> particles;
    
    // 计算边坡边界范围
    auto [min_bound, max_bound] = calculateSlopeBounds(geometry_params, is_3d);
    
    // 生成网格点云
    std::vector<Eigen::Vector3d> grid_points = generateGridPoints(min_bound, max_bound, particle_radius);
    
    // 筛选并生成边坡粒子
    for (const auto& point : grid_points) {
        if (isPointInSlope(point, geometry_params, is_3d)) {
            // 创建粒子
            Particle particle;
            particle.setPosition(point);
            particle.setRadius(particle_radius);
            
            // 设置粒子质量和密度
            double volume = (4.0 / 3.0) * M_PI * std::pow(particle_radius, 3);
            double mass = volume * geomaterial_params.density;
            particle.setMass(mass);
            particle.setDensity(geomaterial_params.density);
            
            // 设置粒子类型
            particle.setType(ParticleType::SOIL);
            
            // 设置岩性参数
            Eigen::VectorXd lithology_params(5);
            lithology_params << 
                geomaterial_params.young_modulus,
                geomaterial_params.poisson_ratio,
                geomaterial_params.cohesion,
                geomaterial_params.friction_angle,
                geomaterial_params.dilation_angle;
            particle.setLithologyParams(lithology_params);
            
            // 设置含水率
            particle.setMoistureContent(geomaterial_params.moisture_content);
            
            particles.push_back(particle);
        }
    }
    
    std::cout << "Generated parametric slope model with " << particles.size() << " particles." << std::endl;
    return particles;
}

// 从外部文件导入边坡模型
std::vector<Particle> SlopeModelBuilder::importExternalModel(
    const std::string& file_path,
    double particle_radius,
    const GeomaterialParams& geomaterial_params
) {
    std::vector<Particle> particles;
    
    // 简化实现：目前只支持自定义格式的模型文件
    // 格式：每行一个点，x y z
    
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
        double x, y, z;
        
        if (iss >> x >> y >> z) {
            Eigen::Vector3d point(x, y, z);
            
            // 创建粒子
            Particle particle;
            particle.setPosition(point);
            particle.setRadius(particle_radius);
            
            // 设置粒子质量和密度
            double volume = (4.0 / 3.0) * M_PI * std::pow(particle_radius, 3);
            double mass = volume * geomaterial_params.density;
            particle.setMass(mass);
            particle.setDensity(geomaterial_params.density);
            
            // 设置粒子类型
            particle.setType(ParticleType::SOIL);
            
            // 设置岩性参数
            Eigen::VectorXd lithology_params(5);
            lithology_params << 
                geomaterial_params.young_modulus,
                geomaterial_params.poisson_ratio,
                geomaterial_params.cohesion,
                geomaterial_params.friction_angle,
                geomaterial_params.dilation_angle;
            particle.setLithologyParams(lithology_params);
            
            // 设置含水率
            particle.setMoistureContent(geomaterial_params.moisture_content);
            
            particles.push_back(particle);
        }
    }
    
    file.close();
    std::cout << "Imported external slope model with " << particles.size() << " particles." << std::endl;
    return particles;
}

// 构建多级边坡模型
std::vector<Particle> SlopeModelBuilder::buildMultiLevelSlope(
    const std::vector<SlopeGeometryParams>& geometry_params_list,
    const GeomaterialParams& geomaterial_params,
    double particle_radius,
    bool is_3d
) {
    std::vector<Particle> particles;
    
    // 简化实现：目前只支持单级边坡
    // 多级边坡需要更复杂的几何判断逻辑
    if (!geometry_params_list.empty()) {
        particles = buildParametricSlope(geometry_params_list[0], geomaterial_params, particle_radius, is_3d);
    }
    
    std::cout << "Generated multi-level slope model with " << particles.size() << " particles." << std::endl;
    return particles;
}

// 添加地下水模型
std::vector<Particle> SlopeModelBuilder::addGroundwaterModel(
    const std::vector<Particle>& particles,
    double water_table_depth,
    double particle_radius
) {
    std::vector<Particle> particles_with_water = particles;
    
    // 计算地下水位高度（假设地面高度为0）
    double water_table_height = -water_table_depth;
    
    // 为地下水位以下的区域添加水体粒子
    for (const auto& particle : particles) {
        const Eigen::Vector3d& pos = particle.getPosition();
        
        if (pos.y() <= water_table_height) {
            // 创建水体粒子
            Particle water_particle;
            water_particle.setPosition(pos + Eigen::Vector3d(particle_radius * 0.5, 0, 0));
            water_particle.setRadius(particle_radius * 0.5);
            
            // 设置水体参数
            double volume = (4.0 / 3.0) * M_PI * std::pow(particle_radius * 0.5, 3);
            double water_density = 1000.0; // 水的密度
            double mass = volume * water_density;
            water_particle.setMass(mass);
            water_particle.setDensity(water_density);
            
            // 设置粒子类型为水体
            water_particle.setType(ParticleType::WATER);
            
            particles_with_water.push_back(water_particle);
        }
    }
    
    std::cout << "Added groundwater model with " << (particles_with_water.size() - particles.size()) << " water particles." << std::endl;
    return particles_with_water;
}

// 添加加固结构
std::vector<Particle> SlopeModelBuilder::addReinforcement(
    const std::vector<Particle>& particles,
    const Eigen::VectorXd& reinforcement_params,
    double particle_radius
) {
    // 简化实现：目前不支持加固结构
    std::cout << "Reinforcement addition not implemented yet." << std::endl;
    return particles;
}

// 设置模型类型
void SlopeModelBuilder::setModelType(SlopeModelType model_type) {
    model_type_ = model_type;
}

// 获取模型类型
SlopeModelType SlopeModelBuilder::getModelType() const {
    return model_type_;
}

// 设置粒子生成器
void SlopeModelBuilder::setParticleGenerator(const ParticleGenerator& particle_generator) {
    particle_generator_ = particle_generator;
}

// 检查点是否在边坡区域内
bool SlopeModelBuilder::isPointInSlope(
    const Eigen::Vector3d& point,
    const SlopeGeometryParams& geometry_params,
    bool is_3d
) const {
    // 简化实现：只考虑单级边坡
    
    double slope_height = geometry_params.slope_height;
    double slope_angle = geometry_params.slope_angle;
    double ground_depth = geometry_params.ground_depth;
    double slope_width = geometry_params.slope_width;
    
    // 检查y方向范围（垂直方向）
    if (point.y() < -ground_depth || point.y() > slope_height) {
        return false;
    }
    
    // 检查x方向范围（水平方向，边坡走向）
    if (point.x() < -slope_width / 2.0 || point.x() > slope_width / 2.0) {
        return false;
    }
    
    // 检查z方向范围（三维模型）
    if (is_3d) {
        if (point.z() < -geometry_params.slope_length / 2.0 || point.z() > geometry_params.slope_length / 2.0) {
            return false;
        }
    }
    
    // 检查是否在边坡区域内
    // 边坡线方程：y = slope_height - x * tan(slope_angle)
    double slope_rad = slope_angle * M_PI / 180.0;
    double slope_ratio = tan(slope_rad);
    
    // 对于x > 0的区域，y必须小于等于边坡线
    if (point.x() > 0.0) {
        double slope_y = slope_height - point.x() * slope_ratio;
        if (point.y() > slope_y) {
            return false;
        }
    }
    
    return true;
}

// 计算边坡的边界范围
std::pair<Eigen::Vector3d, Eigen::Vector3d> SlopeModelBuilder::calculateSlopeBounds(
    const SlopeGeometryParams& geometry_params,
    bool is_3d
) const {
    Eigen::Vector3d min_bound, max_bound;
    
    double slope_height = geometry_params.slope_height;
    double slope_angle = geometry_params.slope_angle;
    double slope_width = geometry_params.slope_width;
    double ground_depth = geometry_params.ground_depth;
    
    // 计算最大水平范围
    double slope_rad = slope_angle * M_PI / 180.0;
    double slope_length_horizontal = slope_height / tan(slope_rad);
    
    // 设置x方向边界
    min_bound.x() = -slope_width / 2.0;
    max_bound.x() = slope_width / 2.0 + slope_length_horizontal;
    
    // 设置y方向边界
    min_bound.y() = -ground_depth;
    max_bound.y() = slope_height;
    
    // 设置z方向边界（三维模型）
    if (is_3d) {
        min_bound.z() = -geometry_params.slope_length / 2.0;
        max_bound.z() = geometry_params.slope_length / 2.0;
    } else {
        min_bound.z() = -0.1;
        max_bound.z() = 0.1;
    }
    
    return {min_bound, max_bound};
}

// 生成三维网格点云
std::vector<Eigen::Vector3d> SlopeModelBuilder::generateGridPoints(
    const Eigen::Vector3d& min_bound,
    const Eigen::Vector3d& max_bound,
    double particle_radius
) const {
    std::vector<Eigen::Vector3d> grid_points;
    
    // 计算网格间距
    double spacing = particle_radius * 2.0;
    
    // 生成网格点
    for (double x = min_bound.x(); x <= max_bound.x(); x += spacing) {
        for (double y = min_bound.y(); y <= max_bound.y(); y += spacing) {
            for (double z = min_bound.z(); z <= max_bound.z(); z += spacing) {
                grid_points.emplace_back(x, y, z);
            }
        }
    }
    
    return grid_points;
}

} // namespace particle_simulation
