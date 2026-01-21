// 边坡模型构建源文件
// 实现边坡模型的参数化构建、外部模型导入/导出以及模型编辑功能

#include "slope/slope_model_builder.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <set>

namespace particle_simulation {

// ----------------------------------------
// SlopeModelBuilder 实现
// ----------------------------------------

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
    size_t particle_id = 0;
    for (const auto& point : grid_points) {
        if (isPointInSlope(point, geometry_params, is_3d)) {
            // 创建粒子
            Particle particle(particle_id++, point, ParticleType::SOIL);
            
            // 设置粒子参数
            particle.setRadius(particle_radius);
            
            // 设置粒子质量和密度
            double volume = (4.0 / 3.0) * M_PI * std::pow(particle_radius, 3);
            double mass = volume * geomaterial_params.density;
            particle.setMass(mass);
            particle.setDensity(geomaterial_params.density);
            
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
    size_t particle_id = 0;
    
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
            Particle particle(particle_id++, point, ParticleType::SOIL);
            
            // 设置粒子参数
            particle.setRadius(particle_radius);
            
            // 设置粒子质量和密度
            double volume = (4.0 / 3.0) * M_PI * std::pow(particle_radius, 3);
            double mass = volume * geomaterial_params.density;
            particle.setMass(mass);
            particle.setDensity(geomaterial_params.density);
            
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
    
    std::cout << "Imported external model with " << particles.size() << " particles." << std::endl;
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
    size_t particle_id = particles.size();
    
    // 计算地下水位高度（假设地面高度为0）
    double water_table_height = -water_table_depth;
    
    // 为地下水位以下的区域添加水体粒子
    for (const auto& particle : particles) {
        const Eigen::Vector3d& pos = particle.getPosition();
        
        if (pos.y() <= water_table_height) {
            // 创建水体粒子
            Particle water_particle(particle_id++, pos + Eigen::Vector3d(particle_radius * 0.5, 0, 0), ParticleType::WATER);
            
            // 设置水体参数
            double volume = (4.0 / 3.0) * M_PI * std::pow(particle_radius * 0.5, 3);
            double water_density = 1000.0; // 水的密度
            double mass = volume * water_density;
            
            water_particle.setRadius(particle_radius * 0.5);
            water_particle.setMass(mass);
            water_particle.setDensity(water_density);
            
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
    std::vector<Particle> particles_with_reinforcement = particles;
    
    // 简化实现：目前不支持加固结构
    std::cout << "Reinforcement addition not implemented yet." << std::endl;
    
    return particles_with_reinforcement;
}

// 导出边坡模型
bool SlopeModelBuilder::exportSlopeModel(
    const std::vector<Particle>& particles,
    const std::string& file_path,
    const std::string& file_format
) {
    std::ofstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file for export: " << file_path << std::endl;
        return false;
    }
    
    // 写入文件头
    file << "# Slope model exported by Particle Slope Simulation Software" << std::endl;
    file << "# Format: " << file_format << std::endl;
    file << "# Number of particles: " << particles.size() << std::endl;
    file << "# x y z vx vy vz ax ay az density pressure" << std::endl;
    
    // 写入粒子数据
    for (const auto& particle : particles) {
        const auto& state = particle.getState();
        file << state.position.x() << " " << state.position.y() << " " << state.position.z() << " "
             << state.velocity.x() << " " << state.velocity.y() << " " << state.velocity.z() << " "
             << state.acceleration.x() << " " << state.acceleration.y() << " " << state.acceleration.z() << " "
             << state.density << " " << state.pressure << std::endl;
    }
    
    file.close();
    std::cout << "Slope model exported to: " << file_path << std::endl;
    return true;
}

// 设置模型类型
void SlopeModelBuilder::setModelType(SlopeModelType model_type) {
    model_type_ = model_type;
}

// 获取模型类型
SlopeModelType SlopeModelBuilder::getModelType() const {
    return model_type_;
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
    double slope_height = geometry_params.slope_height;
    double slope_angle = geometry_params.slope_angle;
    double ground_depth = geometry_params.ground_depth;
    double slope_width = geometry_params.slope_width;
    double slope_length = geometry_params.slope_length;
    
    // 计算边坡的水平范围
    double slope_rad = slope_angle * M_PI / 180.0;
    double horizontal_extent = slope_height / tan(slope_rad);
    
    // 计算边界
    Eigen::Vector3d min_bound,
                    max_bound;
    
    min_bound.x() = -slope_width / 2.0;
    max_bound.x() = slope_width / 2.0 + horizontal_extent;
    min_bound.y() = -ground_depth;
    max_bound.y() = slope_height;
    
    if (is_3d) {
        min_bound.z() = -slope_length / 2.0;
        max_bound.z() = slope_length / 2.0;
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

// ----------------------------------------
// SlopeModelEditor 实现
// ----------------------------------------

// 构造函数
SlopeModelEditor::SlopeModelEditor() {
}

// 预览边坡模型
std::vector<Particle> SlopeModelEditor::previewModel(
    const std::vector<Particle>& particles,
    double resolution
) {
    std::vector<Particle> preview_particles;
    
    // 简化实现：每隔几个粒子取一个作为预览
    for (size_t i = 0; i < particles.size(); i += static_cast<size_t>(1.0 / resolution)) {
        preview_particles.push_back(particles[i]);
    }
    
    std::cout << "Generated preview model with " << preview_particles.size() << " particles." << std::endl;
    return preview_particles;
}

// 修正边坡几何形状
std::vector<Particle> SlopeModelEditor::correctGeometry(
    const std::vector<Particle>& particles,
    double tolerance
) {
    std::vector<Particle> corrected_particles = particles;
    
    // 简化实现：移除重叠粒子
    std::set<size_t> particles_to_remove;
    
    for (size_t i = 0; i < corrected_particles.size(); ++i) {
        for (size_t j = i + 1; j < corrected_particles.size(); ++j) {
            double distance = (corrected_particles[i].getPosition() - corrected_particles[j].getPosition()).norm();
            if (distance < tolerance) {
                particles_to_remove.insert(j);
            }
        }
    }
    
    // 移除重叠粒子
    std::vector<Particle> temp_particles;
    for (size_t i = 0; i < corrected_particles.size(); ++i) {
        if (particles_to_remove.find(i) == particles_to_remove.end()) {
            temp_particles.push_back(corrected_particles[i]);
        }
    }
    
    corrected_particles.swap(temp_particles);
    
    std::cout << "Corrected geometry, removed " << particles_to_remove.size() << " overlapping particles." << std::endl;
    return corrected_particles;
}

// 局部细化边坡模型
std::vector<Particle> SlopeModelEditor::refineLocalRegion(
    const std::vector<Particle>& particles,
    const Eigen::AlignedBox3d& refinement_region,
    double new_particle_radius
) {
    std::vector<Particle> refined_particles;
    size_t particle_id = 0;
    
    // 保留非细化区域的粒子
    for (const auto& particle : particles) {
        if (!refinement_region.contains(particle.getPosition())) {
            Particle new_particle(particle_id++, particle);
            refined_particles.push_back(new_particle);
        }
    }
    
    // 细化区域生成新粒子
    double spacing = new_particle_radius * 2.0;
    
    for (double x = refinement_region.min().x(); x <= refinement_region.max().x(); x += spacing) {
        for (double y = refinement_region.min().y(); y <= refinement_region.max().y(); y += spacing) {
            for (double z = refinement_region.min().z(); z <= refinement_region.max().z(); z += spacing) {
                Eigen::Vector3d point(x, y, z);
                if (refinement_region.contains(point)) {
                    Particle new_particle(particle_id++, point, ParticleType::SOIL);
                    new_particle.setRadius(new_particle_radius);
                    refined_particles.push_back(new_particle);
                }
            }
        }
    }
    
    std::cout << "Refined local region, generated " << (refined_particles.size() - particles.size()) << " new particles." << std::endl;
    return refined_particles;
}

// 合并多个粒子模型
std::vector<Particle> SlopeModelEditor::mergeModels(
    const std::vector<std::vector<Particle>>& particle_lists
) {
    std::vector<Particle> merged_particles;
    size_t particle_id = 0;
    
    for (const auto& particle_list : particle_lists) {
        for (const auto& particle : particle_list) {
            Particle new_particle(particle_id++, particle);
            merged_particles.push_back(new_particle);
        }
    }
    
    std::cout << "Merged " << particle_lists.size() << " models, total particles: " << merged_particles.size() << std::endl;
    return merged_particles;
}

// 裁剪粒子模型
std::vector<Particle> SlopeModelEditor::clipModel(
    const std::vector<Particle>& particles,
    const Eigen::AlignedBox3d& clip_box
) {
    std::vector<Particle> clipped_particles;
    size_t particle_id = 0;
    
    for (const auto& particle : particles) {
        if (clip_box.contains(particle.getPosition())) {
            Particle new_particle(particle_id++, particle);
            clipped_particles.push_back(new_particle);
        }
    }
    
    std::cout << "Clipped model, remaining particles: " << clipped_particles.size() << " out of " << particles.size() << std::endl;
    return clipped_particles;
}

// 计算模型的凸包
std::vector<Particle> SlopeModelEditor::computeConvexHull(const std::vector<Particle>& particles) const {
    // 简化实现：返回所有粒子
    return particles;
}

// 移除重复粒子
std::vector<Particle> SlopeModelEditor::removeDuplicateParticles(
    const std::vector<Particle>& particles,
    double tolerance
) const {
    std::vector<Particle> unique_particles;
    std::set<size_t> particles_to_keep;
    
    // 简化实现：使用简单的距离检查
    for (size_t i = 0; i < particles.size(); ++i) {
        bool is_unique = true;
        for (size_t j = 0; j < i; ++j) {
            double distance = (particles[i].getPosition() - particles[j].getPosition()).norm();
            if (distance < tolerance) {
                is_unique = false;
                break;
            }
        }
        if (is_unique) {
            particles_to_keep.insert(i);
        }
    }
    
    for (size_t i : particles_to_keep) {
        unique_particles.push_back(particles[i]);
    }
    
    return unique_particles;
}

} // namespace particle_simulation
