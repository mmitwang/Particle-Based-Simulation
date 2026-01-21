// VisualizationModule.cpp
// 结果可视化模块实现

#include "VisualizationModule.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>

namespace particle_simulation {

// ----------------------------------------
// VisualizationModule 基类实现
// ----------------------------------------

// 构造函数
VisualizationModule::VisualizationModule(RenderingMode mode)
    : rendering_mode_(mode),
      output_dir_("./visualization"),
      visualization_type_(VisualizationType::PARTICLE_POSITIONS) {
    // 创建输出目录
    std::filesystem::create_directories(output_dir_);
}

// 设置渲染模式
void VisualizationModule::setRenderingMode(RenderingMode mode) {
    rendering_mode_ = mode;
}

// 获取渲染模式
RenderingMode VisualizationModule::getRenderingMode() const {
    return rendering_mode_;
}

// 设置输出目录
void VisualizationModule::setOutputDirectory(const std::string& output_dir) {
    output_dir_ = output_dir;
    std::filesystem::create_directories(output_dir_);
}

// 获取输出目录
std::string VisualizationModule::getOutputDirectory() const {
    return output_dir_;
}

// ----------------------------------------
// VisualizationModuleFactory 实现
// ----------------------------------------

// 创建可视化模块实例
std::unique_ptr<VisualizationModule> VisualizationModuleFactory::createVisualizationModule(RenderingMode mode) {
    std::unique_ptr<VisualizationModule> visualization_module;
    
    switch (mode) {
        case RenderingMode::DYNAMIC:
        case RenderingMode::INTERACTIVE:
            // 默认使用VTK可视化模块
            visualization_module = std::make_unique<VTKVisualization>();
            break;
        case RenderingMode::STATIC:
            // 静态可视化使用Python模块
            visualization_module = std::make_unique<PythonVisualization>();
            break;
        default:
            // 默认使用文本可视化模块
            visualization_module = std::make_unique<TextVisualization>();
            break;
    }
    
    return visualization_module;
}

// ----------------------------------------
// VTKVisualization 实现
// ----------------------------------------

// 构造函数
VTKVisualization::VTKVisualization()
    : VisualizationModule(RenderingMode::DYNAMIC),
      renderer_(nullptr),
      render_window_(nullptr),
      interactor_(nullptr) {
    // 初始化VTK渲染器
    // 注意：这里使用空指针，实际项目中需要初始化VTK相关对象
}

// 初始化可视化模块
bool VTKVisualization::initialize(
    int window_width,
    int window_height,
    const VisualizationParams& params
) {
    std::cout << "Initializing VTK visualization module..." << std::endl;
    
    // 注意：实际项目中需要初始化VTK渲染器、渲染窗口和交互器
    // 这里仅做简化实现
    
    // 设置可视化参数
    setVisualizationParams(params);
    
    std::cout << "VTK visualization module initialized." << std::endl;
    return true;
}

// 更新可视化内容
bool VTKVisualization::updateVisualization(
    const std::vector<Particle>& particles,
    double simulation_time,
    size_t time_step
) {
    // 注意：实际项目中需要更新VTK渲染场景
    // 这里仅做简化实现
    
    std::cout << "Updating VTK visualization: time step = " << time_step << ", simulation time = " << simulation_time << "s" << std::endl;
    
    return true;
}

// 渲染可视化场景
bool VTKVisualization::render() {
    // 注意：实际项目中需要调用VTK的渲染函数
    // 这里仅做简化实现
    
    return true;
}

// 保存可视化结果
bool VTKVisualization::saveVisualization(
    const std::string& file_path,
    const std::string& file_format
) {
    // 注意：实际项目中需要调用VTK的保存函数
    // 这里仅做简化实现
    
    std::cout << "Saving VTK visualization to: " << file_path << ", format: " << file_format << std::endl;
    
    return true;
}

// 设置可视化类型
void VTKVisualization::setVisualizationType(VisualizationType type) {
    visualization_type_ = type;
    
    // 注意：实际项目中需要更新VTK的可视化类型
    std::cout << "Setting VTK visualization type to: " << static_cast<int>(type) << std::endl;
}

// 设置可视化参数
void VTKVisualization::setVisualizationParams(const VisualizationParams& params) {
    visualization_params_ = params;
    
    // 注意：实际项目中需要更新VTK的可视化参数
    std::cout << "Setting VTK visualization parameters." << std::endl;
}

// 记录粒子轨迹
void VTKVisualization::recordParticleTrajectories(
    const std::vector<Particle>& particles,
    double simulation_time
) {
    // 初始化轨迹记录（如果尚未初始化）
    if (particle_trajectories_.empty()) {
        particle_trajectories_.reserve(particles.size());
        for (size_t i = 0; i < particles.size(); ++i) {
            particle_trajectories_.emplace_back(i);
        }
    }
    
    // 更新轨迹记录
    for (size_t i = 0; i < particles.size(); ++i) {
        particle_trajectories_[i].positions.push_back(particles[i].getPosition());
        particle_trajectories_[i].times.push_back(simulation_time);
    }
}

// 可视化粒子轨迹
void VTKVisualization::visualizeParticleTrajectories(
    const std::vector<size_t>& particle_ids
) {
    // 注意：实际项目中需要在VTK中渲染粒子轨迹
    // 这里仅做简化实现
    
    std::cout << "Visualizing particle trajectories." << std::endl;
}

// 可视化稳定性评估结果
void VTKVisualization::visualizeStabilityResults(
    const StabilityIndices& indices,
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles
) {
    // 注意：实际项目中需要在VTK中渲染稳定性评估结果
    // 这里仅做简化实现
    
    std::cout << "Visualizing stability results: safety factor = " << indices.safety_factor << std::endl;
}

// 清除可视化内容
void VTKVisualization::clear() {
    // 注意：实际项目中需要清除VTK渲染场景
    // 这里仅做简化实现
    
    std::cout << "Clearing VTK visualization." << std::endl;
}

// 关闭可视化模块
void VTKVisualization::shutdown() {
    // 注意：实际项目中需要释放VTK相关资源
    // 这里仅做简化实现
    
    std::cout << "Shutting down VTK visualization module." << std::endl;
}

// ----------------------------------------
// PythonVisualization 实现
// ----------------------------------------

// 构造函数
PythonVisualization::PythonVisualization()
    : VisualizationModule(RenderingMode::STATIC),
      python_engine_(nullptr),
      initialized_(false) {
}

// 初始化可视化模块
bool PythonVisualization::initialize(
    int window_width,
    int window_height,
    const VisualizationParams& params
) {
    std::cout << "Initializing Python visualization module..." << std::endl;
    
    // 注意：实际项目中需要初始化Python解释器和可视化库（如Matplotlib、Mayavi等）
    // 这里仅做简化实现
    
    // 设置可视化参数
    setVisualizationParams(params);
    
    initialized_ = true;
    std::cout << "Python visualization module initialized." << std::endl;
    return true;
}

// 更新可视化内容
bool PythonVisualization::updateVisualization(
    const std::vector<Particle>& particles,
    double simulation_time,
    size_t time_step
) {
    if (!initialized_) {
        std::cerr << "Python visualization module not initialized!" << std::endl;
        return false;
    }
    
    // 注意：实际项目中需要更新Python可视化场景
    // 这里仅做简化实现
    
    std::cout << "Updating Python visualization: time step = " << time_step << ", simulation time = " << simulation_time << "s" << std::endl;
    
    return true;
}

// 渲染可视化场景
bool PythonVisualization::render() {
    if (!initialized_) {
        std::cerr << "Python visualization module not initialized!" << std::endl;
        return false;
    }
    
    // 注意：实际项目中需要调用Python的渲染函数
    // 这里仅做简化实现
    
    return true;
}

// 保存可视化结果
bool PythonVisualization::saveVisualization(
    const std::string& file_path,
    const std::string& file_format
) {
    if (!initialized_) {
        std::cerr << "Python visualization module not initialized!" << std::endl;
        return false;
    }
    
    // 注意：实际项目中需要调用Python的保存函数
    // 这里仅做简化实现
    
    std::cout << "Saving Python visualization to: " << file_path << ", format: " << file_format << std::endl;
    
    return true;
}

// 设置可视化类型
void PythonVisualization::setVisualizationType(VisualizationType type) {
    visualization_type_ = type;
    
    // 注意：实际项目中需要更新Python的可视化类型
    std::cout << "Setting Python visualization type to: " << static_cast<int>(type) << std::endl;
}

// 设置可视化参数
void PythonVisualization::setVisualizationParams(const VisualizationParams& params) {
    visualization_params_ = params;
    
    // 注意：实际项目中需要更新Python的可视化参数
    std::cout << "Setting Python visualization parameters." << std::endl;
}

// 记录粒子轨迹
void PythonVisualization::recordParticleTrajectories(
    const std::vector<Particle>& particles,
    double simulation_time
) {
    if (!initialized_) {
        return;
    }
    
    // 初始化轨迹记录（如果尚未初始化）
    if (particle_trajectories_.empty()) {
        particle_trajectories_.reserve(particles.size());
        for (size_t i = 0; i < particles.size(); ++i) {
            particle_trajectories_.emplace_back(i);
        }
    }
    
    // 更新轨迹记录
    for (size_t i = 0; i < particles.size(); ++i) {
        particle_trajectories_[i].positions.push_back(particles[i].getPosition());
        particle_trajectories_[i].times.push_back(simulation_time);
    }
}

// 可视化粒子轨迹
void PythonVisualization::visualizeParticleTrajectories(
    const std::vector<size_t>& particle_ids
) {
    if (!initialized_) {
        return;
    }
    
    // 注意：实际项目中需要在Python中渲染粒子轨迹
    // 这里仅做简化实现
    
    std::cout << "Visualizing particle trajectories using Python." << std::endl;
}

// 可视化稳定性评估结果
void PythonVisualization::visualizeStabilityResults(
    const StabilityIndices& indices,
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles
) {
    if (!initialized_) {
        return;
    }
    
    // 注意：实际项目中需要在Python中渲染稳定性评估结果
    // 这里仅做简化实现
    
    std::cout << "Visualizing stability results using Python." << std::endl;
}

// 清除可视化内容
void PythonVisualization::clear() {
    if (!initialized_) {
        return;
    }
    
    // 注意：实际项目中需要清除Python可视化场景
    // 这里仅做简化实现
    
    std::cout << "Clearing Python visualization." << std::endl;
}

// 关闭可视化模块
void PythonVisualization::shutdown() {
    if (!initialized_) {
        return;
    }
    
    // 注意：实际项目中需要释放Python相关资源
    // 这里仅做简化实现
    
    std::cout << "Shutting down Python visualization module." << std::endl;
    initialized_ = false;
}

// ----------------------------------------
// TextVisualization 实现
// ----------------------------------------

// 构造函数
TextVisualization::TextVisualization()
    : VisualizationModule(RenderingMode::STATIC),
      initialized_(false) {
}

// 初始化可视化模块
bool TextVisualization::initialize(
    int window_width,
    int window_height,
    const VisualizationParams& params
) {
    std::cout << "Initializing text visualization module..." << std::endl;
    
    // 设置可视化参数
    setVisualizationParams(params);
    
    initialized_ = true;
    std::cout << "Text visualization module initialized." << std::endl;
    return true;
}

// 更新可视化内容
bool TextVisualization::updateVisualization(
    const std::vector<Particle>& particles,
    double simulation_time,
    size_t time_step
) {
    if (!initialized_) {
        std::cerr << "Text visualization module not initialized!" << std::endl;
        return false;
    }
    
    std::stringstream ss;
    ss << "Time Step: " << time_step << ", Simulation Time: " << simulation_time << "s\n";
    ss << "Total Particles: " << particles.size() << "\n";
    
    // 根据可视化类型更新数据
    switch (visualization_type_) {
        case VisualizationType::PARTICLE_POSITIONS:
            ss << "Visualization Type: Particle Positions\n";
            ss << "First 5 particle positions:\n";
            for (size_t i = 0; i < std::min(size_t(5), particles.size()); ++i) {
                const Eigen::Vector3d& pos = particles[i].getPosition();
                ss << "  Particle " << i << ": (" << pos.x() << ", " << pos.y() << ", " << pos.z() << ")\n";
            }
            break;
            
        case VisualizationType::PARTICLE_VELOCITIES:
            ss << "Visualization Type: Particle Velocities\n";
            ss << "First 5 particle velocities:\n";
            for (size_t i = 0; i < std::min(size_t(5), particles.size()); ++i) {
                const Eigen::Vector3d& vel = particles[i].getVelocity();
                ss << "  Particle " << i << ": (" << vel.x() << ", " << vel.y() << ", " << vel.z() << "), Magnitude: " << vel.norm() << "\n";
            }
            break;
            
        case VisualizationType::PARTICLE_STRESSES:
            ss << "Visualization Type: Particle Stresses\n";
            ss << "First 5 particle stresses:\n";
            for (size_t i = 0; i < std::min(size_t(5), particles.size()); ++i) {
                double stress_magnitude = particles[i].getStress().norm();
                ss << "  Particle " << i << ": Stress Magnitude: " << stress_magnitude << "\n";
            }
            break;
            
        case VisualizationType::PARTICLE_STRAINS:
            ss << "Visualization Type: Particle Strains\n";
            ss << "First 5 particle strains:\n";
            for (size_t i = 0; i < std::min(size_t(5), particles.size()); ++i) {
                double strain_magnitude = particles[i].getStrain().norm();
                ss << "  Particle " << i << ": Strain Magnitude: " << strain_magnitude << "\n";
            }
            break;
            
        default:
            ss << "Visualization Type: " << static_cast<int>(visualization_type_) << "\n";
            break;
    }
    
    current_data_ = ss.str();
    
    // 渲染静态文本输出
    render();
    
    return true;
}

// 渲染可视化场景
bool TextVisualization::render() {
    if (!initialized_) {
        std::cerr << "Text visualization module not initialized!" << std::endl;
        return false;
    }
    
    // 输出到控制台
    std::cout << "\n" << current_data_ << std::endl;
    
    return true;
}

// 保存可视化结果
bool TextVisualization::saveVisualization(
    const std::string& file_path,
    const std::string& file_format
) {
    if (!initialized_) {
        std::cerr << "Text visualization module not initialized!" << std::endl;
        return false;
    }
    
    // 保存为文本文件
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving: " << file_path << std::endl;
        return false;
    }
    
    file << current_data_;
    file.close();
    
    std::cout << "Text visualization saved to: " << file_path << std::endl;
    return true;
}

// 设置可视化类型
void TextVisualization::setVisualizationType(VisualizationType type) {
    visualization_type_ = type;
    
    std::cout << "Setting text visualization type to: " << static_cast<int>(type) << std::endl;
}

// 设置可视化参数
void TextVisualization::setVisualizationParams(const VisualizationParams& params) {
    visualization_params_ = params;
    
    std::cout << "Setting text visualization parameters." << std::endl;
}

// 记录粒子轨迹
void TextVisualization::recordParticleTrajectories(
    const std::vector<Particle>& particles,
    double simulation_time
) {
    if (!initialized_) {
        return;
    }
    
    // 初始化轨迹记录（如果尚未初始化）
    if (particle_trajectories_.empty()) {
        particle_trajectories_.reserve(particles.size());
        for (size_t i = 0; i < particles.size(); ++i) {
            particle_trajectories_.emplace_back(i);
        }
    }
    
    // 更新轨迹记录
    for (size_t i = 0; i < particles.size(); ++i) {
        particle_trajectories_[i].positions.push_back(particles[i].getPosition());
        particle_trajectories_[i].times.push_back(simulation_time);
    }
}

// 可视化粒子轨迹
void TextVisualization::visualizeParticleTrajectories(
    const std::vector<size_t>& particle_ids
) {
    if (!initialized_) {
        return;
    }
    
    std::cout << "Particle Trajectories:\n";
    
    // 选择要可视化的粒子
    std::vector<size_t> ids_to_visualize;
    if (particle_ids.empty()) {
        // 可视化前5个粒子
        for (size_t i = 0; i < std::min(size_t(5), particle_trajectories_.size()); ++i) {
            ids_to_visualize.push_back(i);
        }
    } else {
        ids_to_visualize = particle_ids;
    }
    
    // 输出轨迹信息
    for (size_t id : ids_to_visualize) {
        if (id < particle_trajectories_.size()) {
            const ParticleTrajectory& trajectory = particle_trajectories_[id];
            std::cout << "Particle " << id << " trajectory (" << trajectory.positions.size() << " points):\n";
            
            // 输出前3个和最后3个点
            for (size_t i = 0; i < std::min(size_t(3), trajectory.positions.size()); ++i) {
                std::cout << "  Time: " << trajectory.times[i] << "s, Position: (" << 
                    trajectory.positions[i].x() << ", " << trajectory.positions[i].y() << ", " << 
                    trajectory.positions[i].z() << ")\n";
            }
            
            if (trajectory.positions.size() > 6) {
                std::cout << "  ...\n";
            }
            
            for (size_t i = std::max(size_t(0), trajectory.positions.size() - 3); i < trajectory.positions.size(); ++i) {
                std::cout << "  Time: " << trajectory.times[i] << "s, Position: (" << 
                    trajectory.positions[i].x() << ", " << trajectory.positions[i].y() << ", " << 
                    trajectory.positions[i].z() << ")\n";
            }
        }
    }
}

// 可视化稳定性评估结果
void TextVisualization::visualizeStabilityResults(
    const StabilityIndices& indices,
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles
) {
    if (!initialized_) {
        return;
    }
    
    std::cout << "\nStability Assessment Results:\n";
    std::cout << "========================================\n";
    std::cout << "Safety Factor: " << indices.safety_factor << "\n";
    std::cout << "Max Displacement: " << indices.max_displacement << "m\n";
    std::cout << "Average Displacement: " << indices.average_displacement << "m\n";
    std::cout << "Displacement Variance: " << indices.displacement_variance << "m²\n";
    std::cout << "Max Stress: " << indices.max_stress << "Pa\n";
    std::cout << "Max Strain: " << indices.max_strain << "\n";
    std::cout << "Potential Slide Surface Points: " << indices.potential_slide_surface.size() << "\n";
    std::cout << "Critical Stress Areas: " << indices.critical_stress_area << "\n";
    std::cout << "Instability Warning Level: " << indices.instability_warning_level << " (0-100)\n";
    std::cout << "========================================\n";
}

// 清除可视化内容
void TextVisualization::clear() {
    if (!initialized_) {
        return;
    }
    
    current_data_.clear();
    std::cout << "Text visualization cleared.\n";
}

// 关闭可视化模块
void TextVisualization::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "Shutting down text visualization module.\n";
    initialized_ = false;
}

} // namespace particle_simulation
