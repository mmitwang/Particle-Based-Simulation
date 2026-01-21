// VisualizationModule.h
// 结果可视化模块
// 实现粒子运动轨迹、物理量分布的动态渲染与静态输出

#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <memory>
#include "Particle.h"
#include "StabilityEvaluator.h"

namespace particle_simulation {

// 可视化类型枚举
enum class VisualizationType {
    PARTICLE_POSITIONS,     // 粒子位置
    PARTICLE_VELOCITIES,    // 粒子速度
    PARTICLE_STRESSES,      // 粒子应力
    PARTICLE_STRAINS,       // 粒子应变
    DISPLACEMENT_FIELD,     // 位移场
    STRESS_FIELD,           // 应力场
    VELOCITY_FIELD,         // 速度场
    PARTICLE_TRAJECTORIES,  // 粒子轨迹
    POTENTIAL_SLIDE_SURFACE // 潜在滑动面
};

// 可视化渲染模式枚举
enum class RenderingMode {
    DYNAMIC,    // 动态实时渲染
    STATIC,     // 静态图像输出
    INTERACTIVE // 交互式可视化
};

// 可视化参数结构体
/**
 * @brief 可视化参数结构体
 * 
 * 包含可视化的基本参数
 * 
 * @param particle_size 粒子渲染大小
 * @param color_map 颜色映射类型
 * @param opacity 透明度
 * @param background_color 背景颜色
 * @param lighting 是否启用光照
 * @param show_axes 是否显示坐标轴
 * @param show_legend 是否显示图例
 */
struct VisualizationParams {
    double particle_size;           // 粒子渲染大小
    std::string color_map;          // 颜色映射类型
    double opacity;                 // 透明度
    Eigen::Vector3d background_color; // 背景颜色
    bool lighting;                  // 是否启用光照
    bool show_axes;                 // 是否显示坐标轴
    bool show_legend;               // 是否显示图例
    
    // 构造函数
    VisualizationParams() :
        particle_size(1.0),
        color_map("viridis"),
        opacity(1.0),
        background_color(1.0, 1.0, 1.0),
        lighting(true),
        show_axes(true),
        show_legend(true) {
    }
};

// 粒子轨迹记录结构体
/**
 * @brief 粒子轨迹记录结构体
 * 
 * 记录粒子的运动轨迹
 * 
 * @param particle_id 粒子ID
 * @param positions 粒子位置历史记录
 * @param times 时间历史记录
 */
struct ParticleTrajectory {
    size_t particle_id;                     // 粒子ID
    std::vector<Eigen::Vector3d> positions; // 位置历史记录
    std::vector<double> times;              // 时间历史记录
    
    // 构造函数
    ParticleTrajectory(size_t id) : particle_id(id) {
    }
};

// 可视化模块基类
/**
 * @brief 可视化模块基类
 * 
 * 实现粒子系统的可视化功能，包括动态渲染和静态输出
 * 
 * 核心功能：
 * 1. 粒子位置、速度、应力、应变的可视化
 * 2. 位移场、应力场、速度场的可视化
 * 3. 粒子运动轨迹的记录和可视化
 * 4. 潜在滑动面的可视化
 * 5. 动态渲染和静态图像输出
 * 
 * 设计思路：
 * - 采用抽象基类设计，支持多种可视化后端
 * - 支持实时动态渲染和静态结果输出
 * - 支持多种物理量的可视化
 * - 支持大规模粒子系统的高效可视化
 */
class VisualizationModule {
public:
    // 构造函数
    VisualizationModule(RenderingMode mode = RenderingMode::DYNAMIC);
    
    // 析构函数
    virtual ~VisualizationModule() = default;
    
    /**
     * @brief 初始化可视化模块
     * 
     * 初始化可视化环境和参数
     * 
     * @param window_width 窗口宽度
     * @param window_height 窗口高度
     * @param params 可视化参数
     * @return 是否初始化成功
     */
    virtual bool initialize(
        int window_width = 800,
        int window_height = 600,
        const VisualizationParams& params = VisualizationParams()
    ) = 0;
    
    /**
     * @brief 更新可视化内容
     * 
     * 更新可视化场景，渲染最新的粒子状态
     * 
     * @param particles 粒子列表
     * @param simulation_time 当前仿真时间
     * @param time_step 当前时间步
     * @return 是否更新成功
     */
    virtual bool updateVisualization(
        const std::vector<Particle>& particles,
        double simulation_time,
        size_t time_step = 0
    ) = 0;
    
    /**
     * @brief 渲染可视化场景
     * 
     * 渲染当前可视化场景
     * 
     * @return 是否渲染成功
     */
    virtual bool render() = 0;
    
    /**
     * @brief 保存可视化结果
     * 
     * 保存当前可视化场景为图像或视频
     * 
     * @param file_path 文件路径
     * @param file_format 文件格式
     * @return 是否保存成功
     */
    virtual bool saveVisualization(
        const std::string& file_path,
        const std::string& file_format = "png"
    ) = 0;
    
    /**
     * @brief 设置可视化类型
     * 
     * 设置当前要可视化的物理量类型
     * 
     * @param type 可视化类型
     */
    virtual void setVisualizationType(VisualizationType type) = 0;
    
    /**
     * @brief 设置可视化参数
     * 
     * 设置可视化的基本参数
     * 
     * @param params 可视化参数
     */
    virtual void setVisualizationParams(const VisualizationParams& params) = 0;
    
    /**
     * @brief 记录粒子轨迹
     * 
     * 记录粒子的运动轨迹
     * 
     * @param particles 粒子列表
     * @param simulation_time 当前仿真时间
     */
    virtual void recordParticleTrajectories(
        const std::vector<Particle>& particles,
        double simulation_time
    ) = 0;
    
    /**
     * @brief 可视化粒子轨迹
     * 
     * 渲染记录的粒子轨迹
     * 
     * @param particle_ids 要可视化的粒子ID列表
     */
    virtual void visualizeParticleTrajectories(
        const std::vector<size_t>& particle_ids = {}
    ) = 0;
    
    /**
     * @brief 可视化稳定性评估结果
     * 
     * 渲染边坡稳定性评估结果
     * 
     * @param indices 稳定性指标
     * @param particles 粒子列表
     * @param initial_particles 初始粒子列表
     */
    virtual void visualizeStabilityResults(
        const StabilityIndices& indices,
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    ) = 0;
    
    /**
     * @brief 清除可视化内容
     * 
     * 清除当前可视化场景
     */
    virtual void clear() = 0;
    
    /**
     * @brief 关闭可视化模块
     * 
     * 关闭可视化环境，释放资源
     */
    virtual void shutdown() = 0;
    
    /**
     * @brief 设置渲染模式
     * 
     * 设置可视化的渲染模式
     * 
     * @param mode 渲染模式
     */
    void setRenderingMode(RenderingMode mode);
    
    /**
     * @brief 获取渲染模式
     * 
     * @return 渲染模式
     */
    RenderingMode getRenderingMode() const;
    
    /**
     * @brief 设置输出目录
     * 
     * 设置可视化结果的输出目录
     * 
     * @param output_dir 输出目录
     */
    void setOutputDirectory(const std::string& output_dir);
    
    /**
     * @brief 获取输出目录
     * 
     * @return 输出目录
     */
    std::string getOutputDirectory() const;
    
protected:
    RenderingMode rendering_mode_;       // 渲染模式
    std::string output_dir_;              // 输出目录
    VisualizationParams visualization_params_; // 可视化参数
    VisualizationType visualization_type_; // 可视化类型
    
    // 粒子轨迹记录
    std::vector<ParticleTrajectory> particle_trajectories_;
};

// 可视化模块工厂类
/**
 * @brief 可视化模块工厂类
 * 
 * 用于创建不同类型的可视化模块实例
 */
class VisualizationModuleFactory {
public:
    /**
     * @brief 创建可视化模块实例
     * 
     * 根据渲染模式创建对应的可视化模块实例
     * 
     * @param mode 渲染模式
     * @return 可视化模块实例指针
     */
    static std::unique_ptr<VisualizationModule> createVisualizationModule(RenderingMode mode);
};

// 基于VTK的可视化模块（预留接口）
class VTKVisualization : public VisualizationModule {
public:
    // 构造函数
    VTKVisualization();
    
    // 初始化可视化模块
    bool initialize(
        int window_width = 800,
        int window_height = 600,
        const VisualizationParams& params = VisualizationParams()
    ) override;
    
    // 更新可视化内容
    bool updateVisualization(
        const std::vector<Particle>& particles,
        double simulation_time,
        size_t time_step = 0
    ) override;
    
    // 渲染可视化场景
    bool render() override;
    
    // 保存可视化结果
    bool saveVisualization(
        const std::string& file_path,
        const std::string& file_format = "png"
    ) override;
    
    // 设置可视化类型
    void setVisualizationType(VisualizationType type) override;
    
    // 设置可视化参数
    void setVisualizationParams(const VisualizationParams& params) override;
    
    // 记录粒子轨迹
    void recordParticleTrajectories(
        const std::vector<Particle>& particles,
        double simulation_time
    ) override;
    
    // 可视化粒子轨迹
    void visualizeParticleTrajectories(
        const std::vector<size_t>& particle_ids = {}
    ) override;
    
    // 可视化稳定性评估结果
    void visualizeStabilityResults(
        const StabilityIndices& indices,
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    ) override;
    
    // 清除可视化内容
    void clear() override;
    
    // 关闭可视化模块
    void shutdown() override;
    
private:
    // VTK渲染器指针（使用void*避免直接依赖VTK库）
    void* renderer_;
    void* render_window_;
    void* interactor_;
};

// 基于Python的可视化模块（预留接口）
class PythonVisualization : public VisualizationModule {
public:
    // 构造函数
    PythonVisualization();
    
    // 初始化可视化模块
    bool initialize(
        int window_width = 800,
        int window_height = 600,
        const VisualizationParams& params = VisualizationParams()
    ) override;
    
    // 更新可视化内容
    bool updateVisualization(
        const std::vector<Particle>& particles,
        double simulation_time,
        size_t time_step = 0
    ) override;
    
    // 渲染可视化场景
    bool render() override;
    
    // 保存可视化结果
    bool saveVisualization(
        const std::string& file_path,
        const std::string& file_format = "png"
    ) override;
    
    // 设置可视化类型
    void setVisualizationType(VisualizationType type) override;
    
    // 设置可视化参数
    void setVisualizationParams(const VisualizationParams& params) override;
    
    // 记录粒子轨迹
    void recordParticleTrajectories(
        const std::vector<Particle>& particles,
        double simulation_time
    ) override;
    
    // 可视化粒子轨迹
    void visualizeParticleTrajectories(
        const std::vector<size_t>& particle_ids = {}
    ) override;
    
    // 可视化稳定性评估结果
    void visualizeStabilityResults(
        const StabilityIndices& indices,
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    ) override;
    
    // 清除可视化内容
    void clear() override;
    
    // 关闭可视化模块
    void shutdown() override;
    
private:
    // Python可视化引擎指针（使用void*避免直接依赖Python库）
    void* python_engine_;
    bool initialized_;
};

// 简单的文本可视化模块（用于调试）
class TextVisualization : public VisualizationModule {
public:
    // 构造函数
    TextVisualization();
    
    // 初始化可视化模块
    bool initialize(
        int window_width = 800,
        int window_height = 600,
        const VisualizationParams& params = VisualizationParams()
    ) override;
    
    // 更新可视化内容
    bool updateVisualization(
        const std::vector<Particle>& particles,
        double simulation_time,
        size_t time_step = 0
    ) override;
    
    // 渲染可视化场景
    bool render() override;
    
    // 保存可视化结果
    bool saveVisualization(
        const std::string& file_path,
        const std::string& file_format = "txt"
    ) override;
    
    // 设置可视化类型
    void setVisualizationType(VisualizationType type) override;
    
    // 设置可视化参数
    void setVisualizationParams(const VisualizationParams& params) override;
    
    // 记录粒子轨迹
    void recordParticleTrajectories(
        const std::vector<Particle>& particles,
        double simulation_time
    ) override;
    
    // 可视化粒子轨迹
    void visualizeParticleTrajectories(
        const std::vector<size_t>& particle_ids = {}
    ) override;
    
    // 可视化稳定性评估结果
    void visualizeStabilityResults(
        const StabilityIndices& indices,
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    ) override;
    
    // 清除可视化内容
    void clear() override;
    
    // 关闭可视化模块
    void shutdown() override;
    
private:
    std::string current_data_;
    bool initialized_;
};

} // namespace particle_simulation
