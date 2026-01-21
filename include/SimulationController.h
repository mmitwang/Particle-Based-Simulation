// SimulationController.h
// 仿真过程控制模块
// 实现仿真的启动、暂停、继续和停止控制，以及实时监控和中断续算

#pragma once

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <Eigen/Dense>
#include "Particle.h"
#include "NeighborSearcher.h"
#include "ConstitutiveModel.h"
#include "ContactDetection.h"
#include "TimeIntegrator.h"
#include "SlopeModelBuilder.h"

namespace particle_simulation {

// 仿真状态枚举
enum class SimulationState {
    IDLE,           // 空闲状态
    INITIALIZING,   // 初始化中
    RUNNING,        // 运行中
    PAUSED,         // 已暂停
    FINISHED,       // 已完成
    ERROR           // 错误状态
};

// 仿真终止条件结构体
/**
 * @brief 仿真终止条件结构体
 * 
 * 包含仿真的终止条件参数
 * 
 * @param max_time_steps 最大时间步数
 * @param max_simulation_time 最大仿真时间（秒）
 * @param min_displacement 最小位移阈值（m）
 * @param max_displacement 最大位移阈值（m）
 * @param convergence_threshold 收敛阈值
 */
struct TerminationConditions {
    size_t max_time_steps;       // 最大时间步数
    double max_simulation_time;  // 最大仿真时间（秒）
    double min_displacement;     // 最小位移阈值
    double max_displacement;     // 最大位移阈值
    double convergence_threshold; // 收敛阈值
    
    // 构造函数
    TerminationConditions() :
        max_time_steps(10000),
        max_simulation_time(100.0),
        min_displacement(1.0e-6),
        max_displacement(10.0),
        convergence_threshold(1.0e-4) {
    }
    
    TerminationConditions(
        size_t max_steps,
        double max_time,
        double min_disp,
        double max_disp,
        double conv_threshold
    ) :
        max_time_steps(max_steps),
        max_simulation_time(max_time),
        min_displacement(min_disp),
        max_displacement(max_disp),
        convergence_threshold(conv_threshold) {
    }
};

// 仿真统计信息结构体
/**
 * @brief 仿真统计信息结构体
 * 
 * 包含仿真过程中的统计信息
 * 
 * @param current_time_step 当前时间步数
 * @param current_simulation_time 当前仿真时间（秒）
 * @param total_particles 总粒子数
 * @param average_displacement 平均位移（m）
 * @param max_displacement 最大位移（m）
 * @param average_velocity 平均速度（m/s）
 * @param max_velocity 最大速度（m/s）
 * @param computational_time 计算时间（秒）
 * @param penetration_count 粒子穿透数
 */
struct SimulationStats {
    size_t current_time_step;       // 当前时间步数
    double current_simulation_time; // 当前仿真时间（秒）
    size_t total_particles;         // 总粒子数
    double average_displacement;    // 平均位移
    double max_displacement;        // 最大位移
    double average_velocity;        // 平均速度
    double max_velocity;            // 最大速度
    double computational_time;      // 计算时间（秒）
    size_t penetration_count;       // 粒子穿透数
    
    // 构造函数
    SimulationStats() :
        current_time_step(0),
        current_simulation_time(0.0),
        total_particles(0),
        average_displacement(0.0),
        max_displacement(0.0),
        average_velocity(0.0),
        max_velocity(0.0),
        computational_time(0.0),
        penetration_count(0) {
    }
};

// 仿真过程控制类
/**
 * @brief 仿真过程控制类
 * 
 * 实现仿真的启动、暂停、继续和停止控制，以及实时监控和中断续算功能
 * 
 * 核心功能：
 * 1. 仿真的初始化和配置
 * 2. 仿真的启动、暂停、继续和停止
 * 3. 实时监控仿真状态和统计信息
 * 4. 支持中断续算功能
 * 5. 管理仿真的时间步长和终止条件
 * 
 * 设计思路：
 * - 采用多线程设计，将仿真计算与UI控制分离
 * - 使用状态机管理仿真的不同状态
 * - 提供丰富的回调接口，支持实时监控和数据采集
 * - 支持仿真结果的保存和加载，实现中断续算
 */
class SimulationController {
public:
    // 仿真回调函数类型定义
    using ProgressCallback = std::function<void(const SimulationStats&)>;
    using StateChangeCallback = std::function<void(SimulationState)>;
    using DataCallback = std::function<void(const std::vector<Particle>&, double)>;
    
    // 构造函数
    SimulationController();
    
    // 析构函数
    ~SimulationController();
    
    /**
     * @brief 初始化仿真
     * 
     * 初始化仿真环境和参数
     * 
     * @param particles 粒子列表
     * @param neighbor_searcher 邻域搜索器
     * @param constitutive_model 本构模型
     * @param contact_detection 接触检测模块
     * @param time_integrator 时间积分器
     * @return 是否初始化成功
     */
    bool initialize(
        const std::vector<Particle>& particles,
        const NeighborSearcher& neighbor_searcher,
        const ConstitutiveModel& constitutive_model,
        const ContactDetection& contact_detection,
        const TimeIntegrator& time_integrator
    );
    
    /**
     * @brief 设置仿真参数
     * 
     * 设置仿真的基本参数
     * 
     * @param time_step 时间步长（秒）
     * @param termination_conditions 终止条件
     * @param save_interval 保存间隔（时间步数）
     * @param stats_interval 统计信息更新间隔（时间步数）
     */
    void setSimulationParameters(
        double time_step,
        const TerminationConditions& termination_conditions,
        size_t save_interval = 100,
        size_t stats_interval = 10
    );
    
    /**
     * @brief 启动仿真
     * 
     * 启动仿真线程，开始执行仿真计算
     * 
     * @return 是否成功启动
     */
    bool start();
    
    /**
     * @brief 暂停仿真
     * 
     * 暂停正在运行的仿真
     * 
     * @return 是否成功暂停
     */
    bool pause();
    
    /**
     * @brief 继续仿真
     * 
     * 继续已暂停的仿真
     * 
     * @return 是否成功继续
     */
    bool resume();
    
    /**
     * @brief 停止仿真
     * 
     * 停止正在运行的仿真
     * 
     * @return 是否成功停止
     */
    bool stop();
    
    /**
     * @brief 保存仿真状态
     * 
     * 保存当前仿真状态，用于中断续算
     * 
     * @param file_path 保存文件路径
     * @return 是否成功保存
     */
    bool saveSimulationState(const std::string& file_path);
    
    /**
     * @brief 加载仿真状态
     * 
     * 加载已保存的仿真状态，实现中断续算
     * 
     * @param file_path 加载文件路径
     * @return 是否成功加载
     */
    bool loadSimulationState(const std::string& file_path);
    
    /**
     * @brief 获取当前仿真状态
     * 
     * @return 当前仿真状态
     */
    SimulationState getSimulationState() const;
    
    /**
     * @brief 获取当前仿真统计信息
     * 
     * @return 当前仿真统计信息
     */
    SimulationStats getSimulationStats() const;
    
    /**
     * @brief 获取当前粒子列表
     * 
     * @return 当前粒子列表
     */
    std::vector<Particle> getParticles() const;
    
    /**
     * @brief 设置进度回调函数
     * 
     * @param callback 进度回调函数
     */
    void setProgressCallback(ProgressCallback callback);
    
    /**
     * @brief 设置状态变化回调函数
     * 
     * @param callback 状态变化回调函数
     */
    void setStateChangeCallback(StateChangeCallback callback);
    
    /**
     * @brief 设置数据回调函数
     * 
     * @param callback 数据回调函数
     */
    void setDataCallback(DataCallback callback);
    
    /**
     * @brief 设置输出目录
     * 
     * 设置仿真结果的输出目录
     * 
     * @param output_dir 输出目录路径
     */
    void setOutputDirectory(const std::string& output_dir);
    
private:
    // 仿真线程函数
    void simulationThread();
    
    // 更新仿真统计信息
    void updateSimulationStats();
    
    // 检查终止条件
    bool checkTerminationConditions();
    
    // 保存仿真结果
    void saveSimulationResults(size_t time_step);
    
    // 粒子列表
    std::vector<Particle> particles_;
    std::vector<Particle> initial_particles_; // 初始粒子状态（用于重置）
    
    // 核心模块
    NeighborSearcher neighbor_searcher_;
    std::unique_ptr<ConstitutiveModel> constitutive_model_;
    ContactDetection contact_detection_;
    TimeIntegrator time_integrator_;
    
    // 仿真参数
    double time_step_;                    // 时间步长（秒）
    TerminationConditions termination_conditions_; // 终止条件
    size_t save_interval_;                // 保存间隔（时间步数）
    size_t stats_interval_;               // 统计信息更新间隔（时间步数）
    
    // 仿真状态
    SimulationState state_;
    mutable std::mutex state_mutex_;
    
    // 仿真统计信息
    SimulationStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 线程控制
    std::thread simulation_thread_;
    std::mutex thread_mutex_;
    std::condition_variable cv_;
    bool terminate_thread_;
    bool pause_thread_;
    
    // 回调函数
    ProgressCallback progress_callback_;
    StateChangeCallback state_change_callback_;
    DataCallback data_callback_;
    
    // 输出目录
    std::string output_dir_;
    
    // 位移历史记录（用于计算位移变化）
    std::vector<Eigen::Vector3d> previous_positions_;
};

} // namespace particle_simulation
