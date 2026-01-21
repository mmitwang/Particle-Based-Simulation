// SimulationController.cpp
// 仿真过程控制模块实现

#include "SimulationController.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <filesystem>

namespace particle_simulation {

// 构造函数
SimulationController::SimulationController()
    : state_(SimulationState::IDLE),
      time_step_(0.001),
      terminate_thread_(false),
      pause_thread_(false),
      output_dir_("./output") {
    // 创建输出目录
    std::filesystem::create_directories(output_dir_);
}

// 析构函数
SimulationController::~SimulationController() {
    // 确保仿真线程已停止
    stop();
}

// 初始化仿真
bool SimulationController::initialize(
    const std::vector<Particle>& particles,
    const NeighborSearcher& neighbor_searcher,
    const ConstitutiveModel& constitutive_model,
    const ContactDetection& contact_detection,
    const TimeIntegrator& time_integrator
) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_ != SimulationState::IDLE && state_ != SimulationState::FINISHED) {
        std::cerr << "Cannot initialize simulation in current state: " << static_cast<int>(state_) << std::endl;
        return false;
    }
    
    // 设置初始状态
    state_ = SimulationState::INITIALIZING;
    
    // 复制粒子列表
    particles_ = particles;
    initial_particles_ = particles;
    
    // 初始化历史位置
    previous_positions_.resize(particles_.size());
    for (size_t i = 0; i < particles_.size(); ++i) {
        previous_positions_[i] = particles_[i].getPosition();
    }
    
    // 复制核心模块
    neighbor_searcher_ = neighbor_searcher;
    constitutive_model_ = constitutive_model::ConstitutiveModelFactory::createModel(constitutive_model.getType());
    if (!constitutive_model_) {
        std::cerr << "Failed to create constitutive model!" << std::endl;
        state_ = SimulationState::ERROR;
        return false;
    }
    contact_detection_ = contact_detection;
    time_integrator_ = time_integrator;
    
    // 初始化统计信息
    stats_ = SimulationStats();
    stats_.total_particles = particles_.size();
    
    // 初始化邻域搜索器
    double search_radius = neighbor_searcher.getSearchRadius();
    neighbor_searcher_.initialize(particles_, search_radius);
    
    // 初始化接触检测模块
    contact_detection_.initialize(particles_, neighbor_searcher_);
    
    // 初始化时间积分器
    time_integrator_.initialize(particles_, *constitutive_model_, contact_detection_);
    
    // 完成初始化
    state_ = SimulationState::IDLE;
    
    // 触发状态变化回调
    if (state_change_callback_) {
        state_change_callback_(state_);
    }
    
    std::cout << "Simulation initialized successfully." << std::endl;
    return true;
}

// 设置仿真参数
void SimulationController::setSimulationParameters(
    double time_step,
    const TerminationConditions& termination_conditions,
    size_t save_interval,
    size_t stats_interval
) {
    time_step_ = time_step;
    termination_conditions_ = termination_conditions;
    save_interval_ = save_interval;
    stats_interval_ = stats_interval;
}

// 启动仿真
bool SimulationController::start() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_ != SimulationState::IDLE && state_ != SimulationState::PAUSED) {
        std::cerr << "Cannot start simulation in current state: " << static_cast<int>(state_) << std::endl;
        return false;
    }
    
    // 设置状态为运行中
    state_ = SimulationState::RUNNING;
    
    // 重置线程控制标志
    terminate_thread_ = false;
    pause_thread_ = false;
    
    // 触发状态变化回调
    if (state_change_callback_) {
        state_change_callback_(state_);
    }
    
    // 启动仿真线程
    simulation_thread_ = std::thread(&SimulationController::simulationThread, this);
    
    std::cout << "Simulation started." << std::endl;
    return true;
}

// 暂停仿真
bool SimulationController::pause() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_ != SimulationState::RUNNING) {
        std::cerr << "Cannot pause simulation in current state: " << static_cast<int>(state_) << std::endl;
        return false;
    }
    
    // 设置状态为已暂停
    state_ = SimulationState::PAUSED;
    pause_thread_ = true;
    
    // 触发状态变化回调
    if (state_change_callback_) {
        state_change_callback_(state_);
    }
    
    std::cout << "Simulation paused." << std::endl;
    return true;
}

// 继续仿真
bool SimulationController::resume() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_ != SimulationState::PAUSED) {
        std::cerr << "Cannot resume simulation in current state: " << static_cast<int>(state_) << std::endl;
        return false;
    }
    
    // 设置状态为运行中
    state_ = SimulationState::RUNNING;
    pause_thread_ = false;
    
    // 通知条件变量，唤醒线程
    cv_.notify_one();
    
    // 触发状态变化回调
    if (state_change_callback_) {
        state_change_callback_(state_);
    }
    
    std::cout << "Simulation resumed." << std::endl;
    return true;
}

// 停止仿真
bool SimulationController::stop() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_ != SimulationState::RUNNING && state_ != SimulationState::PAUSED) {
        return false;
    }
    
    // 设置终止标志
    terminate_thread_ = true;
    pause_thread_ = false;
    
    // 通知条件变量，唤醒线程
    cv_.notify_one();
    
    // 等待仿真线程结束
    if (simulation_thread_.joinable()) {
        simulation_thread_.join();
    }
    
    // 设置状态为已完成
    state_ = SimulationState::FINISHED;
    
    // 触发状态变化回调
    if (state_change_callback_) {
        state_change_callback_(state_);
    }
    
    std::cout << "Simulation stopped." << std::endl;
    return true;
}

// 保存仿真状态
bool SimulationController::saveSimulationState(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving: " << file_path << std::endl;
        return false;
    }
    
    // 保存仿真状态
    file.write(reinterpret_cast<const char*>(&state_), sizeof(SimulationState));
    
    // 保存统计信息
    file.write(reinterpret_cast<const char*>(&stats_), sizeof(SimulationStats));
    
    // 保存时间步长
    file.write(reinterpret_cast<const char*>(&time_step_), sizeof(double));
    
    // 保存终止条件
    file.write(reinterpret_cast<const char*>(&termination_conditions_), sizeof(TerminationConditions));
    
    // 保存粒子数量
    size_t particle_count = particles_.size();
    file.write(reinterpret_cast<const char*>(&particle_count), sizeof(size_t));
    
    // 保存粒子数据
    for (const auto& particle : particles_) {
        // 保存粒子位置
        Eigen::Vector3d pos = particle.getPosition();
        file.write(reinterpret_cast<const char*>(pos.data()), sizeof(double) * 3);
        
        // 保存粒子速度
        Eigen::Vector3d vel = particle.getVelocity();
        file.write(reinterpret_cast<const char*>(vel.data()), sizeof(double) * 3);
        
        // 保存粒子加速度
        Eigen::Vector3d acc = particle.getAcceleration();
        file.write(reinterpret_cast<const char*>(acc.data()), sizeof(double) * 3);
        
        // 保存粒子质量
        double mass = particle.getMass();
        file.write(reinterpret_cast<const char*>(&mass), sizeof(double));
        
        // 保存粒子密度
        double density = particle.getDensity();
        file.write(reinterpret_cast<const char*>(&density), sizeof(double));
        
        // 保存粒子半径
        double radius = particle.getRadius();
        file.write(reinterpret_cast<const char*>(&radius), sizeof(double));
        
        // 保存粒子类型
        ParticleType type = particle.getType();
        file.write(reinterpret_cast<const char*>(&type), sizeof(ParticleType));
        
        // 保存岩性参数
        Eigen::VectorXd lith_params = particle.getLithologyParams();
        size_t params_size = lith_params.size();
        file.write(reinterpret_cast<const char*>(&params_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(lith_params.data()), sizeof(double) * params_size);
        
        // 保存含水率
        double moisture = particle.getMoistureContent();
        file.write(reinterpret_cast<const char*>(&moisture), sizeof(double));
    }
    
    file.close();
    std::cout << "Simulation state saved to: " << file_path << std::endl;
    return true;
}

// 加载仿真状态
bool SimulationController::loadSimulationState(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for loading: " << file_path << std::endl;
        return false;
    }
    
    // 加载仿真状态
    file.read(reinterpret_cast<char*>(&state_), sizeof(SimulationState));
    
    // 加载统计信息
    file.read(reinterpret_cast<char*>(&stats_), sizeof(SimulationStats));
    
    // 加载时间步长
    file.read(reinterpret_cast<char*>(&time_step_), sizeof(double));
    
    // 加载终止条件
    file.read(reinterpret_cast<char*>(&termination_conditions_), sizeof(TerminationConditions));
    
    // 加载粒子数量
    size_t particle_count;
    file.read(reinterpret_cast<char*>(&particle_count), sizeof(size_t));
    
    // 加载粒子数据
    particles_.resize(particle_count);
    previous_positions_.resize(particle_count);
    
    for (size_t i = 0; i < particle_count; ++i) {
        Particle& particle = particles_[i];
        
        // 加载粒子位置
        Eigen::Vector3d pos;
        file.read(reinterpret_cast<char*>(pos.data()), sizeof(double) * 3);
        particle.setPosition(pos);
        previous_positions_[i] = pos;
        
        // 加载粒子速度
        Eigen::Vector3d vel;
        file.read(reinterpret_cast<char*>(vel.data()), sizeof(double) * 3);
        particle.setVelocity(vel);
        
        // 加载粒子加速度
        Eigen::Vector3d acc;
        file.read(reinterpret_cast<char*>(acc.data()), sizeof(double) * 3);
        particle.setAcceleration(acc);
        
        // 加载粒子质量
        double mass;
        file.read(reinterpret_cast<char*>(&mass), sizeof(double));
        particle.setMass(mass);
        
        // 加载粒子密度
        double density;
        file.read(reinterpret_cast<char*>(&density), sizeof(double));
        particle.setDensity(density);
        
        // 加载粒子半径
        double radius;
        file.read(reinterpret_cast<char*>(&radius), sizeof(double));
        particle.setRadius(radius);
        
        // 加载粒子类型
        ParticleType type;
        file.read(reinterpret_cast<char*>(&type), sizeof(ParticleType));
        particle.setType(type);
        
        // 加载岩性参数
        size_t params_size;
        file.read(reinterpret_cast<char*>(&params_size), sizeof(size_t));
        Eigen::VectorXd lith_params(params_size);
        file.read(reinterpret_cast<char*>(lith_params.data()), sizeof(double) * params_size);
        particle.setLithologyParams(lith_params);
        
        // 加载含水率
        double moisture;
        file.read(reinterpret_cast<char*>(&moisture), sizeof(double));
        particle.setMoistureContent(moisture);
    }
    
    file.close();
    
    // 重新初始化核心模块
    neighbor_searcher_.initialize(particles_, neighbor_searcher_.getSearchRadius());
    contact_detection_.initialize(particles_, neighbor_searcher_);
    time_integrator_.initialize(particles_, *constitutive_model_, contact_detection_);
    
    std::cout << "Simulation state loaded from: " << file_path << std::endl;
    return true;
}

// 获取当前仿真状态
SimulationState SimulationController::getSimulationState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

// 获取当前仿真统计信息
SimulationStats SimulationController::getSimulationStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// 获取当前粒子列表
std::vector<Particle> SimulationController::getParticles() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return particles_;
}

// 设置进度回调函数
void SimulationController::setProgressCallback(ProgressCallback callback) {
    progress_callback_ = callback;
}

// 设置状态变化回调函数
void SimulationController::setStateChangeCallback(StateChangeCallback callback) {
    state_change_callback_ = callback;
}

// 设置数据回调函数
void SimulationController::setDataCallback(DataCallback callback) {
    data_callback_ = callback;
}

// 设置输出目录
void SimulationController::setOutputDirectory(const std::string& output_dir) {
    output_dir_ = output_dir;
    std::filesystem::create_directories(output_dir_);
}

// 仿真线程函数
void SimulationController::simulationThread() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (true) {
        // 检查是否需要终止
        if (terminate_thread_) {
            break;
        }
        
        // 检查是否需要暂停
        if (pause_thread_) {
            std::unique_lock<std::mutex> lock(thread_mutex_);
            cv_.wait(lock, [this] { return !pause_thread_ || terminate_thread_; });
            continue;
        }
        
        // 执行一步时间积分
        auto step_start = std::chrono::high_resolution_clock::now();
        
        bool success = time_integrator_.integrateStep(
            particles_,
            time_step_,
            *constitutive_model_,
            contact_detection_,
            neighbor_searcher_
        );
        
        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double>(step_end - step_start).count();
        
        if (!success) {
            std::cerr << "Failed to execute integration step!" << std::endl;
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_ = SimulationState::ERROR;
            if (state_change_callback_) {
                state_change_callback_(state_);
            }
            break;
        }
        
        // 更新统计信息
        stats_.current_time_step++;
        stats_.current_simulation_time += time_step_;
        stats_.computational_time = std::chrono::duration<double>(step_end - start_time).count();
        
        // 定期更新详细统计信息
        if (stats_.current_time_step % stats_interval_ == 0) {
            updateSimulationStats();
            
            // 触发进度回调
            if (progress_callback_) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                progress_callback_(stats_);
            }
        }
        
        // 定期保存仿真结果
        if (stats_.current_time_step % save_interval_ == 0) {
            saveSimulationResults(stats_.current_time_step);
        }
        
        // 触发数据回调
        if (data_callback_) {
            data_callback_(particles_, stats_.current_simulation_time);
        }
        
        // 检查终止条件
        if (checkTerminationConditions()) {
            break;
        }
    }
    
    // 设置最终状态
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = SimulationState::FINISHED;
    
    // 触发状态变化回调
    if (state_change_callback_) {
        state_change_callback_(state_);
    }
    
    // 保存最终结果
    saveSimulationResults(stats_.current_time_step);
    
    std::cout << "Simulation finished. Total time steps: " << stats_.current_time_step << ", Total simulation time: " << stats_.current_simulation_time << "s" << std::endl;
}

// 更新仿真统计信息
void SimulationController::updateSimulationStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // 计算位移统计
    double total_displacement = 0.0;
    double max_displacement = 0.0;
    
    for (size_t i = 0; i < particles_.size(); ++i) {
        const Particle& particle = particles_[i];
        Eigen::Vector3d displacement = particle.getPosition() - previous_positions_[i];
        double disp_magnitude = displacement.norm();
        
        total_displacement += disp_magnitude;
        max_displacement = std::max(max_displacement, disp_magnitude);
    }
    
    stats_.average_displacement = total_displacement / particles_.size();
    stats_.max_displacement = max_displacement;
    
    // 计算速度统计
    double total_velocity = 0.0;
    double max_velocity = 0.0;
    
    for (const auto& particle : particles_) {
        double vel_magnitude = particle.getVelocity().norm();
        total_velocity += vel_magnitude;
        max_velocity = std::max(max_velocity, vel_magnitude);
    }
    
    stats_.average_velocity = total_velocity / particles_.size();
    stats_.max_velocity = max_velocity;
    
    // 检查粒子穿透
    stats_.penetration_count = contact_detection_.checkPenetration(particles_);
}

// 检查终止条件
bool SimulationController::checkTerminationConditions() {
    // 检查最大时间步数
    if (stats_.current_time_step >= termination_conditions_.max_time_steps) {
        std::cout << "Termination condition met: Maximum time steps reached." << std::endl;
        return true;
    }
    
    // 检查最大仿真时间
    if (stats_.current_simulation_time >= termination_conditions_.max_simulation_time) {
        std::cout << "Termination condition met: Maximum simulation time reached." << std::endl;
        return true;
    }
    
    // 检查最大位移
    if (stats_.max_displacement > termination_conditions_.max_displacement) {
        std::cout << "Termination condition met: Maximum displacement exceeded." << std::endl;
        return true;
    }
    
    // 检查收敛条件
    if (stats_.current_time_step > 100) { // 至少运行100步
        if (stats_.average_displacement < termination_conditions_.convergence_threshold) {
            std::cout << "Termination condition met: Convergence reached." << std::endl;
            return true;
        }
    }
    
    return false;
}

// 保存仿真结果
void SimulationController::saveSimulationResults(size_t time_step) {
    // 创建时间步目录
    std::stringstream dir_ss;
    dir_ss << output_dir_ << "/step_" << std::setw(6) << std::setfill('0') << time_step;
    std::string step_dir = dir_ss.str();
    std::filesystem::create_directories(step_dir);
    
    // 保存粒子位置数据
    std::stringstream pos_ss;
    pos_ss << step_dir << "/positions.txt";
    std::ofstream pos_file(pos_ss.str());
    
    if (pos_file.is_open()) {
        pos_file << "# Time step: " << time_step << ", Simulation time: " << stats_.current_simulation_time << "s\n";
        pos_file << "# x y z\n";
        
        for (const auto& particle : particles_) {
            const Eigen::Vector3d& pos = particle.getPosition();
            pos_file << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
        }
        
        pos_file.close();
    }
    
    // 保存粒子速度数据
    std::stringstream vel_ss;
    vel_ss << step_dir << "/velocities.txt";
    std::ofstream vel_file(vel_ss.str());
    
    if (vel_file.is_open()) {
        vel_file << "# Time step: " << time_step << ", Simulation time: " << stats_.current_simulation_time << "s\n";
        vel_file << "# vx vy vz\n";
        
        for (const auto& particle : particles_) {
            const Eigen::Vector3d& vel = particle.getVelocity();
            vel_file << vel.x() << " " << vel.y() << " " << vel.z() << "\n";
        }
        
        vel_file.close();
    }
    
    // 保存统计信息
    std::stringstream stats_ss;
    stats_ss << step_dir << "/stats.txt";
    std::ofstream stats_file(stats_ss.str());
    
    if (stats_file.is_open()) {
        stats_file << "Time step: " << stats_.current_time_step << "\n";
        stats_file << "Simulation time: " << stats_.current_simulation_time << "s\n";
        stats_file << "Total particles: " << stats_.total_particles << "\n";
        stats_file << "Average displacement: " << stats_.average_displacement << "m\n";
        stats_file << "Max displacement: " << stats_.max_displacement << "m\n";
        stats_file << "Average velocity: " << stats_.average_velocity << "m/s\n";
        stats_file << "Max velocity: " << stats_.max_velocity << "m/s\n";
        stats_file << "Computational time: " << stats_.computational_time << "s\n";
        stats_file << "Penetration count: " << stats_.penetration_count << "\n";
        
        stats_file.close();
    }
    
    std::cout << "Simulation results saved for time step " << time_step << std::endl;
}

} // namespace particle_simulation
