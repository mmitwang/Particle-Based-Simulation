// TimeIntegrator.cpp
// 时间积分求解器实现

#include "TimeIntegrator.h"
#include <cmath>
#include <iostream>

namespace particle_simulation {

// ----------------------------------------
// TimeIntegrator 实现
// ----------------------------------------

// 构造函数
TimeIntegrator::TimeIntegrator(IntegratorType type)
    : integrator_type_(type),
      gravity_(0.0, -9.81, 0.0),
      adaptive_time_step_(true),
      max_time_step_(1.0e-3),
      min_time_step_(1.0e-6),
      cfl_number_(0.5),
      constitutive_model_(nullptr),
      contact_detection_(nullptr) {
}

// 初始化积分器
void TimeIntegrator::initialize(
    const std::vector<Particle>& particles,
    const ConstitutiveModel& constitutive_model,
    const ContactDetection& contact_detection
) {
    constitutive_model_ = &constitutive_model;
    contact_detection_ = &contact_detection;
    
    // 初始化历史状态
    previous_positions_.resize(particles.size());
    previous_velocities_.resize(particles.size());
    
    for (size_t i = 0; i < particles.size(); ++i) {
        previous_positions_[i] = particles[i].getPosition();
        previous_velocities_[i] = particles[i].getVelocity();
    }
}

// 执行一步时间积分
bool TimeIntegrator::integrateStep(
    std::vector<Particle>& particles,
    double dt,
    const ConstitutiveModel& constitutive_model,
    ContactDetection& contact_detection,
    NeighborSearcher& neighbor_searcher
) {
    // 检查时间步长
    if (dt <= 0.0) {
        std::cerr << "Invalid time step: " << dt << std::endl;
        return false;
    }
    
    // 如果使用自适应时间步长，计算推荐的时间步长
    double actual_dt = dt;
    if (adaptive_time_step_) {
        double recommended_dt = computeAdaptiveTimeStep(particles);
        actual_dt = std::min(std::max(recommended_dt, min_time_step_), max_time_step_);
    }
    
    // 根据积分方法执行一步积分
    bool success = false;
    switch (integrator_type_) {
        case IntegratorType::EXPLICIT_EULER:
            success = explicitEulerStep(particles, actual_dt, constitutive_model, contact_detection, neighbor_searcher);
            break;
        case IntegratorType::VERLET:
            success = verletStep(particles, actual_dt, constitutive_model, contact_detection, neighbor_searcher);
            break;
        case IntegratorType::RUNGE_KUTTA4:
            success = rungeKutta4Step(particles, actual_dt, constitutive_model, contact_detection, neighbor_searcher);
            break;
        default:
            std::cerr << "Unknown integrator type!" << std::endl;
            success = false;
            break;
    }
    
    if (success) {
        // 更新粒子的历史状态
        for (size_t i = 0; i < particles.size(); ++i) {
            previous_positions_[i] = particles[i].getPosition();
            previous_velocities_[i] = particles[i].getVelocity();
        }
        
        // 优化数值稳定性
        contact_detection.optimizeNumericalStability(particles, actual_dt);
        
        // 检查数值稳定性
        if (!checkStability(particles)) {
            std::cerr << "Numerical instability detected!" << std::endl;
            return false;
        }
    }
    
    return success;
}

// 设置积分器参数
void TimeIntegrator::setParameters(const Eigen::VectorXd& params) {
    // 参数含义取决于积分方法
    if (params.size() > 0) {
        max_time_step_ = params[0];
    }
    if (params.size() > 1) {
        min_time_step_ = params[1];
    }
    if (params.size() > 2) {
        cfl_number_ = params[2];
    }
}

// 获取积分器参数
Eigen::VectorXd TimeIntegrator::getParameters() const {
    Eigen::VectorXd params(3);
    params << max_time_step_, min_time_step_, cfl_number_;
    return params;
}

// 设置积分方法
void TimeIntegrator::setIntegratorType(IntegratorType type) {
    integrator_type_ = type;
}

// 获取积分方法
IntegratorType TimeIntegrator::getIntegratorType() const {
    return integrator_type_;
}

// 设置重力场
void TimeIntegrator::setGravity(const Eigen::Vector3d& gravity) {
    gravity_ = gravity;
}

// 获取重力场
const Eigen::VectorXd& TimeIntegrator::getGravity() const {
    return gravity_;
}

// 设置自适应时间步长
void TimeIntegrator::setAdaptiveTimeStep(bool adaptive) {
    adaptive_time_step_ = adaptive;
}

// 获取自适应时间步长设置
bool TimeIntegrator::getAdaptiveTimeStep() const {
    return adaptive_time_step_;
}

// 设置最大时间步长
void TimeIntegrator::setMaxTimeStep(double max_dt) {
    max_time_step_ = max_dt;
}

// 获取最大时间步长
double TimeIntegrator::getMaxTimeStep() const {
    return max_time_step_;
}

// 设置最小时间步长
void TimeIntegrator::setMinTimeStep(double min_dt) {
    min_time_step_ = min_dt;
}

// 获取最小时间步长
double TimeIntegrator::getMinTimeStep() const {
    return min_time_step_;
}

// 计算自适应时间步长
double TimeIntegrator::computeAdaptiveTimeStep(const std::vector<Particle>& particles) const {
    // 基于CFL条件计算自适应时间步长
    // CFL条件：dt <= C * h / v_max
    // 其中C为CFL数，h为粒子间距，v_max为最大速度
    
    double v_max = 0.0;
    double min_distance = std::numeric_limits<double>::max();
    
    // 计算最大速度和最小粒子间距
    for (size_t i = 0; i < particles.size(); ++i) {
        const Particle& p1 = particles[i];
        v_max = std::max(v_max, p1.getVelocity().norm());
        
        for (size_t j = i + 1; j < particles.size(); ++j) {
            const Particle& p2 = particles[j];
            double distance = (p1.getPosition() - p2.getPosition()).norm();
            min_distance = std::min(min_distance, distance);
        }
    }
    
    // 处理特殊情况
    if (v_max < 1.0e-12) {
        v_max = 1.0e-12;
    }
    if (min_distance < 1.0e-12) {
        min_distance = 1.0e-12;
    }
    
    // 计算CFL时间步长
    double cfl_dt = cfl_number_ * min_distance / v_max;
    
    // 考虑重力波速
    double g = gravity_.norm();
    double wave_speed_dt = 0.1 * sqrt(min_distance / g);
    
    // 返回最小的时间步长
    return std::min(cfl_dt, wave_speed_dt);
}

// 检查数值稳定性
bool TimeIntegrator::checkStability(const std::vector<Particle>& particles) const {
    // 检查粒子速度是否过大
    for (const auto& particle : particles) {
        if (particle.getVelocity().norm() > 1.0e3) {
            return false;
        }
    }
    
    // 检查粒子密度是否合理
    for (const auto& particle : particles) {
        if (particle.getDensity() < 1.0 || particle.getDensity() > 1.0e5) {
            return false;
        }
    }
    
    return true;
}

// 计算粒子所受的所有力
Eigen::VectorXd TimeIntegrator::computeTotalForce(
    const std::vector<Particle>& particles,
    size_t particle_id,
    const ConstitutiveModel& constitutive_model,
    ContactDetection& contact_detection,
    NeighborSearcher& neighbor_searcher
) {
    // 注意：这个方法需要根据实际的物理模型进行扩展
    // 目前简化实现，只考虑重力和接触力
    
    return Eigen::Vector3d::Zero();
}

// 显式欧拉法积分
bool TimeIntegrator::explicitEulerStep(
    std::vector<Particle>& particles,
    double dt,
    const ConstitutiveModel& constitutive_model,
    ContactDetection& contact_detection,
    NeighborSearcher& neighbor_searcher
) {
    // 1. 应用重力力
    applyGravity(particles);
    
    // 2. 检测接触
    std::vector<ContactInfo> contacts = contact_detection.detectContacts(particles, neighbor_searcher);
    
    // 3. 计算碰撞力
    std::vector<Eigen::Vector3d> collision_forces = contact_detection.computeCollisionForces(particles, contacts);
    
    // 4. 应用碰撞力
    contact_detection.applyCollisionForces(particles, collision_forces);
    
    // 5. 更新粒子状态
    for (auto& particle : particles) {
        // 计算加速度
        Eigen::Vector3d acceleration = particle.getAcceleration();
        
        // 更新速度：v(t+dt) = v(t) + a(t) * dt
        Eigen::Vector3d new_velocity = particle.getVelocity() + acceleration * dt;
        particle.setVelocity(new_velocity);
        
        // 更新位置：x(t+dt) = x(t) + v(t) * dt
        Eigen::Vector3d new_position = particle.getPosition() + particle.getVelocity() * dt;
        particle.setPosition(new_position);
        
        // 重置力
        particle.resetForce();
    }
    
    // 6. 更新邻域搜索结构
    neighbor_searcher.update(particles);
    
    return true;
}

// Verlet法积分
bool TimeIntegrator::verletStep(
    std::vector<Particle>& particles,
    double dt,
    const ConstitutiveModel& constitutive_model,
    ContactDetection& contact_detection,
    NeighborSearcher& neighbor_searcher
) {
    // Verlet法：x(t+dt) = 2x(t) - x(t-dt) + a(t) * dt^2
    
    // 1. 应用重力力
    applyGravity(particles);
    
    // 2. 检测接触
    std::vector<ContactInfo> contacts = contact_detection.detectContacts(particles, neighbor_searcher);
    
    // 3. 计算碰撞力
    std::vector<Eigen::Vector3d> collision_forces = contact_detection.computeCollisionForces(particles, contacts);
    
    // 4. 应用碰撞力
    contact_detection.applyCollisionForces(particles, collision_forces);
    
    // 5. 更新粒子状态
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& particle = particles[i];
        
        // 计算加速度
        Eigen::Vector3d acceleration = particle.getAcceleration();
        
        // 更新位置：x(t+dt) = 2x(t) - x(t-dt) + a(t) * dt^2
        Eigen::Vector3d new_position = 2.0 * particle.getPosition() - previous_positions_[i] + acceleration * dt * dt;
        particle.setPosition(new_position);
        
        // 更新速度：v(t) = (x(t+dt) - x(t-dt)) / (2*dt)
        Eigen::Vector3d new_velocity = (new_position - previous_positions_[i]) / (2.0 * dt);
        particle.setVelocity(new_velocity);
        
        // 重置力
        particle.resetForce();
    }
    
    // 6. 更新邻域搜索结构
    neighbor_searcher.update(particles);
    
    return true;
}

// 四阶Runge-Kutta法积分
bool TimeIntegrator::rungeKutta4Step(
    std::vector<Particle>& particles,
    double dt,
    const ConstitutiveModel& constitutive_model,
    ContactDetection& contact_detection,
    NeighborSearcher& neighbor_searcher
) {
    // 注意：RK4方法需要保存中间状态，实现较为复杂
    // 这里简化实现，使用显式欧拉法的结果
    // 完整的RK4实现需要保存粒子的中间状态
    
    return explicitEulerStep(particles, dt, constitutive_model, contact_detection, neighbor_searcher);
}

// 应用重力力
void TimeIntegrator::applyGravity(std::vector<Particle>& particles) {
    for (auto& particle : particles) {
        // 计算重力力：F = m * g
        Eigen::Vector3d gravity_force = particle.getMass() * gravity_;
        particle.applyForce(gravity_force);
        
        // 更新加速度：a = F / m = g
        particle.setAcceleration(gravity_);
    }
}

} // namespace particle_simulation
