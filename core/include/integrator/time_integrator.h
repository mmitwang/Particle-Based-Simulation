// 时间积分求解器头文件
// 实现粒子状态的时间演化，包括显式欧拉法、Verlet法、Runge-Kutta法等

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "particle/particle.h"
#include "neighbor/neighbor_search.h"
#include "constitutive/constitutive_model.h"
#include "contact/contact_detection.h"

namespace particle_simulation {

// 时间积分方法类型枚举
enum class IntegratorType {
    EXPLICIT_EULER,     // 显式欧拉法
    VERLET,             // Verlet法
    RUNGE_KUTTA4,       // 四阶Runge-Kutta法
    IMPLICIT_EULER,     // 隐式欧拉法（预留）
    LEAPFROG            // Leapfrog法（预留）
};

// 时间积分器类
class TimeIntegrator {
public:
    // 构造函数
    TimeIntegrator(IntegratorType type = IntegratorType::EXPLICIT_EULER);
    
    // 初始化积分器
    // 参数：
    // - particles: 粒子列表
    // - constitutive_model: 本构模型
    // - contact_detection: 接触检测模块
    void initialize(
        const std::vector<Particle>& particles,
        const ConstitutiveModel& constitutive_model,
        const ContactDetection& contact_detection
    );
    
    // 执行一步时间积分
    // 参数：
    // - particles: 粒子列表（输入输出）
    // - dt: 时间步长
    // - constitutive_model: 本构模型
    // - contact_detection: 接触检测模块
    // - neighbor_searcher: 邻域搜索器
    // 返回：是否成功执行
    bool integrateStep(
        std::vector<Particle>& particles,
        double dt,
        const ConstitutiveModel& constitutive_model,
        ContactDetection& contact_detection,
        NeighborSearch& neighbor_searcher
    );
    
    // 设置积分器参数
    // 参数：
    // - params: 积分器参数向量
    void setParameters(const Eigen::VectorXd& params);
    
    // 获取积分器参数
    Eigen::VectorXd getParameters() const;
    
    // 设置积分方法
    void setIntegratorType(IntegratorType type);
    
    // 获取积分方法
    IntegratorType getIntegratorType() const;
    
    // 设置重力场
    void setGravity(const Eigen::Vector3d& gravity);
    
    // 获取重力场
    const Eigen::Vector3d& getGravity() const;
    
    // 设置自适应时间步长
    void setAdaptiveTimeStep(bool adaptive);
    
    // 获取自适应时间步长设置
    bool getAdaptiveTimeStep() const;
    
    // 设置最大时间步长
    void setMaxTimeStep(double max_dt);
    
    // 获取最大时间步长
    double getMaxTimeStep() const;
    
    // 设置最小时间步长
    void setMinTimeStep(double min_dt);
    
    // 获取最小时间步长
    double getMinTimeStep() const;
    
    // 计算自适应时间步长
    // 参数：
    // - particles: 粒子列表
    // 返回：推荐的时间步长
    double computeAdaptiveTimeStep(const std::vector<Particle>& particles) const;
    
    // 检查数值稳定性
    // 参数：
    // - particles: 粒子列表
    // 返回：是否稳定
    bool checkStability(const std::vector<Particle>& particles) const;
    
private:
    // 计算粒子所受的所有力
    // 参数：
    // - particles: 粒子列表
    // - particle_id: 目标粒子ID
    // - constitutive_model: 本构模型
    // - contact_detection: 接触检测模块
    // - neighbor_searcher: 邻域搜索器
    // 返回：总合力
    Eigen::Vector3d computeTotalForce(
        const std::vector<Particle>& particles,
        size_t particle_id,
        const ConstitutiveModel& constitutive_model,
        ContactDetection& contact_detection,
        NeighborSearch& neighbor_searcher
    );
    
    // 显式欧拉法积分
    bool explicitEulerStep(
        std::vector<Particle>& particles,
        double dt,
        const ConstitutiveModel& constitutive_model,
        ContactDetection& contact_detection,
        NeighborSearch& neighbor_searcher
    );
    
    // Verlet法积分
    bool verletStep(
        std::vector<Particle>& particles,
        double dt,
        const ConstitutiveModel& constitutive_model,
        ContactDetection& contact_detection,
        NeighborSearch& neighbor_searcher
    );
    
    // 四阶Runge-Kutta法积分
    bool rungeKutta4Step(
        std::vector<Particle>& particles,
        double dt,
        const ConstitutiveModel& constitutive_model,
        ContactDetection& contact_detection,
        NeighborSearch& neighbor_searcher
    );
    
    // 应用重力力
    void applyGravity(std::vector<Particle>& particles);
    
    IntegratorType integrator_type_;  // 积分方法类型
    
    Eigen::Vector3d gravity_;          // 重力加速度
    
    // 自适应时间步长参数
    bool adaptive_time_step_;          // 是否使用自适应时间步长
    double max_time_step_;             // 最大时间步长
    double min_time_step_;             // 最小时间步长
    double cfl_number_;                // CFL数（用于自适应时间步长）
    
    // 粒子状态历史（用于某些积分方法，如Verlet）
    std::vector<Eigen::Vector3d> previous_positions_;  // 前一时刻位置
    std::vector<Eigen::Vector3d> previous_velocities_; // 前一时刻速度
    
    // 本构模型和接触检测模块引用
    const ConstitutiveModel* constitutive_model_;
    const ContactDetection* contact_detection_;
};

} // namespace particle_simulation
