// ContactDetection.h
// 接触检测与碰撞响应模块
// 实现粒子间接触检测、碰撞力计算和边界条件处理

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "Particle.h"
#include "NeighborSearcher.h"

namespace particle_simulation {

// 接触类型枚举
enum class ContactType {
    PARTICLE_PARTICLE,   // 粒子-粒子接触
    PARTICLE_BOUNDARY,   // 粒子-边界接触
    BOUNDARY_BOUNDARY    // 边界-边界接触（预留）
};

// 接触信息结构体
struct ContactInfo {
    size_t particle1_id;     // 第一个粒子ID
    size_t particle2_id;     // 第二个粒子ID（如果是边界接触，可能为特殊值）
    ContactType type;        // 接触类型
    Eigen::Vector3d contact_point;  // 接触点
    Eigen::Vector3d normal;  // 接触法线方向
    double penetration_depth; // 穿透深度
    double contact_force;    // 接触力大小
    Eigen::Vector3d relative_velocity; // 相对速度
};

// 碰撞响应模型类型枚举
enum class CollisionResponseModel {
    SPRING_DAMPER,   // 弹簧-阻尼模型
    HERTZ,           // Hertz接触模型
    DEM              // 离散元接触模型
};

// 接触检测与碰撞响应类
class ContactDetection {
public:
    // 构造函数
    ContactDetection(CollisionResponseModel model = CollisionResponseModel::SPRING_DAMPER);
    
    // 初始化接触检测
    // 参数：
    // - particles: 粒子列表
    // - neighbor_searcher: 邻域搜索器
    void initialize(
        const std::vector<Particle>& particles,
        const NeighborSearcher& neighbor_searcher
    );
    
    // 检测所有接触
    // 参数：
    // - particles: 粒子列表
    // - neighbor_searcher: 邻域搜索器
    // 返回：检测到的接触列表
    std::vector<ContactInfo> detectContacts(
        const std::vector<Particle>& particles,
        const NeighborSearcher& neighbor_searcher
    );
    
    // 计算碰撞力
    // 参数：
    // - particles: 粒子列表
    // - contacts: 接触列表
    // 返回：每个粒子的碰撞力
    std::vector<Eigen::Vector3d> computeCollisionForces(
        const std::vector<Particle>& particles,
        const std::vector<ContactInfo>& contacts
    );
    
    // 应用碰撞力到粒子
    // 参数：
    // - particles: 粒子列表
    // - collision_forces: 碰撞力列表
    void applyCollisionForces(
        std::vector<Particle>& particles,
        const std::vector<Eigen::Vector3d>& collision_forces
    );
    
    // 设置碰撞响应模型参数
    // 参数：
    // - params: 模型参数向量
    void setCollisionParameters(const Eigen::VectorXd& params);
    
    // 获取碰撞响应模型参数
    Eigen::VectorXd getCollisionParameters() const;
    
    // 设置碰撞响应模型
    void setCollisionModel(CollisionResponseModel model);
    
    // 获取碰撞响应模型
    CollisionResponseModel getCollisionModel() const;
    
    // 设置边界条件
    // 参数：
    // - boundary_type: 边界类型
    // - boundary_params: 边界参数
    void setBoundaryCondition(const std::string& boundary_type, const Eigen::VectorXd& boundary_params);
    
    // 检查粒子穿透情况
    // 参数：
    // - particles: 粒子列表
    // 返回：穿透的粒子对数
    size_t checkPenetration(const std::vector<Particle>& particles) const;
    
    // 优化数值稳定性
    // 参数：
    // - particles: 粒子列表
    // - dt: 时间步长
    void optimizeNumericalStability(std::vector<Particle>& particles, double dt);
    
private:
    // 弹簧-阻尼模型参数
    struct SpringDamperParams {
        double spring_stiffness;  // 弹簧刚度
        double damping_coefficient; // 阻尼系数
        double restitution_coefficient; // 恢复系数
        double friction_coefficient; // 摩擦系数
    };
    
    // Hertz接触模型参数
    struct HertzParams {
        double young_modulus;     // 弹性模量
        double poisson_ratio;     // 泊松比
        double restitution_coefficient; // 恢复系数
        double friction_coefficient; // 摩擦系数
    };
    
    // 检测单个粒子的接触
    // 参数：
    // - particle: 目标粒子
    // - particles: 粒子列表
    // - neighbor_ids: 邻近粒子ID列表
    // - contact_list: 接触列表（输出）
    void detectParticleContacts(
        const Particle& particle,
        size_t particle_id,
        const std::vector<Particle>& particles,
        const std::vector<size_t>& neighbor_ids,
        std::vector<ContactInfo>& contact_list
    );
    
    // 检测粒子-边界接触
    // 参数：
    // - particle: 目标粒子
    // - particle_id: 目标粒子ID
    // - contact_list: 接触列表（输出）
    void detectBoundaryContacts(
        const Particle& particle,
        size_t particle_id,
        std::vector<ContactInfo>& contact_list
    );
    
    // 使用弹簧-阻尼模型计算碰撞力
    // 参数：
    // - contact: 接触信息
    // - particles: 粒子列表
    // 返回：碰撞力向量
    Eigen::Vector3d computeSpringDamperForce(
        const ContactInfo& contact,
        const std::vector<Particle>& particles
    );
    
    // 使用Hertz模型计算碰撞力
    // 参数：
    // - contact: 接触信息
    // - particles: 粒子列表
    // 返回：碰撞力向量
    Eigen::Vector3d computeHertzForce(
        const ContactInfo& contact,
        const std::vector<Particle>& particles
    );
    
    // 使用DEM模型计算碰撞力
    // 参数：
    // - contact: 接触信息
    // - particles: 粒子列表
    // 返回：碰撞力向量
    Eigen::Vector3d computeDEMForce(
        const ContactInfo& contact,
        const std::vector<Particle>& particles
    );
    
    // 应用边界条件
    // 参数：
    // - particle: 粒子
    // 返回：边界力
    Eigen::Vector3d applyBoundaryCondition(const Particle& particle);
    
    // 计算粒子间距离和法线
    // 参数：
    // - p1: 第一个粒子
    // - p2: 第二个粒子
    // - distance: 距离（输出）
    // - normal: 法线方向（输出）
    void computeDistanceAndNormal(
        const Particle& p1,
        const Particle& p2,
        double& distance,
        Eigen::Vector3d& normal
    ) const;
    
    CollisionResponseModel collision_model_; // 碰撞响应模型
    
    // 模型参数
    SpringDamperParams spring_damper_params_;
    HertzParams hertz_params_;
    
    // 边界条件参数
    std::string boundary_type_;
    Eigen::VectorXd boundary_params_;
    
    // 数值稳定性参数
    double max_velocity_;      // 最大允许速度
    double penetration_tolerance_; // 穿透容差
    
    // 邻域搜索器引用
    const NeighborSearcher* neighbor_searcher_;
};

} // namespace particle_simulation
