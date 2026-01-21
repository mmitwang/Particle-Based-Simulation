// ContactDetection.cpp
// 接触检测与碰撞响应模块实现

#include "ContactDetection.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace particle_simulation {

// ----------------------------------------
// ContactDetection 实现
// ----------------------------------------

// 构造函数
ContactDetection::ContactDetection(CollisionResponseModel model)
    : collision_model_(model),
      neighbor_searcher_(nullptr),
      max_velocity_(10.0),
      penetration_tolerance_(1e-6) {
    
    // 初始化弹簧-阻尼模型参数
    spring_damper_params_.spring_stiffness = 1.0e5;
    spring_damper_params_.damping_coefficient = 100.0;
    spring_damper_params_.restitution_coefficient = 0.8;
    spring_damper_params_.friction_coefficient = 0.5;
    
    // 初始化Hertz模型参数
    hertz_params_.young_modulus = 1.0e6;
    hertz_params_.poisson_ratio = 0.3;
    hertz_params_.restitution_coefficient = 0.8;
    hertz_params_.friction_coefficient = 0.5;
    
    // 默认边界条件
    boundary_type_ = "rigid";
    boundary_params_ = Eigen::VectorXd::Zero(6); // x_min, x_max, y_min, y_max, z_min, z_max
}

// 初始化接触检测
void ContactDetection::initialize(
    const std::vector<Particle>& particles,
    const NeighborSearcher& neighbor_searcher
) {
    neighbor_searcher_ = &neighbor_searcher;
    
    // 计算默认边界条件
    if (boundary_type_ == "rigid" && boundary_params_.isZero()) {
        Eigen::Vector3d min_bound = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d max_bound = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
        
        for (const auto& particle : particles) {
            const Eigen::Vector3d& pos = particle.getPosition();
            min_bound = min_bound.cwiseMin(pos);
            max_bound = max_bound.cwiseMax(pos);
        }
        
        // 扩展边界
        double boundary_margin = 0.5;
        boundary_params_ = Eigen::VectorXd(6);
        boundary_params_ << 
            min_bound.x() - boundary_margin,
            max_bound.x() + boundary_margin,
            min_bound.y() - boundary_margin,
            max_bound.y() + boundary_margin,
            min_bound.z() - boundary_margin,
            max_bound.z() + boundary_margin;
    }
}

// 检测所有接触
std::vector<ContactInfo> ContactDetection::detectContacts(
    const std::vector<Particle>& particles,
    const NeighborSearcher& neighbor_searcher
) {
    std::vector<ContactInfo> contacts;
    
    // 对每个粒子检测接触
    for (size_t i = 0; i < particles.size(); ++i) {
        const Particle& particle = particles[i];
        
        // 跳过边界粒子（只检测非边界粒子的接触）
        if (particle.getType() == ParticleType::BOUNDARY) {
            continue;
        }
        
        // 查找邻近粒子
        std::vector<size_t> neighbors = neighbor_searcher.findNeighbors(particle);
        
        // 检测粒子-粒子接触
        detectParticleContacts(particle, i, particles, neighbors, contacts);
        
        // 检测粒子-边界接触
        detectBoundaryContacts(particle, i, contacts);
    }
    
    return contacts;
}

// 计算碰撞力
std::vector<Eigen::Vector3d> ContactDetection::computeCollisionForces(
    const std::vector<Particle>& particles,
    const std::vector<ContactInfo>& contacts
) {
    // 初始化碰撞力向量
    std::vector<Eigen::Vector3d> collision_forces(particles.size(), Eigen::Vector3d::Zero());
    
    // 计算每个接触的碰撞力
    for (const auto& contact : contacts) {
        Eigen::Vector3d force;
        
        // 根据碰撞响应模型计算碰撞力
        switch (collision_model_) {
            case CollisionResponseModel::SPRING_DAMPER:
                force = computeSpringDamperForce(contact, particles);
                break;
            case CollisionResponseModel::HERTZ:
                force = computeHertzForce(contact, particles);
                break;
            case CollisionResponseModel::DEM:
                force = computeDEMForce(contact, particles);
                break;
            default:
                force = Eigen::Vector3d::Zero();
                break;
        }
        
        // 应用碰撞力到粒子
        if (contact.type == ContactType::PARTICLE_PARTICLE) {
            // 粒子-粒子接触：力大小相等，方向相反
            collision_forces[contact.particle1_id] += force;
            collision_forces[contact.particle2_id] -= force;
        } else if (contact.type == ContactType::PARTICLE_BOUNDARY) {
            // 粒子-边界接触：只有粒子受到力
            collision_forces[contact.particle1_id] += force;
        }
    }
    
    // 应用边界条件力
    for (size_t i = 0; i < particles.size(); ++i) {
        const Particle& particle = particles[i];
        if (particle.getType() != ParticleType::BOUNDARY) {
            Eigen::Vector3d boundary_force = applyBoundaryCondition(particle);
            collision_forces[i] += boundary_force;
        }
    }
    
    return collision_forces;
}

// 应用碰撞力到粒子
void ContactDetection::applyCollisionForces(
    std::vector<Particle>& particles,
    const std::vector<Eigen::Vector3d>& collision_forces
) {
    // 确保力向量大小与粒子数量一致
    if (collision_forces.size() != particles.size()) {
        std::cerr << "Collision forces size mismatch with particles!" << std::endl;
        return;
    }
    
    // 应用力到每个粒子
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].applyForce(collision_forces[i]);
    }
}

// 设置碰撞响应模型参数
void ContactDetection::setCollisionParameters(const Eigen::VectorXd& params) {
    switch (collision_model_) {
        case CollisionResponseModel::SPRING_DAMPER:
            if (params.size() >= 4) {
                spring_damper_params_.spring_stiffness = params[0];
                spring_damper_params_.damping_coefficient = params[1];
                spring_damper_params_.restitution_coefficient = params[2];
                spring_damper_params_.friction_coefficient = params[3];
            }
            break;
        case CollisionResponseModel::HERTZ:
            if (params.size() >= 4) {
                hertz_params_.young_modulus = params[0];
                hertz_params_.poisson_ratio = params[1];
                hertz_params_.restitution_coefficient = params[2];
                hertz_params_.friction_coefficient = params[3];
            }
            break;
        default:
            break;
    }
}

// 获取碰撞响应模型参数
Eigen::VectorXd ContactDetection::getCollisionParameters() const {
    Eigen::VectorXd params;
    
    switch (collision_model_) {
        case CollisionResponseModel::SPRING_DAMPER:
            params = Eigen::VectorXd(4);
            params << 
                spring_damper_params_.spring_stiffness,
                spring_damper_params_.damping_coefficient,
                spring_damper_params_.restitution_coefficient,
                spring_damper_params_.friction_coefficient;
            break;
        case CollisionResponseModel::HERTZ:
            params = Eigen::VectorXd(4);
            params << 
                hertz_params_.young_modulus,
                hertz_params_.poisson_ratio,
                hertz_params_.restitution_coefficient,
                hertz_params_.friction_coefficient;
            break;
        default:
            params = Eigen::VectorXd::Zero(0);
            break;
    }
    
    return params;
}

// 设置碰撞响应模型
void ContactDetection::setCollisionModel(CollisionResponseModel model) {
    collision_model_ = model;
}

// 获取碰撞响应模型
CollisionResponseModel ContactDetection::getCollisionModel() const {
    return collision_model_;
}

// 设置边界条件
void ContactDetection::setBoundaryCondition(const std::string& boundary_type, const Eigen::VectorXd& boundary_params) {
    boundary_type_ = boundary_type;
    boundary_params_ = boundary_params;
}

// 检查粒子穿透情况
size_t ContactDetection::checkPenetration(const std::vector<Particle>& particles) const {
    size_t penetration_count = 0;
    
    // 简单实现：检查所有粒子对
    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {
            const Particle& p1 = particles[i];
            const Particle& p2 = particles[j];
            
            double distance;
            Eigen::Vector3d normal;
            computeDistanceAndNormal(p1, p2, distance, normal);
            
            double min_distance = p1.getRadius() + p2.getRadius();
            if (distance < min_distance - penetration_tolerance_) {
                penetration_count++;
            }
        }
    }
    
    return penetration_count;
}

// 优化数值稳定性
void ContactDetection::optimizeNumericalStability(std::vector<Particle>& particles, double dt) {
    for (auto& particle : particles) {
        // 限制最大速度
        Eigen::Vector3d velocity = particle.getVelocity();
        if (velocity.norm() > max_velocity_) {
            velocity.normalize();
            velocity *= max_velocity_;
            particle.setVelocity(velocity);
        }
        
        // 应用阻尼（数值阻尼）
        double damping_factor = 0.999;
        particle.setVelocity(particle.getVelocity() * damping_factor);
        
        // 检查并修复粒子穿透（简化实现）
        // 这里可以添加更复杂的穿透修复逻辑
    }
}

// 检测单个粒子的接触
void ContactDetection::detectParticleContacts(
    const Particle& particle,
    size_t particle_id,
    const std::vector<Particle>& particles,
    const std::vector<size_t>& neighbor_ids,
    std::vector<ContactInfo>& contact_list
) {
    for (size_t neighbor_id : neighbor_ids) {
        if (neighbor_id == particle_id) {
            continue; // 跳过自身
        }
        
        const Particle& neighbor = particles[neighbor_id];
        
        // 计算距离和法线
        double distance;
        Eigen::Vector3d normal;
        computeDistanceAndNormal(particle, neighbor, distance, normal);
        
        // 计算最小距离
        double min_distance = particle.getRadius() + neighbor.getRadius();
        
        // 检查是否发生接触
        if (distance < min_distance) {
            // 计算穿透深度
            double penetration_depth = min_distance - distance;
            
            // 计算接触点
            Eigen::Vector3d contact_point = particle.getPosition() + normal * (particle.getRadius() - penetration_depth / 2.0);
            
            // 计算相对速度
            Eigen::Vector3d relative_velocity = neighbor.getVelocity() - particle.getVelocity();
            
            // 创建接触信息
            ContactInfo contact;
            contact.particle1_id = particle_id;
            contact.particle2_id = neighbor_id;
            contact.type = ContactType::PARTICLE_PARTICLE;
            contact.contact_point = contact_point;
            contact.normal = normal;
            contact.penetration_depth = penetration_depth;
            contact.contact_force = 0.0; // 后续计算
            contact.relative_velocity = relative_velocity;
            
            contact_list.push_back(contact);
        }
    }
}

// 检测粒子-边界接触
void ContactDetection::detectBoundaryContacts(
    const Particle& particle,
    size_t particle_id,
    std::vector<ContactInfo>& contact_list
) {
    if (boundary_type_ != "rigid") {
        return; // 目前只支持刚性边界
    }
    
    const Eigen::Vector3d& pos = particle.getPosition();
    double radius = particle.getRadius();
    
    // 检查六个边界面
    for (int i = 0; i < 3; ++i) {
        // 负方向边界
        double boundary_min = boundary_params_[i * 2];
        double distance = pos[i] - (boundary_min + radius);
        
        if (distance < 0) {
            // 发生接触
            Eigen::Vector3d normal = Eigen::Vector3d::Zero();
            normal[i] = 1.0;
            
            Eigen::Vector3d contact_point = pos;
            contact_point[i] = boundary_min + radius;
            
            ContactInfo contact;
            contact.particle1_id = particle_id;
            contact.particle2_id = particles.size(); // 边界粒子ID用特殊值表示
            contact.type = ContactType::PARTICLE_BOUNDARY;
            contact.contact_point = contact_point;
            contact.normal = normal;
            contact.penetration_depth = -distance;
            contact.contact_force = 0.0;
            contact.relative_velocity = -particle.getVelocity();
            
            contact_list.push_back(contact);
        }
        
        // 正方向边界
        double boundary_max = boundary_params_[i * 2 + 1];
        distance = (boundary_max - radius) - pos[i];
        
        if (distance < 0) {
            // 发生接触
            Eigen::Vector3d normal = Eigen::Vector3d::Zero();
            normal[i] = -1.0;
            
            Eigen::Vector3d contact_point = pos;
            contact_point[i] = boundary_max - radius;
            
            ContactInfo contact;
            contact.particle1_id = particle_id;
            contact.particle2_id = particles.size();
            contact.type = ContactType::PARTICLE_BOUNDARY;
            contact.contact_point = contact_point;
            contact.normal = normal;
            contact.penetration_depth = -distance;
            contact.contact_force = 0.0;
            contact.relative_velocity = -particle.getVelocity();
            
            contact_list.push_back(contact);
        }
    }
}

// 使用弹簧-阻尼模型计算碰撞力
Eigen::Vector3d ContactDetection::computeSpringDamperForce(
    const ContactInfo& contact,
    const std::vector<Particle>& particles
) const {
    // 弹簧力：F_spring = k * delta * n
    // 阻尼力：F_damper = c * v_rel_n * n
    // 总力：F = (k * delta + c * v_rel_n) * n
    
    double k = spring_damper_params_.spring_stiffness;
    double c = spring_damper_params_.damping_coefficient;
    
    // 计算法向相对速度
    double v_rel_n = contact.relative_velocity.dot(contact.normal);
    
    // 计算弹簧-阻尼力
    double force_magnitude = k * contact.penetration_depth + c * v_rel_n;
    
    // 确保力为正值（只在压缩时有力）
    force_magnitude = std::max(0.0, force_magnitude);
    
    // 计算碰撞力向量
    Eigen::Vector3d force = force_magnitude * contact.normal;
    
    return force;
}

// 使用Hertz模型计算碰撞力
Eigen::Vector3d ContactDetection::computeHertzForce(
    const ContactInfo& contact,
    const std::vector<Particle>& particles
) const {
    // Hertz接触模型：F = k * delta^(3/2) + c * v_rel_n
    
    const Particle& p1 = particles[contact.particle1_id];
    const Particle& p2 = particles[contact.particle2_id];
    
    // 计算有效半径
    double R1 = p1.getRadius();
    double R2 = p2.getRadius();
    double effective_R = (R1 * R2) / (R1 + R2);
    
    // 计算有效弹性模量
    double E1 = hertz_params_.young_modulus;
    double E2 = hertz_params_.young_modulus;
    double nu1 = hertz_params_.poisson_ratio;
    double nu2 = hertz_params_.poisson_ratio;
    double effective_E = 1.0 / ((1.0 - nu1 * nu1) / E1 + (1.0 - nu2 * nu2) / E2);
    
    // 计算Hertz刚度系数
    double k_hertz = (4.0 / 3.0) * effective_E * sqrt(effective_R);
    
    // 计算阻尼系数（基于恢复系数）
    double e = hertz_params_.restitution_coefficient;
    double m1 = p1.getMass();
    double m2 = p2.getMass();
    double effective_m = (m1 * m2) / (m1 + m2);
    double c_hertz = 2.0 * sqrt(effective_m * k_hertz) * log(e) / sqrt(M_PI * M_PI + log(e) * log(e));
    
    // 计算法向相对速度
    double v_rel_n = contact.relative_velocity.dot(contact.normal);
    
    // 计算Hertz碰撞力
    double force_magnitude = k_hertz * pow(contact.penetration_depth, 1.5) + c_hertz * v_rel_n;
    
    // 确保力为正值
    force_magnitude = std::max(0.0, force_magnitude);
    
    // 计算碰撞力向量
    Eigen::Vector3d force = force_magnitude * contact.normal;
    
    return force;
}

// 使用DEM模型计算碰撞力
Eigen::Vector3d ContactDetection::computeDEMForce(
    const ContactInfo& contact,
    const std::vector<Particle>& particles
) const {
    // 简化DEM模型，使用弹簧-阻尼模型
    return computeSpringDamperForce(contact, particles);
}

// 应用边界条件
Eigen::VectorXd ContactDetection::applyBoundaryCondition(const Particle& particle) {
    // 目前只实现刚性边界条件
    if (boundary_type_ != "rigid") {
        return Eigen::Vector3d::Zero();
    }
    
    const Eigen::Vector3d& pos = particle.getPosition();
    double radius = particle.getRadius();
    Eigen::Vector3d boundary_force = Eigen::Vector3d::Zero();
    
    // 检查并应用刚性边界力
    for (int i = 0; i < 3; ++i) {
        // 负方向边界
        double boundary_min = boundary_params_[i * 2];
        if (pos[i] < boundary_min + radius) {
            double delta = boundary_min + radius - pos[i];
            boundary_force[i] += spring_damper_params_.spring_stiffness * delta;
        }
        
        // 正方向边界
        double boundary_max = boundary_params_[i * 2 + 1];
        if (pos[i] > boundary_max - radius) {
            double delta = pos[i] - (boundary_max - radius);
            boundary_force[i] -= spring_damper_params_.spring_stiffness * delta;
        }
    }
    
    return boundary_force;
}

// 计算粒子间距离和法线
void ContactDetection::computeDistanceAndNormal(
    const Particle& p1,
    const Particle& p2,
    double& distance,
    Eigen::VectorXd& normal
) const {
    Eigen::Vector3d delta = p2.getPosition() - p1.getPosition();
    distance = delta.norm();
    
    if (distance < 1e-12) {
        // 处理零距离情况
        normal = Eigen::Vector3d::UnitX();
    } else {
        normal = delta / distance;
    }
}

} // namespace particle_simulation
