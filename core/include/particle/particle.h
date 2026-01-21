// 粒子系统头文件
// 定义粒子类和相关数据结构

#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace particle_simulation {

// 粒子类型枚举
enum class ParticleType {
    SOIL,       // 土体
    ROCK,       // 岩体
    WATER,      // 水体
    BOUNDARY    // 边界粒子
};

// 粒子状态结构体
struct ParticleState {
    Eigen::Vector3d position;      // 位置 (m)
    Eigen::Vector3d velocity;      // 速度 (m/s)
    Eigen::Vector3d acceleration;  // 加速度 (m/s²)
    Eigen::Matrix3d stress;        // 应力张量 (Pa)
    Eigen::Matrix3d strain;        // 应变张量
    Eigen::Matrix3d strain_rate;   // 应变率张量 (1/s)
    double density;                // 密度 (kg/m³)
    double pressure;               // 压力 (Pa)
    
    // 构造函数
    ParticleState() :
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        acceleration(Eigen::Vector3d::Zero()),
        stress(Eigen::Matrix3d::Zero()),
        strain(Eigen::Matrix3d::Zero()),
        strain_rate(Eigen::Matrix3d::Zero()),
        density(0.0),
        pressure(0.0) {
    }
};

// 粒子类
class Particle {
public:
    // 构造函数
    Particle();
    Particle(size_t id, const Eigen::Vector3d& position, ParticleType type = ParticleType::SOIL);
    
    // 获取粒子ID
    size_t getId() const;
    
    // 获取/设置粒子类型
    ParticleType getType() const;
    void setType(ParticleType type);
    
    // 获取/设置粒子状态
    const ParticleState& getState() const;
    void setState(const ParticleState& state);
    
    // 获取/设置粒子位置
    const Eigen::Vector3d& getPosition() const;
    void setPosition(const Eigen::Vector3d& position);
    
    // 获取/设置粒子速度
    const Eigen::Vector3d& getVelocity() const;
    void setVelocity(const Eigen::Vector3d& velocity);
    
    // 获取/设置粒子加速度
    const Eigen::Vector3d& getAcceleration() const;
    void setAcceleration(const Eigen::Vector3d& acceleration);
    
    // 获取/设置粒子密度
    double getDensity() const;
    void setDensity(double density);
    
    // 获取/设置粒子半径
    double getRadius() const;
    void setRadius(double radius);
    
    // 获取/设置粒子质量
    double getMass() const;
    void setMass(double mass);
    
    // 获取/设置岩性参数
    const Eigen::VectorXd& getLithologyParams() const;
    void setLithologyParams(const Eigen::VectorXd& params);
    
    // 获取/设置含水率
    double getMoistureContent() const;
    void setMoistureContent(double moisture);
    
    // 粒子受力计算
    void applyForce(const Eigen::Vector3d& force);
    
    // 重置受力
    void resetForce();
    
    // 更新粒子状态
    void update(double dt);
    
    // 计算压力
    void computePressure();
    
    // 计算主应力
    Eigen::Vector3d computePrincipalStresses() const;
    
    // 字符串表示
    std::string toString() const;
    
private:
    size_t id_;                     // 粒子唯一ID
    ParticleType type_;             // 粒子类型
    ParticleState state_;           // 粒子状态
    Eigen::Vector3d force_;         // 粒子所受合力 (N)
    double radius_;                 // 粒子半径 (m)
    double mass_;                   // 粒子质量 (kg)
    Eigen::VectorXd lithology_params_; // 岩性参数 (弹性模量、泊松比、黏聚力、内摩擦角等)
    double moisture_content_;       // 含水率 (%)
};

// 粒子系统类，管理所有粒子
class ParticleSystem {
public:
    // 构造函数
    ParticleSystem();
    
    // 添加粒子
    void addParticle(const Particle& particle);
    
    // 获取粒子数量
    size_t getParticleCount() const;
    
    // 获取粒子
    Particle& getParticle(size_t id);
    const Particle& getParticle(size_t id) const;
    
    // 获取所有粒子
    std::vector<Particle>& getParticles();
    const std::vector<Particle>& getParticles() const;
    
    // 清除所有粒子
    void clearParticles();
    
    // 更新所有粒子状态
    void update(double dt);
    
    // 计算粒子系统的边界
    Eigen::AlignedBox3d computeBounds() const;
    
    // 导出粒子数据
    void exportParticles(const std::string& file_path) const;
    
    // 导入粒子数据
    void importParticles(const std::string& file_path);
    
private:
    std::vector<Particle> particles_;  // 粒子列表
    size_t next_id_;                   // 下一个粒子ID
};

} // namespace particle_simulation
