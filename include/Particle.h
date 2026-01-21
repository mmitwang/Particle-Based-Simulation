// Particle.h
// 粒子类定义，包含粒子的基本属性和方法
// 支持不同岩性、含水率等参数的岩土体粒子

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace particle_simulation {

// 粒子类型枚举
enum class ParticleType {
    SOIL,       // 土体
    ROCK,       // 岩体
    WATER,      // 水体
    BOUNDARY    // 边界粒子
};

// 粒子类
class Particle {
public:
    // 构造函数
    Particle();
    Particle(const Eigen::Vector3d& position, double mass, ParticleType type);
    
    // 获取粒子位置
    const Eigen::Vector3d& getPosition() const;
    
    // 设置粒子位置
    void setPosition(const Eigen::Vector3d& position);
    
    // 获取粒子速度
    const Eigen::Vector3d& getVelocity() const;
    
    // 设置粒子速度
    void setVelocity(const Eigen::Vector3d& velocity);
    
    // 获取粒子加速度
    const Eigen::Vector3d& getAcceleration() const;
    
    // 设置粒子加速度
    void setAcceleration(const Eigen::Vector3d& acceleration);
    
    // 获取粒子质量
    double getMass() const;
    
    // 设置粒子质量
    void setMass(double mass);
    
    // 获取粒子密度
    double getDensity() const;
    
    // 设置粒子密度
    void setDensity(double density);
    
    // 获取粒子类型
    ParticleType getType() const;
    
    // 设置粒子类型
    void setType(ParticleType type);
    
    // 获取粒子半径
    double getRadius() const;
    
    // 设置粒子半径
    void setRadius(double radius);
    
    // 获取应力张量
    const Eigen::Matrix3d& getStress() const;
    
    // 设置应力张量
    void setStress(const Eigen::Matrix3d& stress);
    
    // 获取应变张量
    const Eigen::Matrix3d& getStrain() const;
    
    // 设置应变张量
    void setStrain(const Eigen::Matrix3d& strain);
    
    // 获取岩性参数
    const Eigen::VectorXd& getLithologyParams() const;
    
    // 设置岩性参数
    void setLithologyParams(const Eigen::VectorXd& params);
    
    // 获取含水率
    double getMoistureContent() const;
    
    // 设置含水率
    void setMoistureContent(double moisture);
    
    // 粒子受力计算
    void applyForce(const Eigen::Vector3d& force);
    
    // 重置受力
    void resetForce();
    
    // 更新粒子状态
    void update(double dt);
    
private:
    Eigen::Vector3d position_;        // 粒子位置
    Eigen::Vector3d velocity_;        // 粒子速度
    Eigen::Vector3d acceleration_;    // 粒子加速度
    Eigen::Vector3d force_;           // 粒子所受合力
    
    double mass_;                     // 粒子质量
    double density_;                  // 粒子密度
    double radius_;                   // 粒子半径
    
    ParticleType type_;               // 粒子类型
    
    Eigen::Matrix3d stress_;          // 应力张量
    Eigen::Matrix3d strain_;          // 应变张量
    
    Eigen::VectorXd lithology_params_; // 岩性参数（如弹性模量、泊松比、内摩擦角等）
    double moisture_content_;         // 含水率
};

} // namespace particle_simulation
