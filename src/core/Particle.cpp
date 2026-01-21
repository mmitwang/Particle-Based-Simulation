// Particle.cpp
// 粒子类实现

#include "Particle.h"

namespace particle_simulation {

// 构造函数
Particle::Particle() 
    : position_(Eigen::Vector3d::Zero()),
      velocity_(Eigen::Vector3d::Zero()),
      acceleration_(Eigen::Vector3d::Zero()),
      force_(Eigen::Vector3d::Zero()),
      mass_(1.0),
      density_(1000.0),
      radius_(0.1),
      type_(ParticleType::SOIL),
      stress_(Eigen::Matrix3d::Zero()),
      strain_(Eigen::Matrix3d::Zero()),
      lithology_params_(Eigen::VectorXd::Zero(5)), // 默认5个岩性参数
      moisture_content_(0.0) {
}

Particle::Particle(const Eigen::Vector3d& position, double mass, ParticleType type)
    : position_(position),
      velocity_(Eigen::Vector3d::Zero()),
      acceleration_(Eigen::Vector3d::Zero()),
      force_(Eigen::Vector3d::Zero()),
      mass_(mass),
      density_(1000.0),
      radius_(0.1),
      type_(type),
      stress_(Eigen::Matrix3d::Zero()),
      strain_(Eigen::Matrix3d::Zero()),
      lithology_params_(Eigen::VectorXd::Zero(5)),
      moisture_content_(0.0) {
}

// 获取粒子位置
const Eigen::Vector3d& Particle::getPosition() const {
    return position_;
}

// 设置粒子位置
void Particle::setPosition(const Eigen::Vector3d& position) {
    position_ = position;
}

// 获取粒子速度
const Eigen::Vector3d& Particle::getVelocity() const {
    return velocity_;
}

// 设置粒子速度
void Particle::setVelocity(const Eigen::Vector3d& velocity) {
    velocity_ = velocity;
}

// 获取粒子加速度
const Eigen::Vector3d& Particle::getAcceleration() const {
    return acceleration_;
}

// 设置粒子加速度
void Particle::setAcceleration(const Eigen::Vector3d& acceleration) {
    acceleration_ = acceleration;
}

// 获取粒子质量
double Particle::getMass() const {
    return mass_;
}

// 设置粒子质量
void Particle::setMass(double mass) {
    mass_ = mass;
}

// 获取粒子密度
double Particle::getDensity() const {
    return density_;
}

// 设置粒子密度
void Particle::setDensity(double density) {
    density_ = density;
}

// 获取粒子类型
ParticleType Particle::getType() const {
    return type_;
}

// 设置粒子类型
void Particle::setType(ParticleType type) {
    type_ = type;
}

// 获取粒子半径
double Particle::getRadius() const {
    return radius_;
}

// 设置粒子半径
void Particle::setRadius(double radius) {
    radius_ = radius;
}

// 获取应力张量
const Eigen::Matrix3d& Particle::getStress() const {
    return stress_;
}

// 设置应力张量
void Particle::setStress(const Eigen::Matrix3d& stress) {
    stress_ = stress;
}

// 获取应变张量
const Eigen::Matrix3d& Particle::getStrain() const {
    return strain_;
}

// 设置应变张量
void Particle::setStrain(const Eigen::Matrix3d& strain) {
    strain_ = strain;
}

// 获取岩性参数
const Eigen::VectorXd& Particle::getLithologyParams() const {
    return lithology_params_;
}

// 设置岩性参数
void Particle::setLithologyParams(const Eigen::VectorXd& params) {
    lithology_params_ = params;
}

// 获取含水率
double Particle::getMoistureContent() const {
    return moisture_content_;
}

// 设置含水率
void Particle::setMoistureContent(double moisture) {
    moisture_content_ = moisture;
}

// 粒子受力计算
void Particle::applyForce(const Eigen::Vector3d& force) {
    force_ += force;
}

// 重置受力
void Particle::resetForce() {
    force_.setZero();
}

// 更新粒子状态
void Particle::update(double dt) {
    // 计算加速度
    acceleration_ = force_ / mass_;
    
    // 更新速度（显式欧拉法）
    velocity_ += acceleration_ * dt;
    
    // 更新位置
    position_ += velocity_ * dt + 0.5 * acceleration_ * dt * dt;
}

} // namespace particle_simulation
