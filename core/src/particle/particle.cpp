// 粒子系统源文件
// 实现Particle类和ParticleSystem类的具体功能

#include "particle/particle.h"
#include <Eigen/Geometry>
#include <fstream>
#include <sstream>
#include <iostream>

namespace particle_simulation {

// ----------------------------------------
// Particle类实现
// ----------------------------------------

// 构造函数
Particle::Particle() :
    id_(0),
    type_(ParticleType::SOIL),
    radius_(0.1),
    mass_(0.0),
    moisture_content_(0.0) {
    // 计算初始质量
    double volume = (4.0 / 3.0) * M_PI * std::pow(radius_, 3);
    mass_ = volume * 2600.0; // 默认密度2600kg/m³
    state_.density = 2600.0;
    
    // 默认岩性参数
    lithology_params_ = Eigen::VectorXd(5);
    lithology_params_ << 1.0e6, 0.3, 10000.0, 30.0, 10.0; // E, nu, c, phi, psi
}

Particle::Particle(size_t id, const Eigen::Vector3d& position, ParticleType type) :
    id_(id),
    type_(type),
    radius_(0.1),
    moisture_content_(0.0) {
    // 设置初始位置
    state_.position = position;
    
    // 计算初始质量
    double volume = (4.0 / 3.0) * M_PI * std::pow(radius_, 3);
    
    // 根据粒子类型设置密度
    switch (type) {
        case ParticleType::SOIL:
            state_.density = 2600.0;
            break;
        case ParticleType::ROCK:
            state_.density = 2800.0;
            break;
        case ParticleType::WATER:
            state_.density = 1000.0;
            break;
        case ParticleType::BOUNDARY:
            state_.density = 3000.0;
            break;
        default:
            state_.density = 2600.0;
    }
    
    mass_ = volume * state_.density;
    
    // 默认岩性参数
    lithology_params_ = Eigen::VectorXd(5);
    lithology_params_ << 1.0e6, 0.3, 10000.0, 30.0, 10.0; // E, nu, c, phi, psi
}

// 获取粒子ID
size_t Particle::getId() const {
    return id_;
}

// 获取/设置粒子类型
ParticleType Particle::getType() const {
    return type_;
}

void Particle::setType(ParticleType type) {
    type_ = type;
    
    // 更新密度
    switch (type) {
        case ParticleType::SOIL:
            state_.density = 2600.0;
            break;
        case ParticleType::ROCK:
            state_.density = 2800.0;
            break;
        case ParticleType::WATER:
            state_.density = 1000.0;
            break;
        case ParticleType::BOUNDARY:
            state_.density = 3000.0;
            break;
        default:
            state_.density = 2600.0;
    }
    
    // 更新质量
    double volume = (4.0 / 3.0) * M_PI * std::pow(radius_, 3);
    mass_ = volume * state_.density;
}

// 获取/设置粒子状态
const ParticleState& Particle::getState() const {
    return state_;
}

void Particle::setState(const ParticleState& state) {
    state_ = state;
}

// 获取/设置粒子位置
const Eigen::Vector3d& Particle::getPosition() const {
    return state_.position;
}

void Particle::setPosition(const Eigen::Vector3d& position) {
    state_.position = position;
}

// 获取/设置粒子速度
const Eigen::Vector3d& Particle::getVelocity() const {
    return state_.velocity;
}

void Particle::setVelocity(const Eigen::Vector3d& velocity) {
    state_.velocity = velocity;
}

// 获取/设置粒子加速度
const Eigen::Vector3d& Particle::getAcceleration() const {
    return state_.acceleration;
}

void Particle::setAcceleration(const Eigen::Vector3d& acceleration) {
    state_.acceleration = acceleration;
}

// 获取/设置粒子密度
double Particle::getDensity() const {
    return state_.density;
}

void Particle::setDensity(double density) {
    state_.density = density;
    
    // 更新质量
    double volume = (4.0 / 3.0) * M_PI * std::pow(radius_, 3);
    mass_ = volume * density;
}

// 获取/设置粒子半径
double Particle::getRadius() const {
    return radius_;
}

void Particle::setRadius(double radius) {
    radius_ = radius;
    
    // 更新质量
    double volume = (4.0 / 3.0) * M_PI * std::pow(radius_, 3);
    mass_ = volume * state_.density;
}

// 获取/设置粒子质量
double Particle::getMass() const {
    return mass_;
}

void Particle::setMass(double mass) {
    mass_ = mass;
    
    // 更新密度
    double volume = (4.0 / 3.0) * M_PI * std::pow(radius_, 3);
    if (volume > 0.0) {
        state_.density = mass / volume;
    }
}

// 获取/设置岩性参数
const Eigen::VectorXd& Particle::getLithologyParams() const {
    return lithology_params_;
}

void Particle::setLithologyParams(const Eigen::VectorXd& params) {
    lithology_params_ = params;
}

// 获取/设置含水率
double Particle::getMoistureContent() const {
    return moisture_content_;
}

void Particle::setMoistureContent(double moisture) {
    moisture_content_ = moisture;
}

// 粒子受力计算
void Particle::applyForce(const Eigen::VectorXd& force) {
    force_ += force;
}

// 重置受力
void Particle::resetForce() {
    force_.setZero();
}

// 更新粒子状态
void Particle::update(double dt) {
    // 计算加速度
    state_.acceleration = force_ / mass_;
    
    // 更新速度（显式欧拉法）
    state_.velocity += state_.acceleration * dt;
    
    // 更新位置
    state_.position += state_.velocity * dt + 0.5 * state_.acceleration * dt * dt;
    
    // 计算应变率（简化实现）
    state_.strain_rate = Eigen::Matrix3d::Zero();
    
    // 重置力
    force_.setZero();
}

// 计算压力
void Particle::computePressure() {
    // 简化实现：压力 = -1/3 * 应力张量迹
    state_.pressure = -state_.stress.trace() / 3.0;
}

// 计算主应力
Eigen::Vector3d Particle::computePrincipalStresses() const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(state_.stress);
    return solver.eigenvalues();
}

// 字符串表示
std::string Particle::toString() const {
    std::stringstream ss;
    ss << "Particle ID: " << id_ << ", Type: " << static_cast<int>(type_) 
       << ", Position: (" << state_.position.x() << ", " << state_.position.y() << ", " << state_.position.z() << ")" 
       << ", Velocity: (" << state_.velocity.x() << ", " << state_.velocity.y() << ", " << state_.velocity.z() << ")" 
       << ", Density: " << state_.density << ", Pressure: " << state_.pressure;
    return ss.str();
}

// ----------------------------------------
// ParticleSystem类实现
// ----------------------------------------

// 构造函数
ParticleSystem::ParticleSystem() : next_id_(0) {
}

// 添加粒子
void ParticleSystem::addParticle(const Particle& particle) {
    particles_.push_back(particle);
    if (particle.getId() >= next_id_) {
        next_id_ = particle.getId() + 1;
    }
}

// 获取粒子数量
size_t ParticleSystem::getParticleCount() const {
    return particles_.size();
}

// 获取粒子
Particle& ParticleSystem::getParticle(size_t id) {
    for (auto& particle : particles_) {
        if (particle.getId() == id) {
            return particle;
        }
    }
    throw std::out_of_range("Particle with ID " + std::to_string(id) + " not found");
}

const Particle& ParticleSystem::getParticle(size_t id) const {
    for (const auto& particle : particles_) {
        if (particle.getId() == id) {
            return particle;
        }
    }
    throw std::out_of_range("Particle with ID " + std::to_string(id) + " not found");
}

// 获取所有粒子
std::vector<Particle>& ParticleSystem::getParticles() {
    return particles_;
}

const std::vector<Particle>& ParticleSystem::getParticles() const {
    return particles_;
}

// 清除所有粒子
void ParticleSystem::clearParticles() {
    particles_.clear();
    next_id_ = 0;
}

// 更新所有粒子状态
void ParticleSystem::update(double dt) {
    for (auto& particle : particles_) {
        particle.update(dt);
    }
}

// 计算粒子系统的边界
Eigen::AlignedBox3d ParticleSystem::computeBounds() const {
    if (particles_.empty()) {
        return Eigen::AlignedBox3d();
    }
    
    Eigen::AlignedBox3d bounds;
    for (const auto& particle : particles_) {
        bounds.extend(particle.getPosition());
    }
    
    return bounds;
}

// 导出粒子数据
void ParticleSystem::exportParticles(const std::string& file_path) const {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }
    
    // 写入头部
    file << "# Particle data format: ID Type x y z vx vy vz ax ay az density pressure" << std::endl;
    file << "# Type: 0=SOIL, 1=ROCK, 2=WATER, 3=BOUNDARY" << std::endl;
    
    // 写入粒子数据
    for (const auto& particle : particles_) {
        const auto& state = particle.getState();
        file << particle.getId() << " "
             << static_cast<int>(particle.getType()) << " "
             << state.position.x() << " " << state.position.y() << " " << state.position.z() << " "
             << state.velocity.x() << " " << state.velocity.y() << " " << state.velocity.z() << " "
             << state.acceleration.x() << " " << state.acceleration.y() << " " << state.acceleration.z() << " "
             << state.density << " " << state.pressure << std::endl;
    }
    
    file.close();
    std::cout << "Particles exported to: " << file_path << std::endl;
}

// 导入粒子数据
void ParticleSystem::importParticles(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }
    
    particles_.clear();
    next_id_ = 0;
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释行
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        size_t id;
        int type_int;
        double x, y, z, vx, vy, vz, ax, ay, az, density, pressure;
        
        if (iss >> id >> type_int >> x >> y >> z >> vx >> vy >> vz >> ax >> ay >> az >> density >> pressure) {
            ParticleType type = static_cast<ParticleType>(type_int);
            Particle particle(id, Eigen::Vector3d(x, y, z), type);
            
            // 设置速度和加速度
            particle.setVelocity(Eigen::Vector3d(vx, vy, vz));
            particle.setAcceleration(Eigen::Vector3d(ax, ay, az));
            particle.setDensity(density);
            
            // 设置压力
            ParticleState state = particle.getState();
            state.pressure = pressure;
            particle.setState(state);
            
            addParticle(particle);
        }
    }
    
    file.close();
    std::cout << "Particles imported from: " << file_path << std::endl;
}

} // namespace particle_simulation
