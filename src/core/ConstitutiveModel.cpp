// ConstitutiveModel.cpp
// 力学本构模型实现
// 实现弹性、弹塑性、黏弹性等本构模型

#include "ConstitutiveModel.h"
#include <cmath>
#include <iostream>

namespace particle_simulation {

// ----------------------------------------
// ConstitutiveModel 基类实现
// ----------------------------------------

// 构造函数
ConstitutiveModel::ConstitutiveModel(ConstitutiveModelType type)
    : type_(type) {
}

// 获取本构模型类型
ConstitutiveModelType ConstitutiveModel::getType() const {
    return type_;
}

// ----------------------------------------
// ElasticModel 实现
// ----------------------------------------

// 构造函数
ElasticModel::ElasticModel()
    : ConstitutiveModel(ConstitutiveModelType::ELASTIC),
      young_modulus_(1.0e6),
      poisson_ratio_(0.3) {
    updateElasticConstants();
}

ElasticModel::ElasticModel(double young_modulus, double poisson_ratio)
    : ConstitutiveModel(ConstitutiveModelType::ELASTIC),
      young_modulus_(young_modulus),
      poisson_ratio_(poisson_ratio) {
    updateElasticConstants();
}

// 计算应力更新
Eigen::Matrix3d ElasticModel::updateStress(
    const StressStrainState& state,
    double dt
) const {
    // 弹性本构关系：σ = σ_old + D : Δε
    // 其中D为弹性模量张量，Δε为应变增量
    
    Eigen::Matrix3d strain_increment = state.strain_rate * dt;
    Eigen::Matrix3d stress_increment = Eigen::Matrix3d::Zero();
    
    // 计算应力增量（简化版，适用于各向同性弹性材料）
    double lambda = (young_modulus_ * poisson_ratio_) / ((1.0 + poisson_ratio_) * (1.0 - 2.0 * poisson_ratio_));
    double mu = young_modulus_ / (2.0 * (1.0 + poisson_ratio_));
    
    // 体积应变增量
    double volumetric_strain_increment = strain_increment.trace();
    
    // 应力增量计算
    stress_increment = 2.0 * mu * strain_increment + lambda * volumetric_strain_increment * Eigen::Matrix3d::Identity();
    
    // 更新后的应力
    Eigen::Matrix3d new_stress = state.stress + stress_increment;
    
    return new_stress;
}

// 计算弹性模量张量
Eigen::MatrixXd ElasticModel::getElasticModulus() const {
    // 各向同性弹性材料的弹性模量张量（6x6矩阵）
    Eigen::MatrixXd D(6, 6);
    double lambda = (young_modulus_ * poisson_ratio_) / ((1.0 + poisson_ratio_) * (1.0 - 2.0 * poisson_ratio_));
    double mu = young_modulus_ / (2.0 * (1.0 + poisson_ratio_));
    
    D.setZero();
    D(0, 0) = D(1, 1) = D(2, 2) = lambda + 2.0 * mu;
    D(0, 1) = D(0, 2) = D(1, 0) = D(1, 2) = D(2, 0) = D(2, 1) = lambda;
    D(3, 3) = D(4, 4) = D(5, 5) = mu;
    
    return D;
}

// 设置模型参数
void ElasticModel::setParameters(const Eigen::VectorXd& params) {
    if (params.size() >= 2) {
        young_modulus_ = params[0];
        poisson_ratio_ = params[1];
        updateElasticConstants();
    }
}

// 获取模型参数
Eigen::VectorXd ElasticModel::getParameters() const {
    Eigen::VectorXd params(2);
    params[0] = young_modulus_;
    params[1] = poisson_ratio_;
    return params;
}

// 检查模型是否处于屈服状态
bool ElasticModel::isYielding(const StressStrainState& state) const {
    return false; // 弹性模型不会屈服
}

// 更新弹性常数
void ElasticModel::updateElasticConstants() {
    shear_modulus_ = young_modulus_ / (2.0 * (1.0 + poisson_ratio_));
    bulk_modulus_ = young_modulus_ / (3.0 * (1.0 - 2.0 * poisson_ratio_));
}

// ----------------------------------------
// MohrCoulombModel 实现
// ----------------------------------------

// 构造函数
MohrCoulombModel::MohrCoulombModel()
    : ConstitutiveModel(ConstitutiveModelType::MOHR_COULOMB),
      young_modulus_(1.0e6),
      poisson_ratio_(0.3),
      cohesion_(1000.0),
      friction_angle_(30.0 * M_PI / 180.0),
      dilation_angle_(10.0 * M_PI / 180.0) {
    updateElasticConstants();
}

MohrCoulombModel::MohrCoulombModel(
    double young_modulus,
    double poisson_ratio,
    double cohesion,
    double friction_angle,
    double dilation_angle
) : ConstitutiveModel(ConstitutiveModelType::MOHR_COULOMB),
      young_modulus_(young_modulus),
      poisson_ratio_(poisson_ratio),
      cohesion_(cohesion),
      friction_angle_(friction_angle),
      dilation_angle_(dilation_angle) {
    updateElasticConstants();
}

// 计算应力更新
Eigen::Matrix3d MohrCoulombModel::updateStress(
    const StressStrainState& state,
    double dt
) const {
    // 首先尝试弹性更新
    Eigen::Matrix3d strain_increment = state.strain_rate * dt;
    double lambda = (young_modulus_ * poisson_ratio_) / ((1.0 + poisson_ratio_) * (1.0 - 2.0 * poisson_ratio_));
    double mu = young_modulus_ / (2.0 * (1.0 + poisson_ratio_));
    
    // 体积应变增量
    double volumetric_strain_increment = strain_increment.trace();
    
    // 弹性应力增量
    Eigen::Matrix3d elastic_stress_increment = 2.0 * mu * strain_increment + lambda * volumetric_strain_increment * Eigen::Matrix3d::Identity();
    
    // 弹性预测应力
    Eigen::Matrix3d trial_stress = state.stress + elastic_stress_increment;
    
    // 构建弹性预测的应力应变状态
    StressStrainState trial_state;
    trial_state.stress = trial_stress;
    trial_state.strain = state.strain + strain_increment;
    trial_state.strain_rate = state.strain_rate;
    trial_state.density = state.density;
    trial_state.pressure = -trial_stress.trace() / 3.0;
    
    // 计算主应力
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(trial_stress);
    trial_state.principal_stresses = solver.eigenvalues();
    
    // 检查是否屈服
    if (!isYielding(trial_state)) {
        // 未屈服，采用弹性更新
        return trial_stress;
    } else {
        // 屈服，需要进行塑性修正
        // 简化的塑性流动规则（关联流动法则）
        double yield_function = computeYieldFunction(trial_state);
        
        // 计算塑性乘子
        double plastic_multiplier = yield_function / (2.0 * mu + 3.0 * bulk_modulus_ * (tan(friction_angle_) - tan(dilation_angle_)));
        
        // 计算塑性应变增量
        Eigen::Matrix3d plastic_strain_increment = plastic_multiplier * computePlasticFlowDirection(trial_state);
        
        // 最终应力更新：σ = σ_trial - D : ε_plastic
        Eigen::Matrix3d plastic_stress_increment = 2.0 * mu * plastic_strain_increment + lambda * plastic_strain_increment.trace() * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d final_stress = trial_stress - plastic_stress_increment;
        
        return final_stress;
    }
}

// 计算弹性模量张量
Eigen::MatrixXd MohrCoulombModel::getElasticModulus() const {
    // 与弹性模型相同
    ElasticModel elastic_model(young_modulus_, poisson_ratio_);
    return elastic_model.getElasticModulus();
}

// 设置模型参数
void MohrCoulombModel::setParameters(const Eigen::VectorXd& params) {
    if (params.size() >= 5) {
        young_modulus_ = params[0];
        poisson_ratio_ = params[1];
        cohesion_ = params[2];
        friction_angle_ = params[3];
        dilation_angle_ = params[4];
        updateElasticConstants();
    }
}

// 获取模型参数
Eigen::VectorXd MohrCoulombModel::getParameters() const {
    Eigen::VectorXd params(5);
    params[0] = young_modulus_;
    params[1] = poisson_ratio_;
    params[2] = cohesion_;
    params[3] = friction_angle_;
    params[4] = dilation_angle_;
    return params;
}

// 检查模型是否处于屈服状态
bool MohrCoulombModel::isYielding(const StressStrainState& state) const {
    double yield_function = computeYieldFunction(state);
    return yield_function > 1e-6; // 考虑数值误差
}

// 更新弹性常数
void MohrCoulombModel::updateElasticConstants() {
    shear_modulus_ = young_modulus_ / (2.0 * (1.0 + poisson_ratio_));
    bulk_modulus_ = young_modulus_ / (3.0 * (1.0 - 2.0 * poisson_ratio_));
}

// 计算屈服函数值
// Mohr-Coulomb屈服准则：τ = c + σ_n tanφ
// 其中τ为剪应力，c为黏聚力，σ_n为法向应力，φ为内摩擦角
// 三维形式：σ1 - σ3 = 2c cosφ + (σ1 + σ3) sinφ
// 其中σ1 ≥ σ2 ≥ σ3为主应力

// 计算屈服函数值
double MohrCoulombModel::computeYieldFunction(const StressStrainState& state) const {
    // 排序主应力
    Eigen::Vector3d sorted_stresses = state.principal_stresses;
    std::sort(sorted_stresses.data(), sorted_stresses.data() + 3, std::greater<double>());
    
    double sigma1 = sorted_stresses[0];
    double sigma3 = sorted_stresses[2];
    
    // Mohr-Coulomb屈服函数
    double yield = sigma1 - sigma3 - 2.0 * cohesion_ * cos(friction_angle_) - (sigma1 + sigma3) * sin(friction_angle_);
    
    return yield;
}

// 计算塑性流动方向
Eigen::Matrix3d MohrCoulombModel::computePlasticFlowDirection(const StressStrainState& state) const {
    // 关联流动法则，塑性流动方向与屈服函数梯度相同
    // 简化实现：返回单位张量乘以流动方向
    return Eigen::Matrix3d::Identity();
}

// 计算硬化参数
double MohrCoulombModel::computeHardeningParameter(const StressStrainState& state) const {
    // 简化实现：返回0（理想弹塑性）
    return 0.0;
}

// ----------------------------------------
// ViscoElasticModel 实现
// ----------------------------------------

// 构造函数
ViscoElasticModel::ViscoElasticModel()
    : ConstitutiveModel(ConstitutiveModelType::VISCO_ELASTIC),
      young_modulus_(1.0e6),
      poisson_ratio_(0.3),
      viscosity_(1.0e5),
      relaxation_time_(1.0) {
    updateElasticConstants();
}

ViscoElasticModel::ViscoElasticModel(
    double young_modulus,
    double poisson_ratio,
    double viscosity,
    double relaxation_time
) : ConstitutiveModel(ConstitutiveModelType::VISCO_ELASTIC),
      young_modulus_(young_modulus),
      poisson_ratio_(poisson_ratio),
      viscosity_(viscosity),
      relaxation_time_(relaxation_time) {
    updateElasticConstants();
}

// 计算应力更新
Eigen::Matrix3d ViscoElasticModel::updateStress(
    const StressStrainState& state,
    double dt
) const {
    // 黏弹性本构关系（Maxwell模型）
    // σ_dot + σ / τ = E : ε_dot + (E / η) : σ
    // 其中τ为松弛时间，η为黏度
    
    double lambda = (young_modulus_ * poisson_ratio_) / ((1.0 + poisson_ratio_) * (1.0 - 2.0 * poisson_ratio_));
    double mu = young_modulus_ / (2.0 * (1.0 + poisson_ratio_));
    
    // 体积应变率
    double volumetric_strain_rate = state.strain_rate.trace();
    
    // 应力率计算
    Eigen::Matrix3d stress_rate = Eigen::Matrix3d::Zero();
    
    // 弹性部分
    stress_rate += 2.0 * mu * state.strain_rate + lambda * volumetric_strain_rate * Eigen::Matrix3d::Identity();
    
    // 黏性部分
    stress_rate -= state.stress / relaxation_time_;
    
    // 使用欧拉法更新应力
    Eigen::Matrix3d new_stress = state.stress + stress_rate * dt;
    
    return new_stress;
}

// 计算弹性模量张量
Eigen::MatrixXd ViscoElasticModel::getElasticModulus() const {
    // 与弹性模型相同
    ElasticModel elastic_model(young_modulus_, poisson_ratio_);
    return elastic_model.getElasticModulus();
}

// 设置模型参数
void ViscoElasticModel::setParameters(const Eigen::VectorXd& params) {
    if (params.size() >= 4) {
        young_modulus_ = params[0];
        poisson_ratio_ = params[1];
        viscosity_ = params[2];
        relaxation_time_ = params[3];
        updateElasticConstants();
    }
}

// 获取模型参数
Eigen::VectorXd ViscoElasticModel::getParameters() const {
    Eigen::VectorXd params(4);
    params[0] = young_modulus_;
    params[1] = poisson_ratio_;
    params[2] = viscosity_;
    params[3] = relaxation_time_;
    return params;
}

// 检查模型是否处于屈服状态
bool ViscoElasticModel::isYielding(const StressStrainState& state) const {
    return false; // 黏弹性模型不会屈服
}

// 更新弹性常数
void ViscoElasticModel::updateElasticConstants() {
    shear_modulus_ = young_modulus_ / (2.0 * (1.0 + poisson_ratio_));
    bulk_modulus_ = young_modulus_ / (3.0 * (1.0 - 2.0 * poisson_ratio_));
}

// ----------------------------------------
// ConstitutiveModelFactory 实现
// ----------------------------------------

// 创建本构模型实例
std::unique_ptr<ConstitutiveModel> ConstitutiveModelFactory::createModel(
    ConstitutiveModelType type,
    const Eigen::VectorXd& params
) {
    std::unique_ptr<ConstitutiveModel> model;
    
    switch (type) {
        case ConstitutiveModelType::ELASTIC: {
            model = std::make_unique<ElasticModel>();
            break;
        }
        case ConstitutiveModelType::MOHR_COULOMB: {
            model = std::make_unique<MohrCoulombModel>();
            break;
        }
        case ConstitutiveModelType::VISCO_ELASTIC: {
            model = std::make_unique<ViscoElasticModel>();
            break;
        }
        default: {
            std::cerr << "Unknown constitutive model type!" << std::endl;
            return nullptr;
        }
    }
    
    // 设置模型参数
    if (!params.empty()) {
        model->setParameters(params);
    }
    
    return model;
}

} // namespace particle_simulation
