// ConstitutiveModel.h
// 力学本构模型模块
// 实现岩土体的弹塑性、黏弹性等本构模型

#pragma once

#include <Eigen/Dense>
#include <memory>

namespace particle_simulation {

// 本构模型类型枚举
enum class ConstitutiveModelType {
    ELASTIC,             // 弹性模型
    MOHR_COULOMB,        // Mohr-Coulomb弹塑性模型
    VISCO_ELASTIC,       // 黏弹性模型
    DMC                  // 改进的Mohr-Coulomb模型（预留）
};

// 应力应变状态结构体
struct StressStrainState {
    Eigen::Matrix3d stress;       // 应力张量
    Eigen::Matrix3d strain;       // 应变张量
    Eigen::Matrix3d strain_rate;  // 应变率张量
    double density;               // 密度
    double pressure;              // 压力
    Eigen::Vector3d principal_stresses;  // 主应力
};

// 本构模型基类
class ConstitutiveModel {
public:
    // 构造函数
    ConstitutiveModel(ConstitutiveModelType type);
    
    // 析构函数
    virtual ~ConstitutiveModel() = default;
    
    // 计算应力更新
    // 参数：
    // - state: 应力应变状态
    // - dt: 时间步长
    // 返回：更新后的应力张量
    virtual Eigen::Matrix3d updateStress(
        const StressStrainState& state,
        double dt
    ) const = 0;
    
    // 计算弹性模量张量
    // 返回：弹性模量张量
    virtual Eigen::MatrixXd getElasticModulus() const = 0;
    
    // 获取本构模型类型
    ConstitutiveModelType getType() const;
    
    // 设置模型参数
    // 参数：
    // - params: 模型参数向量
    virtual void setParameters(const Eigen::VectorXd& params) = 0;
    
    // 获取模型参数
    virtual Eigen::VectorXd getParameters() const = 0;
    
    // 检查模型是否处于屈服状态
    // 参数：
    // - state: 应力应变状态
    // 返回：是否屈服
    virtual bool isYielding(const StressStrainState& state) const = 0;
    
private:
    ConstitutiveModelType type_;
};

// 弹性本构模型
class ElasticModel : public ConstitutiveModel {
public:
    // 构造函数
    ElasticModel();
    ElasticModel(double young_modulus, double poisson_ratio);
    
    // 计算应力更新
    Eigen::Matrix3d updateStress(
        const StressStrainState& state,
        double dt
    ) const override;
    
    // 计算弹性模量张量
    Eigen::MatrixXd getElasticModulus() const override;
    
    // 设置模型参数
    void setParameters(const Eigen::VectorXd& params) override;
    
    // 获取模型参数
    Eigen::VectorXd getParameters() const override;
    
    // 检查模型是否处于屈服状态
    bool isYielding(const StressStrainState& state) const override;
    
private:
    double young_modulus_;    // 弹性模量
    double poisson_ratio_;    // 泊松比
    double shear_modulus_;    // 剪切模量
    double bulk_modulus_;     // 体积模量
    
    // 更新弹性常数
    void updateElasticConstants();
};

// Mohr-Coulomb弹塑性本构模型
class MohrCoulombModel : public ConstitutiveModel {
public:
    // 构造函数
    MohrCoulombModel();
    MohrCoulombModel(
        double young_modulus,
        double poisson_ratio,
        double cohesion,
        double friction_angle,
        double dilation_angle
    );
    
    // 计算应力更新
    Eigen::Matrix3d updateStress(
        const StressStrainState& state,
        double dt
    ) const override;
    
    // 计算弹性模量张量
    Eigen::MatrixXd getElasticModulus() const override;
    
    // 设置模型参数
    void setParameters(const Eigen::VectorXd& params) override;
    
    // 获取模型参数
    Eigen::VectorXd getParameters() const override;
    
    // 检查模型是否处于屈服状态
    bool isYielding(const StressStrainState& state) const override;
    
private:
    double young_modulus_;    // 弹性模量
    double poisson_ratio_;    // 泊松比
    double cohesion_;         // 黏聚力
    double friction_angle_;   // 内摩擦角（弧度）
    double dilation_angle_;   // 剪胀角（弧度）
    double shear_modulus_;    // 剪切模量
    double bulk_modulus_;     // 体积模量
    
    // 更新弹性常数
    void updateElasticConstants();
    
    // 计算屈服函数值
    double computeYieldFunction(const StressStrainState& state) const;
    
    // 计算塑性流动方向
    Eigen::Matrix3d computePlasticFlowDirection(const StressStrainState& state) const;
    
    // 计算硬化参数
    double computeHardeningParameter(const StressStrainState& state) const;
};

// 黏弹性本构模型
class ViscoElasticModel : public ConstitutiveModel {
public:
    // 构造函数
    ViscoElasticModel();
    ViscoElasticModel(
        double young_modulus,
        double poisson_ratio,
        double viscosity,
        double relaxation_time
    );
    
    // 计算应力更新
    Eigen::Matrix3d updateStress(
        const StressStrainState& state,
        double dt
    ) const override;
    
    // 计算弹性模量张量
    Eigen::MatrixXd getElasticModulus() const override;
    
    // 设置模型参数
    void setParameters(const Eigen::VectorXd& params) override;
    
    // 获取模型参数
    Eigen::VectorXd getParameters() const override;
    
    // 检查模型是否处于屈服状态
    bool isYielding(const StressStrainState& state) const override;
    
private:
    double young_modulus_;    // 弹性模量
    double poisson_ratio_;    // 泊松比
    double viscosity_;        // 黏度
    double relaxation_time_;  // 松弛时间
    double shear_modulus_;    // 剪切模量
    double bulk_modulus_;     // 体积模量
    
    // 更新弹性常数
    void updateElasticConstants();
};

// 本构模型工厂类
class ConstitutiveModelFactory {
public:
    // 创建本构模型实例
    // 参数：
    // - type: 本构模型类型
    // - params: 模型参数
    // 返回：本构模型指针
    static std::unique_ptr<ConstitutiveModel> createModel(
        ConstitutiveModelType type,
        const Eigen::VectorXd& params = Eigen::VectorXd()
    );
};

} // namespace particle_simulation
