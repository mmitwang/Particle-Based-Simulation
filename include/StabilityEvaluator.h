// StabilityEvaluator.h
// 稳定性评估模块
// 实现边坡稳定性关键性能指标的计算和评估

#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <map>
#include "Particle.h"

namespace particle_simulation {

// 边坡稳定性指标结构体
/**
 * @brief 边坡稳定性指标结构体
 * 
 * 包含边坡稳定性的关键性能指标
 * 
 * @param safety_factor 边坡安全系数
 * @param max_displacement 最大位移（m）
 * @param average_displacement 平均位移（m）
 * @param displacement_variance 位移方差（m²）
 * @param max_stress 最大应力（Pa）
 * @param max_strain 最大应变
 * @param potential_slide_surface 潜在滑动面位置
 * @param instability_warning_level 失稳预警等级（0-100）
 * @param critical_stress_area 应力集中区域数量
 */
struct StabilityIndices {
    double safety_factor;                // 安全系数
    double max_displacement;             // 最大位移
    double average_displacement;         // 平均位移
    double displacement_variance;        // 位移方差
    double max_stress;                   // 最大应力
    double max_strain;                   // 最大应变
    std::vector<Eigen::Vector3d> potential_slide_surface; // 潜在滑动面
    int instability_warning_level;       // 失稳预警等级（0-100）
    int critical_stress_area;            // 应力集中区域数量
    
    // 构造函数
    StabilityIndices() :
        safety_factor(1.0),
        max_displacement(0.0),
        average_displacement(0.0),
        displacement_variance(0.0),
        max_stress(0.0),
        max_strain(0.0),
        instability_warning_level(0),
        critical_stress_area(0) {
    }
};

// 位移场分布结构体
/**
 * @brief 位移场分布结构体
 * 
 * 包含位移场的分布信息
 * 
 * @param displacement_map 位移值与对应的粒子ID映射
 * @param displacement_gradient 位移梯度场
 * @param max_displacement_location 最大位移位置
 * @param displacement_contours 位移等值线数据
 */
struct DisplacementField {
    std::map<size_t, Eigen::Vector3d> displacement_map; // 粒子ID到位移的映射
    Eigen::MatrixXd displacement_gradient;              // 位移梯度场
    Eigen::Vector3d max_displacement_location;         // 最大位移位置
    std::vector<std::vector<Eigen::Vector3d>> displacement_contours; // 位移等值线
};

// 应力应变集中区域结构体
/**
 * @brief 应力应变集中区域结构体
 * 
 * 包含应力应变集中区域的信息
 * 
 * @param stress_concentration_areas 应力集中区域列表
 * @param strain_concentration_areas 应变集中区域列表
 * @param max_stress_location 最大应力位置
 * @param max_strain_location 最大应变位置
 */
struct StressStrainConcentration {
    std::vector<std::vector<size_t>> stress_concentration_areas; // 应力集中区域（粒子ID列表）
    std::vector<std::vector<size_t>> strain_concentration_areas; // 应变集中区域（粒子ID列表）
    Eigen::Vector3d max_stress_location;                          // 最大应力位置
    Eigen::Vector3d max_strain_location;                          // 最大应变位置
};

// 稳定性评估器类
/**
 * @brief 稳定性评估器类
 * 
 * 实现边坡稳定性关键性能指标的计算和评估
 * 
 * 核心功能：
 * 1. 计算边坡安全系数
 * 2. 分析位移场分布
 * 3. 检测应力应变集中区域
 * 4. 识别潜在滑动面位置
 * 5. 评估失稳预警阈值
 * 6. 生成稳定性评估报告
 * 
 * 核心算法：
 * 1. 安全系数计算：采用简化Bishop法、Janbu法或有限元强度折减法
 * 2. 位移场分析：基于粒子位移数据，计算位移梯度、等值线等
 * 3. 应力应变集中检测：采用聚类分析或梯度分析方法
 * 4. 潜在滑动面识别：基于位移突变、应力分布或能量法
 * 
 * 理论基础：
 * - 边坡稳定性理论（极限平衡法、有限元法）
 * - 统计学分析方法（方差、聚类分析）
 * - 岩土力学中的强度理论（Mohr-Coulomb准则）
 */
class StabilityEvaluator {
public:
    // 安全系数计算方法枚举
enum class SafetyFactorMethod {
    SIMPLIFIED_BISHOP,   // 简化Bishop法
    JANBU,               // Janbu法
    STRENGTH_REDUCTION,  // 强度折减法
    ENERGY_METHOD        // 能量法
};
    
    // 构造函数
    StabilityEvaluator(SafetyFactorMethod method = SafetyFactorMethod::SIMPLIFIED_BISHOP);
    
    /**
     * @brief 评估边坡稳定性
     * 
     * 计算边坡的各项稳定性指标
     * 
     * @param particles 粒子列表
     * @param initial_particles 初始粒子列表（用于计算位移）
     * @param gravity 重力加速度
     * @return 稳定性指标结构体
     * 
     * 算法流程：
     * 1. 计算位移场分布
     * 2. 分析应力应变分布
     * 3. 计算安全系数
     * 4. 检测潜在滑动面
     * 5. 评估失稳预警等级
     */
    StabilityIndices evaluateStability(
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles,
        const Eigen::Vector3d& gravity = Eigen::Vector3d(0.0, -9.81, 0.0)
    );
    
    /**
     * @brief 计算边坡安全系数
     * 
     * 采用指定方法计算边坡的安全系数
     * 
     * @param particles 粒子列表
     * @param gravity 重力加速度
     * @return 安全系数
     * 
     * 安全系数定义：
     * - 安全系数 = 抗滑力 / 滑动力
     * - 安全系数 > 1.5：稳定
     * - 1.0 < 安全系数 <= 1.5：基本稳定
     * - 安全系数 <= 1.0：不稳定
     */
    double calculateSafetyFactor(
        const std::vector<Particle>& particles,
        const Eigen::Vector3d& gravity = Eigen::Vector3d(0.0, -9.81, 0.0)
    );
    
    /**
     * @brief 分析位移场分布
     * 
     * 计算位移场的各项指标
     * 
     * @param particles 粒子列表
     * @param initial_particles 初始粒子列表
     * @return 位移场分布结构体
     */
    DisplacementField analyzeDisplacementField(
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    );
    
    /**
     * @brief 检测应力应变集中区域
     * 
     * 识别边坡中的应力应变集中区域
     * 
     * @param particles 粒子列表
     * @return 应力应变集中区域结构体
     */
    StressStrainConcentration detectStressStrainConcentration(
        const std::vector<Particle>& particles
    );
    
    /**
     * @brief 识别潜在滑动面
     * 
     * 基于粒子位移和应力分布，识别潜在的滑动面
     * 
     * @param particles 粒子列表
     * @param initial_particles 初始粒子列表
     * @return 潜在滑动面位置列表
     */
    std::vector<Eigen::Vector3d> identifyPotentialSlideSurface(
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    );
    
    /**
     * @brief 评估失稳预警阈值
     * 
     * 根据稳定性指标，评估失稳预警等级
     * 
     * @param indices 稳定性指标
     * @return 失稳预警等级（0-100）
     * 
     * 预警等级定义：
     * - 0-20：稳定
     * - 21-40：基本稳定
     * - 41-60：轻度预警
     * - 61-80：中度预警
     * - 81-100：重度预警
     */
    int evaluateInstabilityWarning(const StabilityIndices& indices);
    
    /**
     * @brief 生成稳定性评估报告
     * 
     * 将稳定性评估结果输出为报告文件
     * 
     * @param indices 稳定性指标
     * @param report_path 报告文件路径
     * @return 是否生成成功
     */
    bool generateStabilityReport(
        const StabilityIndices& indices,
        const std::string& report_path
    );
    
    /**
     * @brief 设置安全系数计算方法
     * 
     * @param method 安全系数计算方法
     */
    void setSafetyFactorMethod(SafetyFactorMethod method);
    
    /**
     * @brief 设置预警阈值参数
     * 
     * @param critical_displacement 临界位移阈值（m）
     * @param critical_stress 临界应力阈值（Pa）
     * @param safety_factor_threshold 安全系数阈值
     */
    void setWarningThresholds(
        double critical_displacement,
        double critical_stress,
        double safety_factor_threshold
    );
    
private:
    // 计算简化Bishop法安全系数
    double calculateSimplifiedBishop(
        const std::vector<Particle>& particles,
        const Eigen::Vector3d& gravity
    );
    
    // 计算Janbu法安全系数
    double calculateJanbu(
        const std::vector<Particle>& particles,
        const Eigen::Vector3d& gravity
    );
    
    // 计算强度折减法安全系数
    double calculateStrengthReduction(
        const std::vector<Particle>& particles,
        const Eigen::Vector3d& gravity
    );
    
    // 计算能量法安全系数
    double calculateEnergyMethod(
        const std::vector<Particle>& particles,
        const Eigen::Vector3d& gravity
    );
    
    // 检测位移突变区域
    std::vector<Eigen::Vector3d> detectDisplacementJump(
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    );
    
    // 计算位移梯度
    Eigen::MatrixXd computeDisplacementGradient(
        const std::vector<Particle>& particles,
        const std::vector<Particle>& initial_particles
    );
    
    // 聚类分析，识别集中区域
    std::vector<std::vector<size_t>> clusterAnalysis(
        const std::vector<Particle>& particles,
        double threshold
    );
    
    SafetyFactorMethod safety_factor_method_; // 安全系数计算方法
    
    // 预警阈值参数
    double critical_displacement_;        // 临界位移阈值
    double critical_stress_;              // 临界应力阈值
    double safety_factor_threshold_;      // 安全系数阈值
    
    // 用于计算的中间变量
    std::vector<Eigen::Vector3d> displacements_; // 粒子位移列表
};

} // namespace particle_simulation
