// StabilityEvaluator.cpp
// 稳定性评估模块实现

#include "StabilityEvaluator.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace particle_simulation {

// 构造函数
StabilityEvaluator::StabilityEvaluator(SafetyFactorMethod method)
    : safety_factor_method_(method),
      critical_displacement_(0.1),
      critical_stress_(1.0e6),
      safety_factor_threshold_(1.2) {
}

// 评估边坡稳定性
StabilityIndices StabilityEvaluator::evaluateStability(
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles,
    const Eigen::Vector3d& gravity
) {
    StabilityIndices indices;
    
    // 检查粒子数量是否一致
    if (particles.size() != initial_particles.size()) {
        std::cerr << "Particle count mismatch!" << std::endl;
        return indices;
    }
    
    // 初始化位移列表
    displacements_.resize(particles.size());
    for (size_t i = 0; i < particles.size(); ++i) {
        displacements_[i] = particles[i].getPosition() - initial_particles[i].getPosition();
    }
    
    // 计算位移场分布
    DisplacementField displacement_field = analyzeDisplacementField(particles, initial_particles);
    
    // 计算安全系数
    indices.safety_factor = calculateSafetyFactor(particles, gravity);
    
    // 分析应力应变集中区域
    StressStrainConcentration stress_strain_concentration = detectStressStrainConcentration(particles);
    indices.critical_stress_area = stress_strain_concentration.stress_concentration_areas.size();
    
    // 识别潜在滑动面
    indices.potential_slide_surface = identifyPotentialSlideSurface(particles, initial_particles);
    
    // 计算位移统计指标
    indices.max_displacement = 0.0;
    indices.average_displacement = 0.0;
    indices.displacement_variance = 0.0;
    
    double total_displacement = 0.0;
    double total_displacement_sq = 0.0;
    
    for (const auto& disp : displacements_) {
        double disp_magnitude = disp.norm();
        indices.max_displacement = std::max(indices.max_displacement, disp_magnitude);
        total_displacement += disp_magnitude;
        total_displacement_sq += disp_magnitude * disp_magnitude;
    }
    
    indices.average_displacement = total_displacement / particles.size();
    indices.displacement_variance = (total_displacement_sq / particles.size()) - (indices.average_displacement * indices.average_displacement);
    
    // 计算最大应力和应变
    indices.max_stress = 0.0;
    indices.max_strain = 0.0;
    
    for (const auto& particle : particles) {
        double stress_magnitude = particle.getStress().norm();
        double strain_magnitude = particle.getStrain().norm();
        
        indices.max_stress = std::max(indices.max_stress, stress_magnitude);
        indices.max_strain = std::max(indices.max_strain, strain_magnitude);
    }
    
    // 评估失稳预警等级
    indices.instability_warning_level = evaluateInstabilityWarning(indices);
    
    return indices;
}

// 计算边坡安全系数
double StabilityEvaluator::calculateSafetyFactor(
    const std::vector<Particle>& particles,
    const Eigen::Vector3d& gravity
) {
    double safety_factor = 1.0;
    
    switch (safety_factor_method_) {
        case SafetyFactorMethod::SIMPLIFIED_BISHOP:
            safety_factor = calculateSimplifiedBishop(particles, gravity);
            break;
        case SafetyFactorMethod::JANBU:
            safety_factor = calculateJanbu(particles, gravity);
            break;
        case SafetyFactorMethod::STRENGTH_REDUCTION:
            safety_factor = calculateStrengthReduction(particles, gravity);
            break;
        case SafetyFactorMethod::ENERGY_METHOD:
            safety_factor = calculateEnergyMethod(particles, gravity);
            break;
        default:
            safety_factor = 1.0;
            break;
    }
    
    return safety_factor;
}

// 分析位移场分布
DisplacementField StabilityEvaluator::analyzeDisplacementField(
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles
) {
    DisplacementField displacement_field;
    
    // 检查粒子数量是否一致
    if (particles.size() != initial_particles.size()) {
        std::cerr << "Particle count mismatch!" << std::endl;
        return displacement_field;
    }
    
    // 计算每个粒子的位移
    displacement_field.displacement_map.clear();
    Eigen::Vector3d max_displacement = Eigen::Vector3d::Zero();
    size_t max_displacement_id = 0;
    
    for (size_t i = 0; i < particles.size(); ++i) {
        Eigen::Vector3d displacement = particles[i].getPosition() - initial_particles[i].getPosition();
        displacement_field.displacement_map[i] = displacement;
        
        if (displacement.norm() > max_displacement.norm()) {
            max_displacement = displacement;
            max_displacement_id = i;
        }
    }
    
    // 获取最大位移位置
    displacement_field.max_displacement_location = particles[max_displacement_id].getPosition();
    
    // 计算位移梯度
    displacement_field.displacement_gradient = computeDisplacementGradient(particles, initial_particles);
    
    return displacement_field;
}

// 检测应力应变集中区域
StressStrainConcentration StabilityEvaluator::detectStressStrainConcentration(
    const std::vector<Particle>& particles
) {
    StressStrainConcentration concentration;
    
    // 简化实现：基于阈值检测
    
    // 检测应力集中区域
    concentration.stress_concentration_areas.clear();
    concentration.max_stress = Eigen::Vector3d::Zero();
    size_t max_stress_id = 0;
    double max_stress_magnitude = 0.0;
    
    for (size_t i = 0; i < particles.size(); ++i) {
        double stress_magnitude = particles[i].getStress().norm();
        
        if (stress_magnitude > critical_stress_) {
            concentration.stress_concentration_areas.push_back({i});
        }
        
        if (stress_magnitude > max_stress_magnitude) {
            max_stress_magnitude = stress_magnitude;
            max_stress_id = i;
        }
    }
    
    concentration.max_stress_location = particles[max_stress_id].getPosition();
    
    // 检测应变集中区域
    concentration.strain_concentration_areas.clear();
    concentration.max_strain = Eigen::Vector3d::Zero();
    size_t max_strain_id = 0;
    double max_strain_magnitude = 0.0;
    
    for (size_t i = 0; i < particles.size(); ++i) {
        double strain_magnitude = particles[i].getStrain().norm();
        
        if (strain_magnitude > 0.01) { // 应变阈值设为0.01
            concentration.strain_concentration_areas.push_back({i});
        }
        
        if (strain_magnitude > max_strain_magnitude) {
            max_strain_magnitude = strain_magnitude;
            max_strain_id = i;
        }
    }
    
    concentration.max_strain_location = particles[max_strain_id].getPosition();
    
    return concentration;
}

// 识别潜在滑动面
std::vector<Eigen::Vector3d> StabilityEvaluator::identifyPotentialSlideSurface(
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles
) {
    // 简化实现：基于位移突变检测
    return detectDisplacementJump(particles, initial_particles);
}

// 评估失稳预警阈值
int StabilityEvaluator::evaluateInstabilityWarning(const StabilityIndices& indices) {
    int warning_level = 0;
    
    // 基于安全系数的预警
    if (indices.safety_factor < 1.0) {
        warning_level += 40;
    } else if (indices.safety_factor < safety_factor_threshold_) {
        warning_level += 20;
    }
    
    // 基于位移的预警
    if (indices.max_displacement > critical_displacement_) {
        warning_level += 30;
    } else if (indices.max_displacement > critical_displacement_ * 0.5) {
        warning_level += 15;
    }
    
    // 基于应力集中的预警
    if (indices.critical_stress_area > 10) {
        warning_level += 20;
    } else if (indices.critical_stress_area > 5) {
        warning_level += 10;
    }
    
    // 基于应变的预警
    if (indices.max_strain > 0.05) {
        warning_level += 10;
    }
    
    // 确保预警等级在0-100范围内
    warning_level = std::max(0, std::min(100, warning_level));
    
    return warning_level;
}

// 生成稳定性评估报告
bool StabilityEvaluator::generateStabilityReport(
    const StabilityIndices& indices,
    const std::string& report_path
) {
    std::ofstream report_file(report_path);
    
    if (!report_file.is_open()) {
        std::cerr << "Failed to open report file: " << report_path << std::endl;
        return false;
    }
    
    // 写入报告标题
    report_file << "========================================" << std::endl;
    report_file << "          岩土边坡稳定性评估报告          " << std::endl;
    report_file << "========================================" << std::endl;
    report_file << std::endl;
    
    // 写入基本信息
    report_file << "1. 基本信息" << std::endl;
    report_file << "----------------------------------------" << std::endl;
    report_file << "安全系数计算方法: ";
    switch (safety_factor_method_) {
        case SafetyFactorMethod::SIMPLIFIED_BISHOP:
            report_file << "简化Bishop法" << std::endl;
            break;
        case SafetyFactorMethod::JANBU:
            report_file << "Janbu法" << std::endl;
            break;
        case SafetyFactorMethod::STRENGTH_REDUCTION:
            report_file << "强度折减法" << std::endl;
            break;
        case SafetyFactorMethod::ENERGY_METHOD:
            report_file << "能量法" << std::endl;
            break;
        default:
            report_file << "未知方法" << std::endl;
            break;
    }
    report_file << std::endl;
    
    // 写入稳定性指标
    report_file << "2. 稳定性指标" << std::endl;
    report_file << "----------------------------------------" << std::endl;
    report_file << std::fixed << std::setprecision(6);
    report_file << "安全系数: " << indices.safety_factor << std::endl;
    report_file << "最大位移: " << indices.max_displacement << " m" << std::endl;
    report_file << "平均位移: " << indices.average_displacement << " m" << std::endl;
    report_file << "位移方差: " << indices.displacement_variance << " m²" << std::endl;
    report_file << "最大应力: " << indices.max_stress << " Pa" << std::endl;
    report_file << "最大应变: " << indices.max_strain << std::endl;
    report_file << "应力集中区域数量: " << indices.critical_stress_area << std::endl;
    report_file << std::endl;
    
    // 写入潜在滑动面信息
    report_file << "3. 潜在滑动面" << std::endl;
    report_file << "----------------------------------------" << std::endl;
    report_file << "潜在滑动面点数量: " << indices.potential_slide_surface.size() << std::endl;
    if (!indices.potential_slide_surface.empty()) {
        report_file << "滑动面位置（部分）: " << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), indices.potential_slide_surface.size()); ++i) {
            const Eigen::Vector3d& point = indices.potential_slide_surface[i];
            report_file << "  " << i+1 << ": (" << point.x() << ", " << point.y() << ", " << point.z() << ")" << std::endl;
        }
    }
    report_file << std::endl;
    
    // 写入失稳预警信息
    report_file << "4. 失稳预警评估" << std::endl;
    report_file << "----------------------------------------" << std::endl;
    report_file << "失稳预警等级: " << indices.instability_warning_level << " (0-100)" << std::endl;
    report_file << "预警等级说明: " << std::endl;
    
    if (indices.instability_warning_level <= 20) {
        report_file << "  稳定: 边坡处于稳定状态，无明显失稳风险" << std::endl;
    } else if (indices.instability_warning_level <= 40) {
        report_file << "  基本稳定: 边坡基本稳定，需定期监测" << std::endl;
    } else if (indices.instability_warning_level <= 60) {
        report_file << "  轻度预警: 边坡出现轻微失稳迹象，需加强监测" << std::endl;
    } else if (indices.instability_warning_level <= 80) {
        report_file << "  中度预警: 边坡出现明显失稳迹象，需采取防护措施" << std::endl;
    } else {
        report_file << "  重度预警: 边坡处于危险状态，极可能发生失稳" << std::endl;
    }
    
    // 写入安全建议
    report_file << std::endl;
    report_file << "5. 安全建议" << std::endl;
    report_file << "----------------------------------------" << std::endl;
    
    if (indices.safety_factor < 1.0) {
        report_file << "- 边坡安全系数小于1.0，处于不稳定状态，建议立即采取加固措施" << std::endl;
    } else if (indices.safety_factor < safety_factor_threshold_) {
        report_file << "- 边坡安全系数较低，建议加强监测并考虑加固措施" << std::endl;
    }
    
    if (indices.max_displacement > critical_displacement_) {
        report_file << "- 边坡位移超过临界值，建议立即停止加载并采取应急措施" << std::endl;
    }
    
    if (indices.critical_stress_area > 10) {
        report_file << "- 边坡存在多处应力集中区域，建议进行局部加固" << std::endl;
    }
    
    report_file << "- 建议定期监测边坡位移、应力等参数，及时掌握边坡状态变化" << std::endl;
    report_file << "- 根据监测结果调整防护措施，确保边坡安全" << std::endl;
    
    // 结束报告
    report_file << "========================================" << std::endl;
    report_file << "报告生成时间: " << __DATE__ << " " << __TIME__ << std::endl;
    report_file << "========================================" << std::endl;
    
    report_file.close();
    
    std::cout << "Stability report generated: " << report_path << std::endl;
    return true;
}

// 设置安全系数计算方法
void StabilityEvaluator::setSafetyFactorMethod(SafetyFactorMethod method) {
    safety_factor_method_ = method;
}

// 设置预警阈值参数
void StabilityEvaluator::setWarningThresholds(
    double critical_displacement,
    double critical_stress,
    double safety_factor_threshold
) {
    critical_displacement_ = critical_displacement;
    critical_stress_ = critical_stress;
    safety_factor_threshold_ = safety_factor_threshold;
}

// 计算简化Bishop法安全系数
double StabilityEvaluator::calculateSimplifiedBishop(
    const std::vector<Particle>& particles,
    const Eigen::Vector3d& gravity
) {
    // 简化Bishop法实现
    // 基本原理：基于极限平衡理论，假设滑动面为圆弧，计算抗滑力矩与滑动力矩的比值
    
    // 简化实现：返回基于平均应力和强度的安全系数
    double total_stress = 0.0;
    double total_strength = 0.0;
    
    for (const auto& particle : particles) {
        // 计算平均应力
        double stress = particle.getStress().norm();
        total_stress += stress;
        
        // 计算抗剪强度（基于Mohr-Coulomb准则）
        const Eigen::VectorXd& params = particle.getLithologyParams();
        double cohesion = params[2];
        double friction_angle = params[3] * M_PI / 180.0; // 转换为弧度
        double normal_stress = -particle.getStress().trace() / 3.0; // 有效应力
        double strength = cohesion + normal_stress * tan(friction_angle);
        total_strength += strength;
    }
    
    if (total_stress == 0.0) {
        return 1.0;
    }
    
    return total_strength / total_stress;
}

// 计算Janbu法安全系数
double StabilityEvaluator::calculateJanbu(
    const std::vector<Particle>& particles,
    const Eigen::Vector3d& gravity
) {
    // 简化实现：与Bishop法类似
    return calculateSimplifiedBishop(particles, gravity);
}

// 计算强度折减法安全系数
double StabilityEvaluator::calculateStrengthReduction(
    const std::vector<Particle>& particles,
    const Eigen::Vector3d& gravity
) {
    // 简化实现：返回基于位移的安全系数估计
    double max_displacement = 0.0;
    
    for (const auto& displacement : displacements_) {
        max_displacement = std::max(max_displacement, displacement.norm());
    }
    
    // 基于位移的简化安全系数计算
    if (max_displacement < 1.0e-4) {
        return 2.0; // 位移很小，安全系数高
    } else if (max_displacement < 0.01) {
        return 1.5;
    } else if (max_displacement < 0.1) {
        return 1.2;
    } else {
        return 0.8; // 位移很大，安全系数低
    }
}

// 计算能量法安全系数
double StabilityEvaluator::calculateEnergyMethod(
    const std::vector<Particle>& particles,
    const Eigen::Vector3d& gravity
) {
    // 简化实现：基于势能变化的安全系数计算
    double total_kinetic_energy = 0.0;
    double total_potential_energy = 0.0;
    
    for (const auto& particle : particles) {
        // 计算动能
        double velocity_sq = particle.getVelocity().squaredNorm();
        total_kinetic_energy += 0.5 * particle.getMass() * velocity_sq;
        
        // 计算势能
        double height = particle.getPosition().y();
        total_potential_energy += particle.getMass() * gravity.norm() * height;
    }
    
    // 基于能量的简化安全系数计算
    if (total_kinetic_energy < 1.0e-4) {
        return 1.5; // 动能很小，安全系数高
    } else {
        return std::max(0.5, 1.5 - total_kinetic_energy / 100.0);
    }
}

// 检测位移突变区域
std::vector<Eigen::Vector3d> StabilityEvaluator::detectDisplacementJump(
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles
) {
    std::vector<Eigen::Vector3d> jump_points;
    
    // 简化实现：检测位移梯度超过阈值的区域
    double displacement_gradient_threshold = 1.0; // 位移梯度阈值
    
    // 计算位移梯度（简化为相邻粒子的位移差）
    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {
            // 计算粒子间距离
            double distance = (particles[i].getPosition() - particles[j].getPosition()).norm();
            if (distance > 0.1) { // 只考虑距离较近的粒子
                continue;
            }
            
            // 计算位移差
            Eigen::Vector3d disp_i = particles[i].getPosition() - initial_particles[i].getPosition();
            Eigen::Vector3d disp_j = particles[j].getPosition() - initial_particles[j].getPosition();
            double disp_diff = (disp_i - disp_j).norm();
            
            // 计算位移梯度
            double disp_gradient = disp_diff / distance;
            
            if (disp_gradient > displacement_gradient_threshold) {
                // 记录位移突变点
                Eigen::Vector3d mid_point = (particles[i].getPosition() + particles[j].getPosition()) / 2.0;
                jump_points.push_back(mid_point);
            }
        }
    }
    
    return jump_points;
}

// 计算位移梯度
Eigen::MatrixXd StabilityEvaluator::computeDisplacementGradient(
    const std::vector<Particle>& particles,
    const std::vector<Particle>& initial_particles
) {
    // 简化实现：返回单位矩阵
    Eigen::MatrixXd gradient = Eigen::MatrixXd::Identity(3, 3);
    return gradient;
}

} // namespace particle_simulation
