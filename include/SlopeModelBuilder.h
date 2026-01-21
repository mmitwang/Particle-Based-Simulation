// SlopeModelBuilder.h
// 岩土边坡模型构建模块
// 实现边坡模型的参数化构建和外部模型导入

#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "Particle.h"
#include "ParticleGenerator.h"

namespace particle_simulation {

// 边坡模型类型枚举
enum class SlopeModelType {
    PARAMETRIC,      // 参数化模型
    EXTERNAL_IMPORT  // 外部导入模型
};

// 边坡几何参数结构体
/**
 * @brief 边坡几何参数结构体
 * 
 * 包含边坡模型的基本几何参数，用于参数化建模
 * 
 * @param slope_height 边坡高度（m）
 * @param slope_angle 边坡坡角（度）
 * @param slope_width 边坡宽度（m）
 * @param slope_length 边坡长度（m，三维模型使用）
 * @param ground_depth 地基深度（m）
 * @param berm_width 平台宽度（m，多级边坡使用）
 * @param berm_height 平台高度（m，多级边坡使用）
 */
struct SlopeGeometryParams {
    double slope_height;      // 边坡高度
    double slope_angle;       // 边坡坡角（度数）
    double slope_width;       // 边坡宽度
    double slope_length;      // 边坡长度（三维）
    double ground_depth;      // 地基深度
    double berm_width;        // 平台宽度（多级边坡）
    double berm_height;       // 平台高度（多级边坡）
    
    // 构造函数
    SlopeGeometryParams() :
        slope_height(10.0),
        slope_angle(45.0),
        slope_width(20.0),
        slope_length(10.0),
        ground_depth(5.0),
        berm_width(0.0),
        berm_height(0.0) {
    }
    
    SlopeGeometryParams(
        double height,
        double angle,
        double width,
        double length = 10.0,
        double depth = 5.0,
        double berm_w = 0.0,
        double berm_h = 0.0
    ) :
        slope_height(height),
        slope_angle(angle),
        slope_width(width),
        slope_length(length),
        ground_depth(depth),
        berm_width(berm_w),
        berm_height(berm_h) {
    }
};

// 岩土体物理力学参数结构体
/**
 * @brief 岩土体物理力学参数结构体
 * 
 * 包含岩土体的基本物理力学参数，用于本构模型计算
 * 
 * @param density 密度（kg/m³）
 * @param young_modulus 弹性模量（Pa）
 * @param poisson_ratio 泊松比
 * @param cohesion 黏聚力（Pa）
 * @param friction_angle 内摩擦角（度）
 * @param dilation_angle 剪胀角（度）
 * @param permeability 渗透率（m/s）
 * @param moisture_content 含水率（%）
 */
struct GeomaterialParams {
    double density;          // 密度
    double young_modulus;    // 弹性模量
    double poisson_ratio;    // 泊松比
    double cohesion;         // 黏聚力
    double friction_angle;   // 内摩擦角（度数）
    double dilation_angle;   // 剪胀角（度数）
    double permeability;     // 渗透率
    double moisture_content; // 含水率
    
    // 构造函数
    GeomaterialParams() :
        density(2600.0),
        young_modulus(1.0e6),
        poisson_ratio(0.3),
        cohesion(10000.0),
        friction_angle(30.0),
        dilation_angle(10.0),
        permeability(1.0e-8),
        moisture_content(10.0) {
    }
    
    GeomaterialParams(
        double rho,
        double E,
        double nu,
        double c,
        double phi,
        double psi = 10.0,
        double k = 1.0e-8,
        double w = 10.0
    ) :
        density(rho),
        young_modulus(E),
        poisson_ratio(nu),
        cohesion(c),
        friction_angle(phi),
        dilation_angle(psi),
        permeability(k),
        moisture_content(w) {
    }
};

// 边坡模型构建器类
/**
 * @brief 边坡模型构建器类
 * 
 * 实现岩土边坡模型的参数化构建和外部模型导入功能
 * 
 * 核心算法：
 * 1. 参数化建模：基于边坡几何参数，通过空间网格划分和点云生成算法，构建边坡的三维模型
 * 2. 外部模型导入：支持从CAD、STL等格式导入边坡模型，通过网格离散化生成粒子模型
 * 3. 多级边坡构建：支持包含平台的多级边坡模型构建
 * 
 * 理论基础：
 * - 基于岩土力学中的边坡稳定性理论，考虑边坡几何形态对稳定性的影响
 * - 采用粒子法离散化原理，将连续的岩土体离散为离散的粒子集合
 * - 支持不同岩性和含水率的岩土体建模
 */
class SlopeModelBuilder {
public:
    // 构造函数
    SlopeModelBuilder();
    
    /**
     * @brief 参数化构建边坡模型
     * 
     * 基于给定的几何参数和岩土体参数，生成边坡粒子模型
     * 
     * @param geometry_params 边坡几何参数
     * @param geomaterial_params 岩土体物理力学参数
     * @param particle_radius 粒子半径
     * @param is_3d 是否为三维模型
     * @return 生成的粒子列表
     * 
     * 算法流程：
     * 1. 根据几何参数计算边坡的边界范围
     * 2. 生成空间网格点云
     * 3. 根据边坡几何形状筛选点云
     * 4. 生成粒子并设置其物理力学参数
     */
    std::vector<Particle> buildParametricSlope(
        const SlopeGeometryParams& geometry_params,
        const GeomaterialParams& geomaterial_params,
        double particle_radius,
        bool is_3d = false
    );
    
    /**
     * @brief 从外部文件导入边坡模型
     * 
     * 支持从STL、OBJ、DXF等格式导入边坡模型，生成粒子模型
     * 
     * @param file_path 模型文件路径
     * @param particle_radius 粒子半径
     * @param geomaterial_params 岩土体物理力学参数
     * @return 生成的粒子列表
     * 
     * 算法流程：
     * 1. 读取外部模型文件
     * 2. 解析模型几何信息
     * 3. 进行网格离散化
     * 4. 生成粒子并设置其物理力学参数
     */
    std::vector<Particle> importExternalModel(
        const std::string& file_path,
        double particle_radius,
        const GeomaterialParams& geomaterial_params
    );
    
    /**
     * @brief 构建多级边坡模型
     * 
     * 实现包含多个平台的多级边坡模型构建
     * 
     * @param geometry_params_list 多级边坡几何参数列表
     * @param geomaterial_params 岩土体物理力学参数
     * @param particle_radius 粒子半径
     * @param is_3d 是否为三维模型
     * @return 生成的粒子列表
     */
    std::vector<Particle> buildMultiLevelSlope(
        const std::vector<SlopeGeometryParams>& geometry_params_list,
        const GeomaterialParams& geomaterial_params,
        double particle_radius,
        bool is_3d = false
    );
    
    /**
     * @brief 添加地下水模型
     * 
     * 在边坡模型中添加地下水层
     * 
     * @param particles 现有粒子列表
     * @param water_table_depth 地下水位深度
     * @param particle_radius 粒子半径
     * @return 包含地下水的粒子列表
     */
    std::vector<Particle> addGroundwaterModel(
        const std::vector<Particle>& particles,
        double water_table_depth,
        double particle_radius
    );
    
    /**
     * @brief 添加加固结构
     * 
     * 在边坡模型中添加加固结构（如锚杆、土钉等）
     * 
     * @param particles 现有粒子列表
     * @param reinforcement_params 加固结构参数
     * @param particle_radius 粒子半径
     * @return 包含加固结构的粒子列表
     */
    std::vector<Particle> addReinforcement(
        const std::vector<Particle>& particles,
        const Eigen::VectorXd& reinforcement_params,
        double particle_radius
    );
    
    /**
     * @brief 设置模型类型
     * 
     * @param model_type 模型类型
     */
    void setModelType(SlopeModelType model_type);
    
    /**
     * @brief 获取模型类型
     * 
     * @return 模型类型
     */
    SlopeModelType getModelType() const;
    
    /**
     * @brief 设置粒子生成器
     * 
     * @param particle_generator 粒子生成器
     */
    void setParticleGenerator(const ParticleGenerator& particle_generator);
    
private:
    // 检查点是否在边坡区域内
    /**
     * @brief 检查点是否在边坡区域内
     * 
     * 基于边坡几何参数，判断给定点是否在边坡模型范围内
     * 
     * @param point 三维点坐标
     * @param geometry_params 边坡几何参数
     * @param is_3d 是否为三维模型
     * @return 是否在边坡区域内
     */
    bool isPointInSlope(
        const Eigen::Vector3d& point,
        const SlopeGeometryParams& geometry_params,
        bool is_3d = false
    ) const;
    
    // 计算边坡的边界范围
    /**
     * @brief 计算边坡的边界范围
     * 
     * 根据几何参数计算边坡模型的最小和最大边界
     * 
     * @param geometry_params 边坡几何参数
     * @param is_3d 是否为三维模型
     * @return 边界范围（min, max）
     */
    std::pair<Eigen::Vector3d, Eigen::Vector3d> calculateSlopeBounds(
        const SlopeGeometryParams& geometry_params,
        bool is_3d = false
    ) const;
    
    // 生成三维网格点云
    /**
     * @brief 生成三维网格点云
     * 
     * 基于边界范围和粒子半径，生成均匀分布的三维点云
     * 
     * @param min_bound 最小边界
     * @param max_bound 最大边界
     * @param particle_radius 粒子半径
     * @return 生成的点云列表
     */
    std::vector<Eigen::Vector3d> generateGridPoints(
        const Eigen::Vector3d& min_bound,
        const Eigen::Vector3d& max_bound,
        double particle_radius
    ) const;
    
    SlopeModelType model_type_;              // 模型类型
    ParticleGenerator particle_generator_;   // 粒子生成器
};

} // namespace particle_simulation
