// 边坡模型构建头文件
// 实现边坡模型的参数化构建和外部模型导入

#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "particle/particle.h"

namespace particle_simulation {

// 边坡模型类型枚举
enum class SlopeModelType {
    PARAMETRIC,      // 参数化模型
    EXTERNAL_IMPORT  // 外部导入模型
};

// 边坡几何参数结构体
struct SlopeGeometryParams {
    double slope_height;      // 边坡高度 (m)
    double slope_angle;       // 边坡坡角 (度)
    double slope_width;       // 边坡宽度 (m)
    double slope_length;      // 边坡长度 (m，三维模型使用)
    double ground_depth;      // 地基深度 (m)
    double berm_width;        // 平台宽度 (m，多级边坡使用)
    double berm_height;       // 平台高度 (m，多级边坡使用)
    
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
struct GeomaterialParams {
    double density;          // 密度 (kg/m³)
    double young_modulus;    // 弹性模量 (Pa)
    double poisson_ratio;    // 泊松比
    double cohesion;         // 黏聚力 (Pa)
    double friction_angle;   // 内摩擦角 (度)
    double dilation_angle;   // 剪胀角 (度)
    double permeability;     // 渗透率 (m/s)
    double moisture_content; // 含水率 (%)
    
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
     * @brief 导出边坡模型
     * 
     * 将边坡粒子模型导出为外部文件格式
     * 
     * @param particles 粒子列表
     * @param file_path 输出文件路径
     * @param file_format 文件格式
     * @return 是否导出成功
     */
    bool exportSlopeModel(
        const std::vector<Particle>& particles,
        const std::string& file_path,
        const std::string& file_format = "stl"
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
    
private:
    // 检查点是否在边坡区域内
    bool isPointInSlope(
        const Eigen::Vector3d& point,
        const SlopeGeometryParams& geometry_params,
        bool is_3d = false
    ) const;
    
    // 计算边坡的边界范围
    std::pair<Eigen::Vector3d, Eigen::Vector3d> calculateSlopeBounds(
        const SlopeGeometryParams& geometry_params,
        bool is_3d = false
    ) const;
    
    // 生成三维网格点云
    std::vector<Eigen::Vector3d> generateGridPoints(
        const Eigen::Vector3d& min_bound,
        const Eigen::Vector3d& max_bound,
        double particle_radius
    ) const;
    
    SlopeModelType model_type_;              // 模型类型
};

// 边坡模型编辑器类，提供模型预览、几何修正、局部细化等编辑功能
class SlopeModelEditor {
public:
    // 构造函数
    SlopeModelEditor();
    
    /**
     * @brief 预览边坡模型
     * 
     * 生成边坡模型的简化预览
     * 
     * @param particles 粒子列表
     * @param resolution 预览分辨率
     * @return 预览模型的粒子列表
     */
    std::vector<Particle> previewModel(
        const std::vector<Particle>& particles,
        double resolution = 0.5
    );
    
    /**
     * @brief 修正边坡几何形状
     * 
     * 对边坡模型进行几何修正，确保模型的合理性
     * 
     * @param particles 粒子列表
     * @param tolerance 修正容差
     * @return 修正后的粒子列表
     */
    std::vector<Particle> correctGeometry(
        const std::vector<Particle>& particles,
        double tolerance = 0.01
    );
    
    /**
     * @brief 局部细化边坡模型
     * 
     * 对边坡模型的特定区域进行局部细化
     * 
     * @param particles 粒子列表
     * @param refinement_region 细化区域（边界框）
     * @param new_particle_radius 细化后的粒子半径
     * @return 细化后的粒子列表
     */
    std::vector<Particle> refineLocalRegion(
        const std::vector<Particle>& particles,
        const Eigen::AlignedBox3d& refinement_region,
        double new_particle_radius
    );
    
    /**
     * @brief 合并多个粒子模型
     * 
     * 将多个粒子模型合并为一个模型
     * 
     * @param particle_lists 粒子列表列表
     * @return 合并后的粒子列表
     */
    std::vector<Particle> mergeModels(
        const std::vector<std::vector<Particle>>& particle_lists
    );
    
    /**
     * @brief 裁剪粒子模型
     * 
     * 根据给定的边界框裁剪粒子模型
     * 
     * @param particles 粒子列表
     * @param clip_box 裁剪边界框
     * @return 裁剪后的粒子列表
     */
    std::vector<Particle> clipModel(
        const std::vector<Particle>& particles,
        const Eigen::AlignedBox3d& clip_box
    );
    
private:
    // 计算模型的凸包
    std::vector<Particle> computeConvexHull(const std::vector<Particle>& particles) const;
    
    // 移除重复粒子
    std::vector<Particle> removeDuplicateParticles(
        const std::vector<Particle>& particles,
        double tolerance = 1e-6
    ) const;
};

} // namespace particle_simulation
