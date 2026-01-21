// main.cpp
// 粒子法仿真软件主程序
// 演示如何使用各个模块进行岩土边坡稳定性评估

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "include/Particle.h"
#include "include/ParticleGenerator.h"
#include "include/NeighborSearcher.h"
#include "include/ConstitutiveModel.h"
#include "include/ContactDetection.h"
#include "include/TimeIntegrator.h"
#include "include/SlopeModelBuilder.h"
#include "include/SimulationController.h"
#include "include/StabilityEvaluator.h"
#include "include/VisualizationModule.h"

using namespace particle_simulation;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        粒子法岩土边坡仿真软件           " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. 设置仿真参数
        double particle_radius = 0.2;    // 粒子半径
        double time_step = 0.001;        // 时间步长
        size_t max_time_steps = 1000;    // 最大时间步数
        
        // 2. 构建边坡模型
        std::cout << "1. 构建边坡模型..." << std::endl;
        
        // 创建边坡几何参数
        SlopeGeometryParams geometry_params;
        geometry_params.slope_height = 10.0;    // 边坡高度10m
        geometry_params.slope_angle = 45.0;     // 边坡坡角45度
        geometry_params.slope_width = 20.0;     // 边坡宽度20m
        geometry_params.ground_depth = 5.0;     // 地基深度5m
        
        // 创建岩土体参数
        GeomaterialParams geomaterial_params;
        geomaterial_params.density = 2600.0;                 // 密度2600kg/m³
        geomaterial_params.young_modulus = 1.0e6;           // 弹性模量1e6Pa
        geomaterial_params.poisson_ratio = 0.3;             // 泊松比0.3
        geomaterial_params.cohesion = 10000.0;              // 黏聚力10000Pa
        geomaterial_params.friction_angle = 30.0;           // 内摩擦角30度
        geomaterial_params.moisture_content = 10.0;         // 含水率10%
        
        // 创建边坡模型构建器
        SlopeModelBuilder slope_builder;
        
        // 构建参数化边坡模型
        std::vector<Particle> particles = slope_builder.buildParametricSlope(
            geometry_params,
            geomaterial_params,
            particle_radius,
            false  // 二维模型
        );
        
        // 保存初始粒子状态
        std::vector<Particle> initial_particles = particles;
        
        std::cout << "边坡模型构建完成，粒子数量: " << particles.size() << std::endl;
        std::cout << std::endl;
        
        // 3. 初始化核心模块
        std::cout << "2. 初始化核心模块..." << std::endl;
        
        // 邻域搜索器
        double search_radius = particle_radius * 2.0;  // 搜索半径为2倍粒子半径
        NeighborSearcher neighbor_searcher(SearchAlgorithm::GRID);
        neighbor_searcher.initialize(particles, search_radius);
        
        // 本构模型（Mohr-Coulomb模型）
        auto constitutive_model = ConstitutiveModelFactory::createModel(
            ConstitutiveModelType::MOHR_COULOMB
        );
        
        // 接触检测模块
        ContactDetection contact_detection(CollisionResponseModel::SPRING_DAMPER);
        contact_detection.initialize(particles, neighbor_searcher);
        
        // 时间积分器
        TimeIntegrator time_integrator(IntegratorType::EXPLICIT_EULER);
        time_integrator.initialize(particles, *constitutive_model, contact_detection);
        time_integrator.setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));  // 设置重力
        
        std::cout << "核心模块初始化完成" << std::endl;
        std::cout << std::endl;
        
        // 4. 初始化仿真控制器
        std::cout << "3. 初始化仿真控制器..." << std::endl;
        
        SimulationController sim_controller;
        
        // 设置仿真参数
        TerminationConditions termination_conditions;
        termination_conditions.max_time_steps = max_time_steps;
        termination_conditions.max_simulation_time = max_time_steps * time_step;
        
        sim_controller.setSimulationParameters(
            time_step,
            termination_conditions,
            100,  // 保存间隔
            10    // 统计信息更新间隔
        );
        
        // 初始化仿真
        sim_controller.initialize(
            particles,
            neighbor_searcher,
            *constitutive_model,
            contact_detection,
            time_integrator
        );
        
        std::cout << "仿真控制器初始化完成" << std::endl;
        std::cout << std::endl;
        
        // 5. 初始化可视化模块
        std::cout << "4. 初始化可视化模块..." << std::endl;
        
        auto visualization_module = VisualizationModuleFactory::createVisualizationModule(
            RenderingMode::STATIC
        );
        
        VisualizationParams vis_params;
        visualization_module->initialize(800, 600, vis_params);
        
        std::cout << "可视化模块初始化完成" << std::endl;
        std::cout << std::endl;
        
        // 6. 运行仿真
        std::cout << "5. 运行仿真..." << std::endl;
        
        // 注册仿真进度回调
        sim_controller.setProgressCallback([&](const SimulationStats& stats) {
            std::cout << "\rTime Step: " << stats.current_time_step << 
                ", Simulation Time: " << stats.current_simulation_time << "s" << 
                ", Max Displacement: " << stats.max_displacement << "m" << 
                ", Penetration Count: " << stats.penetration_count << 
                std::flush;
            
            // 每100步更新一次可视化
            if (stats.current_time_step % 100 == 0) {
                std::vector<Particle> current_particles = sim_controller.getParticles();
                visualization_module->updateVisualization(
                    current_particles,
                    stats.current_simulation_time,
                    stats.current_time_step
                );
            }
        });
        
        // 启动仿真
        sim_controller.start();
        
        // 等待仿真完成
        while (sim_controller.getSimulationState() == SimulationState::RUNNING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n" << std::endl;
        std::cout << "仿真运行完成" << std::endl;
        std::cout << std::endl;
        
        // 7. 评估边坡稳定性
        std::cout << "6. 评估边坡稳定性..." << std::endl;
        
        // 获取最终粒子状态
        std::vector<Particle> final_particles = sim_controller.getParticles();
        
        // 创建稳定性评估器
        StabilityEvaluator stability_evaluator(SafetyFactorMethod::SIMPLIFIED_BISHOP);
        
        // 评估稳定性
        StabilityIndices indices = stability_evaluator.evaluateStability(
            final_particles,
            initial_particles
        );
        
        // 生成稳定性评估报告
        stability_evaluator.generateStabilityReport(
            indices,
            "./slope_stability_report.txt"
        );
        
        // 可视化稳定性评估结果
        visualization_module->visualizeStabilityResults(
            indices,
            final_particles,
            initial_particles
        );
        
        std::cout << "边坡稳定性评估完成" << std::endl;
        std::cout << "安全系数: " << indices.safety_factor << std::endl;
        std::cout << "失稳预警等级: " << indices.instability_warning_level << " (0-100)" << std::endl;
        std::cout << "最大位移: " << indices.max_displacement << "m" << std::endl;
        std::cout << std::endl;
        
        // 8. 保存仿真结果
        std::cout << "7. 保存仿真结果..." << std::endl;
        
        // 保存最终粒子位置
        std::ofstream pos_file("./final_positions.txt");
        if (pos_file.is_open()) {
            for (const auto& particle : final_particles) {
                const Eigen::Vector3d& pos = particle.getPosition();
                pos_file << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
            }
            pos_file.close();
            std::cout << "最终粒子位置已保存到: ./final_positions.txt" << std::endl;
        }
        
        std::cout << "仿真结果保存完成" << std::endl;
        std::cout << std::endl;
        
        // 9. 关闭仿真
        std::cout << "8. 关闭仿真环境..." << std::endl;
        
        // 关闭可视化模块
        visualization_module->shutdown();
        
        std::cout << "仿真环境关闭完成" << std::endl;
        std::cout << std::endl;
        
        std::cout << "========================================" << std::endl;
        std::cout << "        仿真程序执行完成               " << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "仿真过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
