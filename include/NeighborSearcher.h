// NeighborSearcher.h
// 邻域搜索算法模块
// 实现高效的空间划分算法，如网格法、树状法

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "Particle.h"

namespace particle_simulation {

// 邻域搜索算法类型
enum class SearchAlgorithm {
    GRID,       // 网格法
    OCTREE      // 八叉树法
};

// 邻域搜索器类
class NeighborSearcher {
public:
    // 构造函数
    NeighborSearcher(SearchAlgorithm algorithm = SearchAlgorithm::GRID);
    
    // 初始化搜索器
    // 参数：
    // - particles: 粒子列表
    // - search_radius: 搜索半径
    void initialize(
        const std::vector<Particle>& particles,
        double search_radius
    );
    
    // 查找单个粒子的邻近粒子
    // 参数：
    // - particle: 目标粒子
    // 返回：邻近粒子的索引列表
    std::vector<size_t> findNeighbors(const Particle& particle) const;
    
    // 查找所有粒子的邻近粒子
    // 返回：每个粒子的邻近粒子索引列表
    std::vector<std::vector<size_t>> findAllNeighbors() const;
    
    // 更新搜索结构（当粒子位置变化时）
    void update(const std::vector<Particle>& particles);
    
    // 设置搜索算法
    void setAlgorithm(SearchAlgorithm algorithm);
    
    // 设置搜索半径
    void setSearchRadius(double search_radius);
    
    // 获取搜索半径
    double getSearchRadius() const;
    
private:
    // 网格法实现
    class GridSearch {
    public:
        GridSearch(double search_radius);
        
        void initialize(const std::vector<Particle>& particles);
        void update(const std::vector<Particle>& particles);
        std::vector<size_t> findNeighbors(const Particle& particle) const;
        std::vector<std::vector<size_t>> findAllNeighbors() const;
        
    private:
        // 网格单元格键类型
        using GridKey = Eigen::Vector3i;
        
        // 计算点所在的网格单元格键
        GridKey getGridKey(const Eigen::Vector3d& point) const;
        
        double search_radius_;
        double cell_size_;
        std::vector<Particle> particles_;
        std::map<GridKey, std::vector<size_t>> grid_;
        Eigen::Vector3d min_bound_;
        Eigen::Vector3d max_bound_;
    };
    
    // 八叉树法实现（预留接口，后续扩展）
    class OctreeSearch {
    public:
        OctreeSearch(double search_radius);
        
        void initialize(const std::vector<Particle>& particles);
        void update(const std::vector<Particle>& particles);
        std::vector<size_t> findNeighbors(const Particle& particle) const;
        std::vector<std::vector<size_t>> findAllNeighbors() const;
        
    private:
        double search_radius_;
        std::vector<Particle> particles_;
    };
    
    SearchAlgorithm algorithm_;
    double search_radius_;
    std::vector<Particle> particles_;
    
    // 算法实例
    std::unique_ptr<GridSearch> grid_search_;
    std::unique_ptr<OctreeSearch> octree_search_;
};

} // namespace particle_simulation
