// 邻域搜索头文件
// 实现高效的空间划分算法，如网格法、八叉树法

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "particle/particle.h"

namespace particle_simulation {

// 邻域搜索算法类型
enum class NeighborSearchAlgorithm {
    GRID,       // 网格法
    OCTREE      // 八叉树法
};

// 邻域搜索基类
class NeighborSearch {
public:
    // 构造函数
    NeighborSearch(NeighborSearchAlgorithm algorithm = NeighborSearchAlgorithm::GRID);
    
    // 析构函数
    virtual ~NeighborSearch() = default;
    
    // 初始化搜索结构
    // 参数：
    // - particles: 粒子列表
    // - search_radius: 搜索半径
    virtual void initialize(
        const std::vector<Particle>& particles,
        double search_radius
    ) = 0;
    
    // 查找单个粒子的邻近粒子
    // 参数：
    // - particle: 目标粒子
    // 返回：邻近粒子的索引列表
    virtual std::vector<size_t> findNeighbors(const Particle& particle) const = 0;
    
    // 查找所有粒子的邻近粒子
    // 返回：每个粒子的邻近粒子索引列表
    virtual std::vector<std::vector<size_t>> findAllNeighbors() const = 0;
    
    // 更新搜索结构（当粒子位置变化时）
    virtual void update(const std::vector<Particle>& particles) = 0;
    
    // 设置搜索半径
    void setSearchRadius(double search_radius);
    
    // 获取搜索半径
    double getSearchRadius() const;
    
    // 设置搜索算法
    void setAlgorithm(NeighborSearchAlgorithm algorithm);
    
    // 获取搜索算法
    NeighborSearchAlgorithm getAlgorithm() const;
    
protected:
    NeighborSearchAlgorithm algorithm_;  // 搜索算法类型
    double search_radius_;              // 搜索半径
};

// 网格法邻域搜索
class GridNeighborSearch : public NeighborSearch {
public:
    // 构造函数
    GridNeighborSearch();
    explicit GridNeighborSearch(double search_radius);
    
    // 初始化搜索结构
    void initialize(
        const std::vector<Particle>& particles,
        double search_radius
    ) override;
    
    // 查找单个粒子的邻近粒子
    std::vector<size_t> findNeighbors(const Particle& particle) const override;
    
    // 查找所有粒子的邻近粒子
    std::vector<std::vector<size_t>> findAllNeighbors() const override;
    
    // 更新搜索结构
    void update(const std::vector<Particle>& particles) override;
    
private:
    // 网格单元格键类型
    using GridKey = Eigen::Vector3i;
    
    // 计算点所在的网格单元格键
    GridKey computeGridKey(const Eigen::Vector3d& point) const;
    
    // 获取邻近的网格单元格键
    std::vector<GridKey> getNeighborGridKeys(const GridKey& key) const;
    
    double cell_size_;                              // 网格单元格大小
    Eigen::Vector3d min_bound_;                     // 粒子系统最小边界
    Eigen::Vector3d max_bound_;                     // 粒子系统最大边界
    std::map<GridKey, std::vector<size_t>> grid_;   // 网格映射，键为网格键，值为粒子索引列表
    const std::vector<Particle>* particles_;        // 粒子列表指针
};

// 八叉树法邻域搜索（预留接口，后续实现）
class OctreeNeighborSearch : public NeighborSearch {
public:
    // 构造函数
    OctreeNeighborSearch();
    explicit OctreeNeighborSearch(double search_radius);
    
    // 初始化搜索结构
    void initialize(
        const std::vector<Particle>& particles,
        double search_radius
    ) override;
    
    // 查找单个粒子的邻近粒子
    std::vector<size_t> findNeighbors(const Particle& particle) const override;
    
    // 查找所有粒子的邻近粒子
    std::vector<std::vector<size_t>> findAllNeighbors() const override;
    
    // 更新搜索结构
    void update(const std::vector<Particle>& particles) override;
    
private:
    // 八叉树节点类
    class OctreeNode {
    public:
        OctreeNode(const Eigen::AlignedBox3d& bounds, int depth = 0);
        ~OctreeNode();
        
        // 插入粒子
        void insert(size_t particle_index, const Particle& particle);
        
        // 查找邻近粒子
        void findNeighbors(
            const Particle& particle,
            double search_radius,
            std::vector<size_t>& neighbors
        ) const;
        
        // 是否为叶子节点
        bool isLeaf() const;
        
        // 获取子节点
        const std::vector<OctreeNode*>& getChildren() const;
        
        // 获取粒子索引列表
        const std::vector<size_t>& getParticleIndices() const;
        
        // 获取节点边界
        const Eigen::AlignedBox3d& getBounds() const;
        
private:
        Eigen::AlignedBox3d bounds_;              // 节点边界
        int depth_;                               // 节点深度
        std::vector<size_t> particle_indices_;     // 粒子索引列表
        std::vector<OctreeNode*> children_;        // 子节点列表
        bool is_leaf_;                            // 是否为叶子节点
        const int max_depth_ = 10;                 // 最大深度
        const int max_particles_per_node_ = 10;    // 每个节点最大粒子数
    };
    
    OctreeNode* root_;                              // 八叉树根节点
    const std::vector<Particle>* particles_;        // 粒子列表指针
};

// 邻域搜索工厂类，用于创建邻域搜索实例
class NeighborSearchFactory {
public:
    // 创建邻域搜索实例
    // 参数：
    // - algorithm: 搜索算法类型
    // - search_radius: 搜索半径
    // 返回：邻域搜索实例指针
    static std::unique_ptr<NeighborSearch> createNeighborSearch(
        NeighborSearchAlgorithm algorithm = NeighborSearchAlgorithm::GRID,
        double search_radius = 0.0
    );
};

} // namespace particle_simulation
