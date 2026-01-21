// NeighborSearcher.cpp
// 邻域搜索算法实现
// 主要实现网格法的邻域搜索功能

#include "NeighborSearcher.h"
#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>

namespace particle_simulation {

// NeighborSearcher 构造函数
NeighborSearcher::NeighborSearcher(SearchAlgorithm algorithm)
    : algorithm_(algorithm),
      search_radius_(0.0) {
}

// 初始化搜索器
void NeighborSearcher::initialize(
    const std::vector<Particle>& particles,
    double search_radius
) {
    particles_ = particles;
    search_radius_ = search_radius;
    
    switch (algorithm_) {
        case SearchAlgorithm::GRID:
            grid_search_ = std::make_unique<GridSearch>(search_radius);
            grid_search_->initialize(particles);
            break;
        case SearchAlgorithm::OCTREE:
            octree_search_ = std::make_unique<OctreeSearch>(search_radius);
            octree_search_->initialize(particles);
            break;
        default:
            std::cerr << "Unknown search algorithm!" << std::endl;
            break;
    }
}

// 查找单个粒子的邻近粒子
std::vector<size_t> NeighborSearcher::findNeighbors(const Particle& particle) const {
    switch (algorithm_) {
        case SearchAlgorithm::GRID:
            return grid_search_->findNeighbors(particle);
        case SearchAlgorithm::OCTREE:
            return octree_search_->findNeighbors(particle);
        default:
            return {};
    }
}

// 查找所有粒子的邻近粒子
std::vector<std::vector<size_t>> NeighborSearcher::findAllNeighbors() const {
    switch (algorithm_) {
        case SearchAlgorithm::GRID:
            return grid_search_->findAllNeighbors();
        case SearchAlgorithm::OCTREE:
            return octree_search_->findAllNeighbors();
        default:
            return {};
    }
}

// 更新搜索结构
void NeighborSearcher::update(const std::vector<Particle>& particles) {
    particles_ = particles;
    
    switch (algorithm_) {
        case SearchAlgorithm::GRID:
            grid_search_->update(particles);
            break;
        case SearchAlgorithm::OCTREE:
            octree_search_->update(particles);
            break;
        default:
            break;
    }
}

// 设置搜索算法
void NeighborSearcher::setAlgorithm(SearchAlgorithm algorithm) {
    algorithm_ = algorithm;
    
    // 根据新算法重新初始化
    if (search_radius_ > 0.0 && !particles_.empty()) {
        initialize(particles_, search_radius_);
    }
}

// 设置搜索半径
void NeighborSearcher::setSearchRadius(double search_radius) {
    search_radius_ = search_radius;
    
    // 根据新搜索半径重新初始化
    if (!particles_.empty()) {
        initialize(particles_, search_radius);
    }
}

// 获取搜索半径
double NeighborSearcher::getSearchRadius() const {
    return search_radius_;
}

// ----------------------------------------
// GridSearch 实现
// ----------------------------------------

// GridSearch 构造函数
NeighborSearcher::GridSearch::GridSearch(double search_radius)
    : search_radius_(search_radius),
      cell_size_(search_radius) {
}

// 初始化网格搜索
void NeighborSearcher::GridSearch::initialize(const std::vector<Particle>& particles) {
    particles_ = particles;
    grid_.clear();
    
    // 计算粒子边界
    min_bound_ = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    max_bound_ = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
    
    for (const auto& particle : particles_) {
        const Eigen::Vector3d& pos = particle.getPosition();
        min_bound_ = min_bound_.cwiseMin(pos);
        max_bound_ = max_bound_.cwiseMax(pos);
    }
    
    // 扩展边界以包含所有粒子
    min_bound_ -= Eigen::Vector3d::Constant(search_radius_);
    max_bound_ += Eigen::Vector3d::Constant(search_radius_);
    
    // 将粒子分配到网格单元格
    for (size_t i = 0; i < particles_.size(); ++i) {
        const auto& particle = particles_[i];
        GridKey key = getGridKey(particle.getPosition());
        grid_[key].push_back(i);
    }
    
    std::cout << "Grid initialized with " << grid_.size() << " cells." << std::endl;
}

// 更新网格搜索
void NeighborSearcher::GridSearch::update(const std::vector<Particle>& particles) {
    initialize(particles); // 简单实现，重新初始化整个网格
}

// 查找单个粒子的邻近粒子
std::vector<size_t> NeighborSearcher::GridSearch::findNeighbors(const Particle& particle) const {
    std::vector<size_t> neighbors;
    const Eigen::Vector3d& pos = particle.getPosition();
    
    // 获取当前粒子所在的网格单元格
    GridKey center_key = getGridKey(pos);
    
    // 检查当前单元格及其相邻的26个单元格
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                GridKey neighbor_key = center_key + GridKey(dx, dy, dz);
                
                // 查找该单元格中的粒子
                auto it = grid_.find(neighbor_key);
                if (it != grid_.end()) {
                    // 检查单元格中的每个粒子是否在搜索半径内
                    for (size_t particle_idx : it->second) {
                        const auto& neighbor_particle = particles_[particle_idx];
                        double distance = (neighbor_particle.getPosition() - pos).norm();
                        
                        if (distance <= search_radius_ && distance > 1e-6) {
                            neighbors.push_back(particle_idx);
                        }
                    }
                }
            }
        }
    }
    
    return neighbors;
}

// 查找所有粒子的邻近粒子
std::vector<std::vector<size_t>> NeighborSearcher::GridSearch::findAllNeighbors() const {
    std::vector<std::vector<size_t>> all_neighbors;
    all_neighbors.reserve(particles_.size());
    
    for (const auto& particle : particles_) {
        all_neighbors.push_back(findNeighbors(particle));
    }
    
    return all_neighbors;
}

// 计算点所在的网格单元格键
NeighborSearcher::GridSearch::GridKey NeighborSearcher::GridSearch::getGridKey(const Eigen::Vector3d& point) const {
    GridKey key;
    key.x() = static_cast<int>(floor((point.x() - min_bound_.x()) / cell_size_));
    key.y() = static_cast<int>(floor((point.y() - min_bound_.y()) / cell_size_));
    key.z() = static_cast<int>(floor((point.z() - min_bound_.z()) / cell_size_));
    return key;
}

// ----------------------------------------
// OctreeSearch 实现（预留）
// ----------------------------------------

// OctreeSearch 构造函数
NeighborSearcher::OctreeSearch::OctreeSearch(double search_radius)
    : search_radius_(search_radius) {
}

// 初始化八叉树搜索
void NeighborSearcher::OctreeSearch::initialize(const std::vector<Particle>& particles) {
    particles_ = particles;
    std::cout << "Octree initialized." << std::endl;
}

// 更新八叉树搜索
void NeighborSearcher::OctreeSearch::update(const std::vector<Particle>& particles) {
    initialize(particles);
}

// 查找单个粒子的邻近粒子
std::vector<size_t> NeighborSearcher::OctreeSearch::findNeighbors(const Particle& particle) const {
    std::vector<size_t> neighbors;
    const Eigen::Vector3d& pos = particle.getPosition();
    
    // 简单实现：线性搜索（后续替换为八叉树搜索）
    for (size_t i = 0; i < particles_.size(); ++i) {
        const auto& neighbor_particle = particles_[i];
        double distance = (neighbor_particle.getPosition() - pos).norm();
        
        if (distance <= search_radius_ && distance > 1e-6) {
            neighbors.push_back(i);
        }
    }
    
    return neighbors;
}

// 查找所有粒子的邻近粒子
std::vector<std::vector<size_t>> NeighborSearcher::OctreeSearch::findAllNeighbors() const {
    std::vector<std::vector<size_t>> all_neighbors;
    all_neighbors.reserve(particles_.size());
    
    for (const auto& particle : particles_) {
        all_neighbors.push_back(findNeighbors(particle));
    }
    
    return all_neighbors;
}

} // namespace particle_simulation
