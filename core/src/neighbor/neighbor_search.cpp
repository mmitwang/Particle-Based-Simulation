// 邻域搜索源文件
// 实现网格法和八叉树法邻域搜索

#include "neighbor/neighbor_search.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>

namespace particle_simulation {

// ----------------------------------------
// NeighborSearch 基类实现
// ----------------------------------------

// 构造函数
NeighborSearch::NeighborSearch(NeighborSearchAlgorithm algorithm)
    : algorithm_(algorithm),
      search_radius_(0.0) {
}

// 设置搜索半径
void NeighborSearch::setSearchRadius(double search_radius) {
    search_radius_ = search_radius;
}

// 获取搜索半径
double NeighborSearch::getSearchRadius() const {
    return search_radius_;
}

// 设置搜索算法
void NeighborSearch::setAlgorithm(NeighborSearchAlgorithm algorithm) {
    algorithm_ = algorithm;
}

// 获取搜索算法
NeighborSearchAlgorithm NeighborSearch::getAlgorithm() const {
    return algorithm_;
}

// ----------------------------------------
// GridNeighborSearch 实现
// ----------------------------------------

// 构造函数
GridNeighborSearch::GridNeighborSearch()
    : NeighborSearch(NeighborSearchAlgorithm::GRID),
      cell_size_(0.0),
      particles_(nullptr) {
}

GridNeighborSearch::GridNeighborSearch(double search_radius)
    : NeighborSearch(NeighborSearchAlgorithm::GRID),
      cell_size_(search_radius),
      particles_(nullptr) {
    setSearchRadius(search_radius);
}

// 初始化搜索结构
void GridNeighborSearch::initialize(
    const std::vector<Particle>& particles,
    double search_radius
) {
    setSearchRadius(search_radius);
    cell_size_ = search_radius;
    particles_ = &particles;
    
    // 清空网格
    grid_.clear();
    
    // 计算粒子边界
    Eigen::AlignedBox3d bounds;
    for (const auto& particle : particles) {
        bounds.extend(particle.getPosition());
    }
    
    // 扩展边界以包含所有粒子
    min_bound_ = bounds.min();
    max_bound_ = bounds.max();
    
    // 扩展边界，确保所有粒子都在网格内
    Eigen::Vector3d extension = Eigen::Vector3d::Constant(search_radius);
    min_bound_ -= extension;
    max_bound_ += extension;
    
    // 将粒子分配到网格单元格
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& particle = particles[i];
        GridKey key = computeGridKey(particle.getPosition());
        grid_[key].push_back(i);
    }
    
    std::cout << "Grid initialized with " << grid_.size() << " cells." << std::endl;
}

// 查找单个粒子的邻近粒子
std::vector<size_t> GridNeighborSearch::findNeighbors(const Particle& particle) const {
    std::vector<size_t> neighbors;
    const Eigen::Vector3d& pos = particle.getPosition();
    
    // 获取当前粒子所在的网格单元格
    GridKey center_key = computeGridKey(pos);
    
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
                        const Particle& neighbor_particle = (*particles_)[particle_idx];
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
std::vector<std::vector<size_t>> GridNeighborSearch::findAllNeighbors() const {
    std::vector<std::vector<size_t>> all_neighbors;
    all_neighbors.reserve(particles_->size());
    
    for (const auto& particle : *particles_) {
        all_neighbors.push_back(findNeighbors(particle));
    }
    
    return all_neighbors;
}

// 更新搜索结构
void GridNeighborSearch::update(const std::vector<Particle>& particles) {
    // 重新初始化网格
    initialize(particles, search_radius_);
}

// 计算点所在的网格单元格键
GridNeighborSearch::GridKey GridNeighborSearch::computeGridKey(const Eigen::Vector3d& point) const {
    GridKey key;
    key.x() = static_cast<int>(floor((point.x() - min_bound_.x()) / cell_size_));
    key.y() = static_cast<int>(floor((point.y() - min_bound_.y()) / cell_size_));
    key.z() = static_cast<int>(floor((point.z() - min_bound_.z()) / cell_size_));
    return key;
}

// 获取邻近的网格单元格键
std::vector<GridNeighborSearch::GridKey> GridNeighborSearch::getNeighborGridKeys(const GridKey& key) const {
    std::vector<GridKey> neighbor_keys;
    neighbor_keys.reserve(27);
    
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                neighbor_keys.push_back(key + GridKey(dx, dy, dz));
            }
        }
    }
    
    return neighbor_keys;
}

// ----------------------------------------
// OctreeNeighborSearch 实现
// ----------------------------------------

// 构造函数
OctreeNeighborSearch::OctreeNeighborSearch()
    : NeighborSearch(NeighborSearchAlgorithm::OCTREE),
      root_(nullptr),
      particles_(nullptr) {
}

OctreeNeighborSearch::OctreeNeighborSearch(double search_radius)
    : NeighborSearch(NeighborSearchAlgorithm::OCTREE),
      root_(nullptr),
      particles_(nullptr) {
    setSearchRadius(search_radius);
}

// 初始化搜索结构
void OctreeNeighborSearch::initialize(
    const std::vector<Particle>& particles,
    double search_radius
) {
    setSearchRadius(search_radius);
    particles_ = &particles;
    
    // 计算粒子边界
    Eigen::AlignedBox3d bounds;
    for (const auto& particle : particles) {
        bounds.extend(particle.getPosition());
    }
    
    // 扩展边界
    Eigen::Vector3d extension = Eigen::Vector3d::Constant(search_radius);
    bounds.extend(bounds.min() - extension);
    bounds.extend(bounds.max() + extension);
    
    // 创建八叉树根节点
    delete root_;
    root_ = new OctreeNode(bounds);
    
    // 插入所有粒子
    for (size_t i = 0; i < particles.size(); ++i) {
        root_->insert(i, particles[i]);
    }
    
    std::cout << "Octree initialized." << std::endl;
}

// 查找单个粒子的邻近粒子
std::vector<size_t> OctreeNeighborSearch::findNeighbors(const Particle& particle) const {
    std::vector<size_t> neighbors;
    if (root_) {
        root_->findNeighbors(particle, search_radius_, neighbors);
    }
    return neighbors;
}

// 查找所有粒子的邻近粒子
std::vector<std::vector<size_t>> OctreeNeighborSearch::findAllNeighbors() const {
    std::vector<std::vector<size_t>> all_neighbors;
    all_neighbors.reserve(particles_->size());
    
    for (const auto& particle : *particles_) {
        all_neighbors.push_back(findNeighbors(particle));
    }
    
    return all_neighbors;
}

// 更新搜索结构
void OctreeNeighborSearch::update(const std::vector<Particle>& particles) {
    // 重新初始化八叉树
    initialize(particles, search_radius_);
}

// ----------------------------------------
// OctreeNode 实现
// ----------------------------------------

// 构造函数
OctreeNeighborSearch::OctreeNode::OctreeNode(const Eigen::AlignedBox3d& bounds, int depth)
    : bounds_(bounds),
      depth_(depth),
      is_leaf_(true) {
}

// 析构函数
OctreeNeighborSearch::OctreeNode::~OctreeNode() {
    // 递归删除子节点
    for (auto child : children_) {
        delete child;
    }
}

// 插入粒子
void OctreeNeighborSearch::OctreeNode::insert(size_t particle_index, const Particle& particle) {
    if (is_leaf_) {
        // 如果是叶子节点，直接插入
        particle_indices_.push_back(particle_index);
        
        // 如果粒子数超过阈值且深度未达到最大值，分裂节点
        if (particle_indices_.size() > max_particles_per_node_ && depth_ < max_depth_) {
            // 分裂节点
            split();
        }
    } else {
        // 如果不是叶子节点，找到合适的子节点插入
        for (auto child : children_) {
            if (child->getBounds().contains(particle.getPosition())) {
                child->insert(particle_index, particle);
                return;
            }
        }
        
        // 如果没有合适的子节点，插入到当前节点
        particle_indices_.push_back(particle_index);
    }
}

// 查找邻近粒子
void OctreeNeighborSearch::OctreeNode::findNeighbors(
    const Particle& particle,
    double search_radius,
    std::vector<size_t>& neighbors
) const {
    // 检查当前节点是否与搜索球相交
    Eigen::Vector3d center = particle.getPosition();
    if (!bounds_.intersects(Eigen::AlignedBox3d(
        center - Eigen::Vector3d::Constant(search_radius),
        center + Eigen::Vector3d::Constant(search_radius)
    ))) {
        return;
    }
    
    // 检查当前节点的粒子
    for (size_t index : particle_indices_) {
        neighbors.push_back(index);
    }
    
    // 递归检查子节点
    if (!is_leaf_) {
        for (auto child : children_) {
            child->findNeighbors(particle, search_radius, neighbors);
        }
    }
}

// 是否为叶子节点
bool OctreeNeighborSearch::OctreeNode::isLeaf() const {
    return is_leaf_;
}

// 获取子节点
const std::vector<OctreeNeighborSearch::OctreeNode*>& OctreeNeighborSearch::OctreeNode::getChildren() const {
    return children_;
}

// 获取粒子索引列表
const std::vector<size_t>& OctreeNeighborSearch::OctreeNode::getParticleIndices() const {
    return particle_indices_;
}

// 获取节点边界
const Eigen::AlignedBox3d& OctreeNeighborSearch::OctreeNode::getBounds() const {
    return bounds_;
}

// 分裂节点（私有方法）
void OctreeNeighborSearch::OctreeNode::split() {
    // 创建8个子节点
    Eigen::Vector3d center = bounds_.center();
    Eigen::Vector3d extents = bounds_.sizes() / 2.0;
    
    // 子节点边界
    std::vector<Eigen::AlignedBox3d> child_bounds(8);
    child_bounds[0] = Eigen::AlignedBox3d(
        center - extents,
        center
    );
    child_bounds[1] = Eigen::AlignedBox3d(
        Eigen::Vector3d(center.x(), center.y(), center.z() - extents.z()),
        Eigen::Vector3d(center.x() + extents.x(), center.y() + extents.y(), center.z())
    );
    child_bounds[2] = Eigen::AlignedBox3d(
        Eigen::Vector3d(center.x() - extents.x(), center.y(), center.z()),
        Eigen::Vector3d(center.x(), center.y() + extents.y(), center.z() + extents.z())
    );
    child_bounds[3] = Eigen::AlignedBox3d(
        center,
        center + extents
    );
    child_bounds[4] = Eigen::AlignedBox3d(
        Eigen::Vector3d(center.x() - extents.x(), center.y() - extents.y(), center.z()),
        Eigen::Vector3d(center.x(), center.y(), center.z() + extents.z())
    );
    child_bounds[5] = Eigen::AlignedBox3d(
        Eigen::Vector3d(center.x(), center.y() - extents.y(), center.z() - extents.z()),
        Eigen::Vector3d(center.x() + extents.x(), center.y(), center.z())
    );
    child_bounds[6] = Eigen::AlignedBox3d(
        bounds_.min(),
        center
    );
    child_bounds[7] = Eigen::AlignedBox3d(
        Eigen::Vector3d(center.x(), bounds_.min().y(), bounds_.min().z()),
        Eigen::Vector3d(bounds_.max().x(), center.y(), center.z())
    );
    
    // 创建子节点
    children_.resize(8);
    for (int i = 0; i < 8; ++i) {
        children_[i] = new OctreeNode(child_bounds[i], depth_ + 1);
    }
    
    // 将当前节点的粒子分配到子节点
    for (size_t index : particle_indices_) {
        // 这里需要访问粒子列表，暂时无法实现，留待后续完善
        // 目前简单实现，将粒子保留在当前节点
    }
    
    // 标记为非叶子节点
    is_leaf_ = false;
}

// ----------------------------------------
// NeighborSearchFactory 实现
// ----------------------------------------

// 创建邻域搜索实例
std::unique_ptr<NeighborSearch> NeighborSearchFactory::createNeighborSearch(
    NeighborSearchAlgorithm algorithm,
    double search_radius
) {
    std::unique_ptr<NeighborSearch> search;
    
    switch (algorithm) {
        case NeighborSearchAlgorithm::GRID:
            search = std::make_unique<GridNeighborSearch>(search_radius);
            break;
        case NeighborSearchAlgorithm::OCTREE:
            search = std::make_unique<OctreeNeighborSearch>(search_radius);
            break;
        default:
            search = std::make_unique<GridNeighborSearch>(search_radius);
            break;
    }
    
    return search;
}

} // namespace particle_simulation
