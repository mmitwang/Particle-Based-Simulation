# CMakeLists.txt
# 粒子法岩土边坡仿真软件构建配置

cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(ParticleSlopeSimulation VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# 查找依赖库
# Eigen库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
)

# 源文件列表
file(GLOB_RECURSE CORE_SOURCES ${PROJECT_SOURCE_DIR}/src/core/*.cpp)
file(GLOB_RECURSE MODELS_SOURCES ${PROJECT_SOURCE_DIR}/src/models/*.cpp)
file(GLOB_RECURSE UTILS_SOURCES ${PROJECT_SOURCE_DIR}/src/utils/*.cpp)
file(GLOB_RECURSE IO_SOURCES ${PROJECT_SOURCE_DIR}/src/io/*.cpp)
file(GLOB_RECURSE VISUALIZATION_SOURCES ${PROJECT_SOURCE_DIR}/src/visualization/*.cpp)

# 主程序源文件
set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/src/main.cpp)

# 合并所有源文件
set(SOURCES
    ${CORE_SOURCES}
    ${MODELS_SOURCES}
    ${UTILS_SOURCES}
    ${IO_SOURCES}
    ${VISUALIZATION_SOURCES}
    ${MAIN_SOURCE}
)

# 创建可执行文件
add_executable(particle_slope_simulation ${SOURCES})

# 链接依赖库
target_link_libraries(particle_slope_simulation
    Eigen3::Eigen
)

# 设置输出目录
set_target_properties(particle_slope_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)

# 安装配置
install(TARGETS particle_slope_simulation
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 测试配置（可选）
# enable_testing()
# add_subdirectory(tests)

# 示例配置
# add_subdirectory(examples)
