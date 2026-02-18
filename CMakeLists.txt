cmake_minimum_required(VERSION 3.14)
project(Vulkan_Cloth_Collision_Sim)

# Standard C++ version
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCES 
    src/Vulkan_XPBD_SphereCollision.cpp
)
include(FetchContent)

# -------------------------------------------------------------------
# 1. Vulkan SDK (need to be installed)
# -------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# -------------------------------------------------------------------
# 2. GLFW (window generationa and input management)
# -------------------------------------------------------------------
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4	# recent version tag
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# -------------------------------------------------------------------
# 3. GLM (math lib)
# -------------------------------------------------------------------
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.3	# recent version tag
)
FetchContent_MakeAvailable(glm)

# -------------------------------------------------------------------
# 4. exe file generation and link
# -------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# -------------------------------------------------------------------
# 5. copy shaders folder to exe file
# -------------------------------------------------------------------
add_custom_command(TARGET VulkanPhysicsSim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:VulkanPhysicsSim>/shaders"
    COMMENT "Copying shaders to the executable directory..."
)

# header and library link
target_link_libraries(${PROJECT_NAME} PRIVATE 
    Vulkan::Vulkan 
    glfw 
    glm::glm
)