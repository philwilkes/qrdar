

cmake_minimum_required(VERSION 2.8)
project(qrdar)

SET(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} /home/ucfaptv/opt/treeseg/include )
link_directories(/home/ucfaptv/opt/treeseg/lib)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

	add_executable(mon2pcd mon2pcd.cpp)
    target_link_libraries(mon2pcd pthread scanlib-mt riboost_system-mt)
	
	add_executable(rxp2pcd_refl rxp2pcd_refl.cpp)
	target_link_libraries(rxp2pcd_refl pthread scanlib-mt riboost_system-mt)
	
	add_executable(rxp2pcd_loc rxp2pcd_loc.cpp)
	target_link_libraries(rxp2pcd_loc pthread scanlib-mt riboost_system-mt)
	
ENDIF()

