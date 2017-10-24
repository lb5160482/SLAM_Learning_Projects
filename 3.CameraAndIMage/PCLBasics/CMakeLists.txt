cmake_minimum_required( VERSION 2.8 )
project( PCLBasics )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigrn3/")

find_package(PCL REQUIRED COMPONENT common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(PCLBasics PCLBasics.cpp)
target_link_libraries(PCLBasics
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	)