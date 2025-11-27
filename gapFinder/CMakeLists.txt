cmake_minimum_required(VERSION 3.8)
project(gap_finder)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  configs
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/gap_finder_node.py
  scripts/corridor_controller.py
  scripts/gap_finder_processing.py
  scripts/tunnel_following_node.py
  scripts/corridor_enable_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()