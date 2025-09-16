cmake_minimum_required(VERSION 3.16)

project(thread-pinning
  VERSION 0.1
  LANGUAGES CXX
)

include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.1.1
)

FetchContent_MakeAvailable(kamping)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(rank-pinning main.cpp)

target_link_libraries(rank-pinning
  PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
  PUBLIC
    kamping::kamping
)
