#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <vector>

#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>

struct Info {
  int thread_id;
  int num_threads;
  int logical_core_id;
  std::array<char, MPI_MAX_PROCESSOR_NAME> mpi_proc_name;
  int mpi_name_length = 0;
};

void print_pinning() {
  kamping::Communicator comm;
  std::vector<Info> info_objects(omp_get_max_threads());
  std::array<char, MPI_MAX_PROCESSOR_NAME> mpi_proc_name;
  int mpi_name_length = 0;
  MPI_Get_processor_name(mpi_proc_name.data(), &mpi_name_length);

#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();    /* OpenMP thread_id */
    const int num_threads = omp_get_num_threads(); /* OpenMP num_threads */
    const int core_id = sched_getcpu();            /* ... core_id */
    auto &info_object = info_objects[thread_id];
    info_object.thread_id = thread_id;
    info_object.num_threads = num_threads;
    info_object.logical_core_id = core_id;
    info_object.mpi_proc_name = mpi_proc_name;
    info_object.mpi_name_length = mpi_name_length;
  }
  auto [recv_buf, recv_count] =
      comm.gather(kamping::send_buf(info_objects), kamping::recv_count_out());
  if (comm.is_root()) {
    for (std::size_t i = 0; i < recv_buf.size(); i += recv_count) {
      for (std::size_t j = i; j < i + static_cast<std::size_t>(recv_count);
           ++j) {
        auto const &info_object = recv_buf[j];
        std::string const node(info_object.mpi_proc_name.data(),
                               info_object.mpi_name_length);
        std::cout << "Rank: " << std::setw(5) << (i / recv_count)
                  << " OMP-Thread: " << std::setw(5) << info_object.thread_id
                  << "/" << std::setw(5) << info_object.num_threads
                  << " on logical core: " << std::setw(5)
                  << info_object.logical_core_id << " node:" << node
                  << std::endl;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  kamping::Environment env;
  print_pinning();
}
