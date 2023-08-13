#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "my_xla/test_graph.h"

int main(int argc, char** argv) {
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());


  foo::bar::MatMulAdd matmuladd;
  matmuladd.set_thread_pool(&device);

  // Set up args and run the computation.
  int d = 1;
  for (int i = 0; i < 2; i++)
	  for (int j = 0; j < 3; j++)
		  matmuladd.arg0(i, j) = d++;

  for (int i = 0; i < 3; i++)
	  for (int j = 0; j < 4; j++)
		  matmuladd.arg1(i, j) = d++;

  for (int i = 0; i < 2; i++)
	  for (int j = 0; j < 4; j++)
		  matmuladd.arg2(i, j) = d++;

  std::cout << "x=" << std::endl;
  for (int i = 0; i < 2; i++) {
	  for (int j = 0; j < 3; j++)
		  std::cout << matmuladd.arg0(i, j) << " ";
	  std::cout << std::endl;
  }

  std::cout << "y=" << std::endl;
  for (int i = 0; i < 3; i++) {
	  for (int j = 0; j < 4; j++)
		  std::cout << matmuladd.arg1(i, j) << " ";
	  std::cout << std::endl;
  }

  std::cout << "basis=" << std::endl;
  for (int i = 0; i < 2; i++) {
	  for (int j = 0; j < 4; j++)
		  std::cout << matmuladd.arg2(i, j) << " ";
	  std::cout << std::endl;
  }


  matmuladd.Run();

  std::cout << "out=" << std::endl;
  for (int i = 0; i < 2; i++) {
	  for (int j = 0; j < 4; j++)
		  std::cout << matmuladd.result0(i, j) << " ";
	  std::cout << std::endl;
  }

  return 0;
}
