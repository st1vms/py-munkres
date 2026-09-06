import time
import matplotlib.pyplot as plt
import numpy as np

# Module imports for benchmark targets
from pymunkres import munkres as st1vms_solver
from munkres import Munkres as BMC_Munkres
from scipy.optimize import linear_sum_assignment as scipy_solver

def run_benchmark():
    # Matrix dimensions for testing
    matrix_sizes = [10, 20, 30, 40, 50, 75, 100]

    timing_st1vms = []
    timing_bmc = []
    timing_scipy = []

    bmc_engine = BMC_Munkres()

    for size in matrix_sizes:
        # Generate uniform random cost matrix
        matrix_np = np.random.randint(1, 100, size=(size, size))
        matrix_list = matrix_np.tolist()

        # 1. Measure st1vms/py-munkres performance
        start_time = time.perf_counter()
        st1vms_solver(matrix_list)
        timing_st1vms.append(time.perf_counter() - start_time)

        # 2. Measure bmc/munkres performance
        start_time = time.perf_counter()
        bmc_engine.compute(matrix_list)
        timing_bmc.append(time.perf_counter() - start_time)

        # 3. Measure scipy performance (C extension reference)
        start_time = time.perf_counter()
        scipy_solver(matrix_np)
        timing_scipy.append(time.perf_counter() - start_time)

    # Plot execution performance on logarithmic scale
    plt.figure(figsize=(9, 5))
    plt.plot(matrix_sizes, timing_st1vms, label='st1vms/py-munkres', marker='o', color='red', linewidth=2)
    plt.plot(matrix_sizes, timing_bmc, label='bmc/munkres', marker='^', color='orange', linewidth=2, linestyle='--')
    plt.plot(matrix_sizes, timing_scipy, label='scipy.optimize', marker='s', color='green', linewidth=2, linestyle=':')

    plt.yscale('log')
    plt.xlabel('Matrix Dimension (N x N)')
    plt.ylabel('Execution Time in Seconds (Log Scale)')
    plt.title('Performance Benchmark: st1vms vs bmc vs SciPy')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('munkres_benchmark.png')
    plt.show()

if __name__ == "__main__":
    run_benchmark()