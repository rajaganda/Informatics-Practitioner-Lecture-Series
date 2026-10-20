import os
import time
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

def compute_chunk(args):
    start_idx, end_idx, step, total_steps = args
    local_sum = 0.0
    for i in range(start_idx, end_idx):
        x = (i + 0.5) * step
        local_sum += 4.0 / (1.0 + x * x)
    return local_sum

def run_benchmark(total_steps=40_000_000, num_workers=1):
    step = 1.0 / total_steps
    chunk_size = total_steps // num_workers
    ranges = []

    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = total_steps if i == num_workers - 1 else (i + 1) * chunk_size
        ranges.append((start_idx, end_idx, step, total_steps))

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_chunk, ranges))

    pi_estimate = sum(results) * step
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    total_flops = total_steps * 4.0
    gflops = (total_flops / elapsed_time) / 1e9

    return elapsed_time, gflops, pi_estimate

if __name__ == "__main__":
    total_cpu_cores = os.cpu_count() or 4
    print("=" * 60)
    print("BENCHMARK ARSITEKTUR KOMPUTER & HPC (MULTI-CORE)")
    print(f"Total logical CPU core terdeteksi: {total_cpu_cores}")
    print("=" * 60)

    core_configurations = [1]
    for c in [2, 4, 8, 16]:
        if c <= total_cpu_cores:
            core_configurations.append(c)

    execution_times = []
    gflops_list = []
    speedups = []

    print("\n[1/3] Menjalankan Benchmark Serial (1 Core)...")
    base_time, base_gflops, _ = run_benchmark(num_workers=1)
    execution_times.append(base_time)
    gflops_list.append(base_gflops)
    speedups.append(1.0)
    print(f"-> 1 Core: Waktu = {base_time:.4f} detik | Kinerja = {base_gflops:.4f} GFLOPS")

    print("\n[2/3] Menjalankan Benchmark Paralel Multi-Core...")
    for cores in core_configurations[1:]:
        print(f"-> Menguji {cores} Cores...")
        t, g, _ = run_benchmark(num_workers=cores)
        speedup = base_time / t
        execution_times.append(t)
        gflops_list.append(g)
        speedups.append(speedup)
        print(f"   Waktu = {t:.4f} detik | Kinerja = {g:.4f} GFLOPS | Speedup = {speedup:.2f}x")

    print("\n[3/3] Membuat grafik visualisasi performa...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(core_configurations, execution_times, marker='o', color='crimson', linewidth=2, label='Waktu Nyata')
    ax1.set_title('Waktu Eksekusi vs Jumlah Core CPU', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Jumlah Core CPU')
    ax1.set_ylabel('Waktu Eksekusi (detik)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(core_configurations, speedups, marker='s', color='royalblue', linewidth=2, label='Speedup Terukur')
    ax2.plot(core_configurations, core_configurations, linestyle='--', color='gray', label='Speedup Ideal (Linear)')
    ax2.set_title('Analisis Speedup (Hukum Amdahl)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Jumlah Core CPU')
    ax2.set_ylabel('Peningkatan Kecepatan (Speedup Factor)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_image = "benchmark_result.png"
    plt.savefig(output_image, dpi=300)
    print(f"Grafik berhasil disimpan sebagai '{output_image}'.")

    print("\n" + "=" * 60)
    print("RINGKASAN HASIL PENGUJIAN PERFORMA (OAK & HPC)")
    print(f"{'Cores':<10} | {'Waktu (detik)':<15} | {'GFLOPS':<15} | {'Speedup':<10}")
    print("-" * 60)
    for c, t, g, s in zip(core_configurations, execution_times, gflops_list, speedups):
        print(f"{c:<10} | {t:<15.4f} | {g:<15.4f} | {s:<10.2f}x")
    print("=" * 60)
