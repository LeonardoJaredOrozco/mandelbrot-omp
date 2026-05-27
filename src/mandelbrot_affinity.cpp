/*
 * mandelbrot_affinity.cpp
 * Benchmark final:
 * - Speedup de Tarea A y B con 1 a 6 hilos
 * - Comparación con/sin afinidad de hilos (OMP_PROC_BIND / OMP_PLACES)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <omp.h>

static const int WIDTH    = 7680;
static const int HEIGHT   = 4320;
static const int MAX_ITER = 1000;
static const double X_MIN = -2.5, X_MAX = 1.0;
static const double Y_MIN = -1.25, Y_MAX = 1.25;
static const int RADIUS   = 5;
static const int KSIZE    = 2 * RADIUS + 1;

struct Pixel { unsigned char r, g, b; };

Pixel colorize(int iter) {
    if (iter == MAX_ITER) return {0, 0, 0};
    double t = static_cast<double>(iter) / MAX_ITER;
    unsigned char r = static_cast<unsigned char>(9   * (1-t) * t*t*t * 255);
    unsigned char g = static_cast<unsigned char>(15  * (1-t)*(1-t) * t*t * 255);
    unsigned char b = static_cast<unsigned char>(8.5 * (1-t)*(1-t)*(1-t) * t * 255);
    return {r, g, b};
}

void build_gaussian_kernel(std::vector<double>& kernel, double sigma = 2.0) {
    kernel.resize(KSIZE * KSIZE);
    double sum = 0.0;
    for (int ky = -RADIUS; ky <= RADIUS; ++ky)
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            double v = std::exp(-(kx*kx + ky*ky) / (2.0 * sigma * sigma));
            kernel[(ky+RADIUS)*KSIZE + (kx+RADIUS)] = v;
            sum += v;
        }
    for (auto& v : kernel) v /= sum;
}

// ---------------------------------------------------------------------------
// Tarea A — retorna tiempo
// ---------------------------------------------------------------------------
double run_taskA(std::vector<Pixel>& img, int nthreads) {
    omp_set_num_threads(nthreads);
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < HEIGHT; ++py)
        for (int px = 0; px < WIDTH; ++px) {
            double cx = X_MIN + (X_MAX - X_MIN) * px / (WIDTH  - 1);
            double cy = Y_MIN + (Y_MAX - Y_MIN) * py / (HEIGHT - 1);
            double zx = 0.0, zy = 0.0;
            int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
                zx = tmp;
                ++iter;
            }
            img[py * WIDTH + px] = colorize(iter);
        }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Tarea B — retorna tiempo (versión canales separados)
// ---------------------------------------------------------------------------
double run_taskB(const std::vector<Pixel>& src,
                       std::vector<Pixel>& dst,
                 const std::vector<double>& kernel,
                 int nthreads) {
    omp_set_num_threads(nthreads);
    const int N  = WIDTH * HEIGHT;
    const int KN = KSIZE * KSIZE;

    std::vector<float> ch_r(N), ch_g(N), ch_b(N);
    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i) {
        ch_r[i] = src[i].r;
        ch_g[i] = src[i].g;
        ch_b[i] = src[i].b;
    }

    float kf[KN];
    for (int i = 0; i < KN; ++i) kf[i] = static_cast<float>(kernel[i]);

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;
            int k = 0;
            for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
                int ny = std::max(0, std::min(HEIGHT-1, py+ky));
                const float* row_r = &ch_r[ny * WIDTH];
                const float* row_g = &ch_g[ny * WIDTH];
                const float* row_b = &ch_b[ny * WIDTH];
                #pragma omp simd reduction(+:sr,sg,sb)
                for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                    int nx = std::max(0, std::min(WIDTH-1, px+kx));
                    float w = kf[k + kx + RADIUS];
                    sr += w * row_r[nx];
                    sg += w * row_g[nx];
                    sb += w * row_b[nx];
                }
                k += KSIZE;
            }
            dst[py * WIDTH + px] = {
                static_cast<unsigned char>(sr),
                static_cast<unsigned char>(sg),
                static_cast<unsigned char>(sb)
            };
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void save_ppm(const std::string& filename, const std::vector<Pixel>& img) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (const auto& p : img)
        f.write(reinterpret_cast<const char*>(&p), 3);
}

int main() {
    std::vector<Pixel> img(WIDTH * HEIGHT);
    std::vector<Pixel> dst(WIDTH * HEIGHT);
    std::vector<double> kernel;
    build_gaussian_kernel(kernel);

    // Tiempos secuenciales de referencia
    omp_set_num_threads(1);
    double seq_A = run_taskA(img, 1);
    double seq_B = run_taskB(img, dst, kernel, 1);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Referencia secuencial:\n";
    std::cout << "  Tarea A: " << seq_A << " s\n";
    std::cout << "  Tarea B: " << seq_B << " s\n\n";

    // ── Tabla speedup 1 a 6 hilos ──────────────────────────────────────────
    std::cout << std::left
              << std::setw(8)  << "Hilos"
              << std::setw(12) << "A(s)"
              << std::setw(12) << "SpeedupA"
              << std::setw(12) << "B(s)"
              << std::setw(12) << "SpeedupB"
              << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (int t = 1; t <= 6; ++t) {
        double tA = run_taskA(img, t);
        double tB = run_taskB(img, dst, kernel, t);
        std::cout << std::setw(8)  << t
                  << std::setw(12) << tA
                  << std::setw(12) << seq_A / tA
                  << std::setw(12) << tB
                  << std::setw(12) << seq_B / tB
                  << "\n";
    }

    // ── Benchmark de afinidad ──────────────────────────────────────────────
    std::cout << "\n--- Afinidad de hilos (3 hilos fisicos) ---\n";
    std::cout << "OMP_PROC_BIND=" << (getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "no definido") << "\n";
    std::cout << "OMP_PLACES="    << (getenv("OMP_PLACES")    ? getenv("OMP_PLACES")    : "no definido") << "\n\n";

    omp_set_num_threads(3);
    double tA_aff = run_taskA(img, 3);
    double tB_aff = run_taskB(img, dst, kernel, 3);
    std::cout << "Con configuracion actual (3 hilos):\n";
    std::cout << "  Tarea A: " << tA_aff << " s  (speedup: " << seq_A/tA_aff << "x)\n";
    std::cout << "  Tarea B: " << tB_aff << " s  (speedup: " << seq_B/tB_aff << "x)\n";

    save_ppm("mandelbrot_final.ppm", dst);
    std::cout << "\nImagen guardada: mandelbrot_final.ppm\n";

    return 0;
}
