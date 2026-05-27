/*
 * mandelbrot_schedulers.cpp
 * Evaluación empírica de schedulers OpenMP en Tarea A:
 * static, dynamic y guided con distintos chunk sizes
 */

#include <iostream>
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

struct Pixel { unsigned char r, g, b; };

Pixel colorize(int iter) {
    if (iter == MAX_ITER) return {0, 0, 0};
    double t = static_cast<double>(iter) / MAX_ITER;
    unsigned char r = static_cast<unsigned char>(9   * (1-t) * t*t*t * 255);
    unsigned char g = static_cast<unsigned char>(15  * (1-t)*(1-t) * t*t * 255);
    unsigned char b = static_cast<unsigned char>(8.5 * (1-t)*(1-t)*(1-t) * t * 255);
    return {r, g, b};
}

// ---------------------------------------------------------------------------
// Genera Mandelbrot con scheduler configurable
// ---------------------------------------------------------------------------
double run_mandelbrot(std::vector<Pixel>& img,
                      const std::string& sched_name,
                      omp_sched_t sched,
                      int chunk) {
    omp_set_schedule(sched, chunk);

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(runtime)
    for (int py = 0; py < HEIGHT; ++py) {
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
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    int nthreads = omp_get_max_threads();
    std::cout << "Hilos: " << nthreads << "\n";
    std::cout << "Resolución: " << WIDTH << "×" << HEIGHT << "\n\n";

    std::vector<Pixel> img(WIDTH * HEIGHT);

    // Referencia secuencial
    double t_seq;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
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
        t_seq = std::chrono::duration<double>(t1 - t0).count();
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Secuencial (1 hilo):       " << t_seq << " s\n\n";
    }

    // Tabla de resultados
    std::cout << std::left
              << std::setw(28) << "Scheduler"
              << std::setw(12) << "Tiempo(s)"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    // ── static con distintos chunk sizes ───────────────────────────────────
    std::vector<int> chunks_static = {0, 1, 4, 16, 64, 256};
    for (int c : chunks_static) {
        double t = run_mandelbrot(img, "static", omp_sched_static, c);
        std::string label = "static(chunk=" + (c==0 ? std::string("default") : std::to_string(c)) + ")";
        std::cout << std::setw(28) << label
                  << std::setw(12) << t
                  << std::setw(10) << t_seq / t
                  << "\n";
    }

    std::cout << "\n";

    // ── dynamic con distintos chunk sizes ──────────────────────────────────
    std::vector<int> chunks_dynamic = {1, 4, 16, 64, 256};
    for (int c : chunks_dynamic) {
        double t = run_mandelbrot(img, "dynamic", omp_sched_dynamic, c);
        std::string label = "dynamic(chunk=" + std::to_string(c) + ")";
        std::cout << std::setw(28) << label
                  << std::setw(12) << t
                  << std::setw(10) << t_seq / t
                  << "\n";
    }

    std::cout << "\n";

    // ── guided con distintos chunk sizes ───────────────────────────────────
    std::vector<int> chunks_guided = {1, 4, 16, 64};
    for (int c : chunks_guided) {
        double t = run_mandelbrot(img, "guided", omp_sched_guided, c);
        std::string label = "guided(chunk=" + std::to_string(c) + ")";
        std::cout << std::setw(28) << label
                  << std::setw(12) << t
                  << std::setw(10) << t_seq / t
                  << "\n";
    }

    return 0;
}
