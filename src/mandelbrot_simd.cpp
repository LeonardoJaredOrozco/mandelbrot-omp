/*
 * mandelbrot_simd.cpp
 * Tarea B con vectorización SIMD forzada (SPMD)
 * v1: accesos no contiguos (lento, documentado)
 * v2: canales separados para acceso contiguo (vectorizable)
 * Verificar vectorización con: -fopt-info-vec-optimized
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

void generate_mandelbrot(std::vector<Pixel>& img) {
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
// TAREA B sin SIMD — versión base paralela
// ---------------------------------------------------------------------------
double blur_no_simd(const std::vector<Pixel>& src,
                          std::vector<Pixel>& dst,
                    const std::vector<double>& kernel) {
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            double sr = 0, sg = 0, sb = 0;
            for (int ky = -RADIUS; ky <= RADIUS; ++ky)
                for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                    int ny = std::max(0, std::min(HEIGHT-1, py+ky));
                    int nx = std::max(0, std::min(WIDTH-1,  px+kx));
                    double w = kernel[(ky+RADIUS)*KSIZE + (kx+RADIUS)];
                    const Pixel& p = src[ny * WIDTH + nx];
                    sr += w * p.r;
                    sg += w * p.g;
                    sb += w * p.b;
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

// ---------------------------------------------------------------------------
// TAREA B con SIMD v1 — accesos no contiguos (documentado como antipatrón)
// ---------------------------------------------------------------------------
double blur_simd(const std::vector<Pixel>& src,
                       std::vector<Pixel>& dst,
                 const std::vector<double>& kernel) {
    const int KN = KSIZE * KSIZE;
    float kf[KN];
    for (int i = 0; i < KN; ++i) kf[i] = static_cast<float>(kernel[i]);

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;

            int ny_arr[KSIZE * KSIZE];
            int nx_arr[KSIZE * KSIZE];
            int idx = 0;
            for (int ky = -RADIUS; ky <= RADIUS; ++ky)
                for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                    ny_arr[idx] = std::max(0, std::min(HEIGHT-1, py+ky));
                    nx_arr[idx] = std::max(0, std::min(WIDTH-1,  px+kx));
                    ++idx;
                }

            #pragma omp simd reduction(+:sr,sg,sb)
            for (int k = 0; k < KN; ++k) {
                const Pixel& p = src[ny_arr[k] * WIDTH + nx_arr[k]];
                sr += kf[k] * p.r;
                sg += kf[k] * p.g;
                sb += kf[k] * p.b;
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

// ---------------------------------------------------------------------------
// TAREA B con SIMD v2 — canales separados para acceso contiguo
// ---------------------------------------------------------------------------
double blur_simd_v2(const std::vector<Pixel>& src,
                          std::vector<Pixel>& dst,
                    const std::vector<double>& kernel) {
    const int N = WIDTH * HEIGHT;
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
    std::cout << "Hilos: " << omp_get_max_threads() << "\n\n";

    std::vector<Pixel> img(WIDTH * HEIGHT);
    std::vector<Pixel> dst1(WIDTH * HEIGHT);
    std::vector<Pixel> dst2(WIDTH * HEIGHT);
    std::vector<double> kernel;

    build_gaussian_kernel(kernel);
    generate_mandelbrot(img);

    // ── Sin SIMD ───────────────────────────────────────────────────────────
    double t_no_simd = blur_no_simd(img, dst1, kernel);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[Tarea B sin SIMD]  " << t_no_simd << " s\n";

    // ── Con SIMD v1 (antipatrón) ───────────────────────────────────────────
    double t_simd = blur_simd(img, dst2, kernel);
    std::cout << "[Tarea B SIMD v1]   " << t_simd << " s  (accesos no contiguos)\n";
    std::cout << "Speedup SIMD v1:    " << t_no_simd / t_simd << "x\n\n";

    // ── Con SIMD v2 (canales separados) ────────────────────────────────────
    double t_simd2 = blur_simd_v2(img, dst2, kernel);
    std::cout << "[Tarea B SIMD v2]   " << t_simd2 << " s  (canales separados)\n";
    std::cout << "Speedup SIMD v2:    " << t_no_simd / t_simd2 << "x\n\n";

    save_ppm("mandelbrot_simd.ppm", dst2);
    std::cout << "Imagen guardada: mandelbrot_simd.ppm\n";

    return 0;
}
