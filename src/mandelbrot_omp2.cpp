/*
 * mandelbrot_omp2.cpp
 * Compara: atomic vs reduction vs false sharing
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
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
    #pragma omp parallel for schedule(static) collapse(2)
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

void apply_gaussian_blur(const std::vector<Pixel>& src,
                               std::vector<Pixel>& dst,
                         const std::vector<double>& kernel) {
    #pragma omp parallel for schedule(static)
    for (int py = 0; py < HEIGHT; ++py)
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

// ---------------------------------------------------------------------------
// HISTOGRAMA v1 — atomic (correcto, más lento)
// ---------------------------------------------------------------------------
void histogram_atomic(const std::vector<Pixel>& img,
                      std::vector<int>& hr,
                      std::vector<int>& hg,
                      std::vector<int>& hb) {
    hr.assign(256, 0);
    hg.assign(256, 0);
    hb.assign(256, 0);

    #pragma omp parallel for
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        #pragma omp atomic
        hr[img[i].r]++;
        #pragma omp atomic
        hg[img[i].g]++;
        #pragma omp atomic
        hb[img[i].b]++;
    }
}

// ---------------------------------------------------------------------------
// HISTOGRAMA v2 — reduction con arreglos locales (correcto, más rápido)
// ---------------------------------------------------------------------------
void histogram_reduction(const std::vector<Pixel>& img,
                         std::vector<int>& hr,
                         std::vector<int>& hg,
                         std::vector<int>& hb) {
    hr.assign(256, 0);
    hg.assign(256, 0);
    hb.assign(256, 0);

    #pragma omp parallel
    {
        // Cada hilo tiene su propio histograma local
        std::vector<int> local_r(256, 0);
        std::vector<int> local_g(256, 0);
        std::vector<int> local_b(256, 0);

        #pragma omp for
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            local_r[img[i].r]++;
            local_g[img[i].g]++;
            local_b[img[i].b]++;
        }

        // Reducción manual: un hilo a la vez suma su local al global
        #pragma omp critical
        {
            for (int k = 0; k < 256; ++k) {
                hr[k] += local_r[k];
                hg[k] += local_g[k];
                hb[k] += local_b[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HISTOGRAMA v3 — false sharing
// ---------------------------------------------------------------------------
void histogram_false_sharing(const std::vector<Pixel>& img) {
    // Arreglo contiguo: hilo 0 → [0..255], hilo 1 → [256..511], etc.
    int nthreads = omp_get_max_threads();
    std::vector<int> shared_hist(256 * nthreads, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* my_hist = &shared_hist[tid * 256];

        #pragma omp for
        for (int i = 0; i < WIDTH * HEIGHT; ++i)
            my_hist[img[i].r]++;
    }
    std::cout << "[False Sharing] Demostración completada\n";
}

void save_ppm(const std::string& filename, const std::vector<Pixel>& img) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (const auto& p : img)
        f.write(reinterpret_cast<const char*>(&p), 3);
    std::cout << "Imagen guardada: " << filename << "\n";
}

int main() {
    std::cout << "Hilos: " << omp_get_max_threads() << "\n\n";

    std::vector<Pixel> img(WIDTH * HEIGHT);
    std::vector<Pixel> blurred(WIDTH * HEIGHT);
    std::vector<double> kernel;
    build_gaussian_kernel(kernel);
    generate_mandelbrot(img);
    apply_gaussian_blur(img, blurred, kernel);
    save_ppm("mandelbrot_omp2.ppm", blurred);

    // ── Histograma atomic ──────────────────────────────────────────────────
    std::vector<int> hr, hg, hb;
    auto t0 = std::chrono::high_resolution_clock::now();
    histogram_atomic(blurred, hr, hg, hb);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_atomic = std::chrono::duration<double>(t1-t0).count();
    std::cout << "[Histograma atomic]     tiempo: " << t_atomic << " s\n";
    std::cout << "  R[0]=" << hr[0] << "  R[1]=" << hr[1] << "\n";

    // ── Histograma reduction ───────────────────────────────────────────────
    auto t2 = std::chrono::high_resolution_clock::now();
    histogram_reduction(blurred, hr, hg, hb);
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_reduction = std::chrono::duration<double>(t3-t2).count();
    std::cout << "[Histograma reduction]  tiempo: " << t_reduction << " s\n";
    std::cout << "  R[0]=" << hr[0] << "  R[1]=" << hr[1] << "\n";

    // ── False sharing demo ─────────────────────────────────────────────────
    histogram_false_sharing(blurred);

    std::cout << "\nSpeedup reduction vs atomic: "
              << t_atomic / t_reduction << "x\n";

    return 0;
}
