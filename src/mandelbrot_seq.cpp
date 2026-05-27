/*
 * mandelbrot_seq.cpp
 * Proyecto Final - Programación Paralela y Concurrente
 * Código base secuencial:
 *   Tarea A: Genera imagen 8K del conjunto de Mandelbrot (PPM)
 *   Tarea B: Aplica filtro Gaussiano 2D (radio 5) sobre la imagen
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

// ── Resolución 8K ──────────────────────────────────────────────────────────
static const int WIDTH  = 7680;
static const int HEIGHT = 4320;

// ── Parámetros Mandelbrot ───────────────────────────────────────────────────
static const int    MAX_ITER  = 1000;
static const double X_MIN     = -2.5;
static const double X_MAX     =  1.0;
static const double Y_MIN     = -1.25;
static const double Y_MAX     =  1.25;

// ── Kernel Gaussiano (radio 5, sigma 2) ────────────────────────────────────
static const int RADIUS = 5;
static const int KSIZE  = 2 * RADIUS + 1;   // 11×11

// ── Estructura de píxel RGB ─────────────────────────────────────────────────
struct Pixel { unsigned char r, g, b; };

// ---------------------------------------------------------------------------
// Colorea un píxel según las iteraciones (gradiente suave)
// ---------------------------------------------------------------------------
Pixel colorize(int iter) {
    if (iter == MAX_ITER) return {0, 0, 0};
    double t = static_cast<double>(iter) / MAX_ITER;
    unsigned char r = static_cast<unsigned char>(9   * (1-t) * t*t*t * 255);
    unsigned char g = static_cast<unsigned char>(15  * (1-t)*(1-t) * t*t * 255);
    unsigned char b = static_cast<unsigned char>(8.5 * (1-t)*(1-t)*(1-t) * t * 255);
    return {r, g, b};
}

// ---------------------------------------------------------------------------
// TAREA A: Genera el conjunto de Mandelbrot
// ---------------------------------------------------------------------------
void generate_mandelbrot(std::vector<Pixel>& img) {
    auto t0 = std::chrono::high_resolution_clock::now();

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
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[Tarea A] Tiempo secuencial: " << elapsed << " s\n";
}

// ---------------------------------------------------------------------------
// Construye el kernel Gaussiano normalizado
// ---------------------------------------------------------------------------
void build_gaussian_kernel(std::vector<double>& kernel, double sigma = 2.0) {
    kernel.resize(KSIZE * KSIZE);
    double sum = 0.0;
    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            double v = std::exp(-(kx*kx + ky*ky) / (2.0 * sigma * sigma));
            kernel[(ky+RADIUS)*KSIZE + (kx+RADIUS)] = v;
            sum += v;
        }
    }
    for (auto& v : kernel) v /= sum;
}

// ---------------------------------------------------------------------------
// TAREA B: Aplica el filtro Gaussiano
// ---------------------------------------------------------------------------
void apply_gaussian_blur(const std::vector<Pixel>& src,
                               std::vector<Pixel>& dst,
                         const std::vector<double>& kernel) {
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            double sr = 0, sg = 0, sb = 0;
            for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
                for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                    int ny = py + ky;
                    int nx = px + kx;
                    // Clamp en bordes
                    ny = std::max(0, std::min(HEIGHT-1, ny));
                    nx = std::max(0, std::min(WIDTH-1,  nx));
                    double w = kernel[(ky+RADIUS)*KSIZE + (kx+RADIUS)];
                    const Pixel& p = src[ny * WIDTH + nx];
                    sr += w * p.r;
                    sg += w * p.g;
                    sb += w * p.b;
                }
            }
            dst[py * WIDTH + px] = {
                static_cast<unsigned char>(sr),
                static_cast<unsigned char>(sg),
                static_cast<unsigned char>(sb)
            };
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[Tarea B] Tiempo secuencial: " << elapsed << " s\n";
}

// ---------------------------------------------------------------------------
// Guarda la imagen en formato PPM (sin dependencias externas)
// ---------------------------------------------------------------------------
void save_ppm(const std::string& filename, const std::vector<Pixel>& img) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
 s   for (const auto& p : img)
        f.write(reinterpret_cast<const char*>(&p), 3);
    std::cout << "Imagen guardada: " << filename << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Resolución: " << WIDTH << "×" << HEIGHT << "\n";
    std::cout << "Iteraciones máx: " << MAX_ITER << "\n\n";

    std::vector<Pixel> img(WIDTH * HEIGHT);
    std::vector<Pixel> blurred(WIDTH * HEIGHT);
    std::vector<double> kernel;

    build_gaussian_kernel(kernel);

    generate_mandelbrot(img);
    apply_gaussian_blur(img, blurred, kernel);
    save_ppm("mandelbrot_blurred.ppm", blurred);

    return 0;
}
