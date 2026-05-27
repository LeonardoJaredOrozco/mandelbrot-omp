# Mandelbrot OPENMP - proyecto Final

**Programacion Paralela y Concurrente**
Centro Universitario de Ciencias Exactas e Ingenierias

## Descripcion
Generacion de imagen 8k del conjunto de Mandelbrot con filtro Gaussiano, 
optimizado progresivamente con OpenMP

## Compilacion (secuencial)
```bash
g++ -02 -o mandelbrot_seq src/mandelbrot_seq.cpp
./mandelbrot_seq
```

##Historial de commits

Commit  |  Descripcion

  1	| Codigo base secuencial (Tarea A + B)	|
  2	| Paralelizacion OpenMP baseline (IA)	|
  3	| Arreglo race condition en histograma (atomic vs reduction)    |
  4	| Schedulers: static/ dynamic/ guided - optimo: dynamic (chunk=1)    |
  5	| SPMD vectorizacion - SIMD v1 con accesos no contiguos    |
  5.1	| Arreglo SIMD v2: canales separados, speedup 1.71x por cache	|
  6	| Afinidad de hilos + benchmark final speedup 1-6 hilos    |
