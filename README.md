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
  3	| Fix race condition en histograma (atomic vs reduction)    |
