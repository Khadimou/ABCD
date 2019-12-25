#ifndef DEF_NN
#define DEF_NN
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define max(a, b) ((a) > (b)) ? (a) : (b)
//2D access in a 1D array
#define INDEX(x, y, w) (((x) * (w)) + (y))

typedef unsigned u32;
typedef unsigned char byte;
typedef unsigned long long u64;

typedef struct ppm_s {
  int w; //Width
  int h; //Height 
  int t; //Threshold
  byte *px; }ppm_t;

typedef struct pgm_s{
    u64 w;
    u64 h;
    u64 t;
    byte *p;
}pgm_t;

ppm_t *rgbengrayscale(ppm_t *train_images);

u64 relu(u32 x);

pgm_t *pgm_load(char *fname);

void pgm_save(char *fname, pgm_t *p);

pgm_t *pgm_create(u64 h, u64 w, u64 t);

void pgm_close(pgm_t *p);

int convolve(byte *m, u64 mh, u64 mw, int *f, u64 fh, u64 fw);

void pgm_apply_sobel_filter(byte *img_in, byte *img_out, u64 h, u64 w, float threshold);

void pgm_apply_prewitt_filter(byte *img_in, byte *img_out, u64 h, u64 w, float threshold);

void pgm_apply_kirsch_filter(byte *img_in, byte *img_out, u64 h, u64 w);

ppm_t *ppm_open(char *fname);

void ppm_save(char *fname, ppm_t *p);

void ppm_close(ppm_t *p);

ppm_t *ppm_create(u64 h, u64 w, u64 t);

void test(ppm_t *test_image,int n_w,int n_h);

void train(ppm_t *pp_train_images);

#endif
