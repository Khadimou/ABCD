#include "protos.h"

//
int main(int argc, char **argv)
{
    if (argc < 4)
        return printf("Usage: [mode] [input pgm] [output pgm]\n"
                      "\t-sobel  : apply Sobel filter\n"
                      "\t-kirsch : apply Kirsch filter\n"
                      "\t-prewitt: apply Prewitt filter\n", argv[0]), -1;
 
    pgm_t *p_in  = pgm_load(argv[2]);

    if (!p_in)
        return printf("Error: cannot open pgm file (%s)\n", argv[2]), -2;

    pgm_t *p_out = pgm_create(p_in->h, p_in->w, p_in->t);

    printf("Resolution: %llu Pixels, %llu MPixels\n", (p_in->h * p_in->w), (p_in->h * p_in->w) / 1000000);
    
    if (!strncmp(argv[1], "-sobel", 6))
        pgm_apply_sobel_filter(p_in->p, p_out->p, p_in->h, p_in->w, 100);
    else
        if (!strncmp(argv[1], "-kirsch", 7))
        pgm_apply_kirsch_filter(p_in->p, p_out->p, p_in->h, p_in->w);
    else
        if (!strncmp(argv[1], "-prewitt", 8))
        pgm_apply_prewitt_filter(p_in->p, p_out->p, p_in->h, p_in->w, 100);
    else
        return printf("Usage: [mode] [input pgm] [output pgm]\n"
                      "\t-sobel  : apply Sobel filter\n"
                      "\t-kirsch : apply Kirsch filter\n"
                      "\t-prewitt: apply Prewitt filter\n", argv[0]), -1;
    //
    pgm_save(argv[3], p_out);

    //
    pgm_close(p_in);
    pgm_close(p_out);

    return 0;
}
