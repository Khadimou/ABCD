#define _GNU_SOURCE
#include "protos.h"
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <dirent.h>
#include <stdio.h>

//
int main(int argc, char **argv)
{
    if(argc<2){
      return printf("Usage : [train ppm] \n"),-1;
    }

    char* current_test_char = "Cancer not detected\n";
    int bad_test_classifications = 0,total_test_classifications=0;
    float tdeb,tfin;

    char *path = argv[1];
    struct dirent **list = NULL;
    int nb_files = scandir(path,&list,NULL,alphasort);

    if(nb_files<0){
      perror("scandir");
      return 1;
    }

    ppm_t *train_images = (ppm_t*)malloc(sizeof(ppm_t));
    //Loading images pre-processed, and training

    tdeb = omp_get_wtime();
    for(int i=2;i<nb_files;i++){
      char buf[512];
      snprintf(buf,512,"%s%s",path,list[i]->d_name);
      train_images = ppm_open(buf);
      printf("Resolution: %u Pixels, %u MPixels\n", (train_images->h * train_images->w), (train_images->h * train_images->w) / 1000000);
      trainer(nb_files-2,train_images,"./data.txt");
      //data_test("./data.txt",train_images);
      if (current_test_char == output_test_char(get_res(train_images),train_images)){
            bad_test_classifications++;
      }

        total_test_classifications++;
        ppm_close(train_images);
    }
    tfin = omp_get_wtime();
    printf("Fin de la boucle\n");
    printf("Temps elapsed : %f s\n", tfin-tdeb);

    //Percentage of bad test classifications
    printf("bad test classifications %d\n", bad_test_classifications );
    printf("total classifications %d\n",total_test_classifications);
    float pbtc = (bad_test_classifications * 100) / total_test_classifications;
    printf("pourcentage de mauvaises classifications %f\n" ,pbtc);
    
    if (!train_images )
      return printf("Error: cannot open ppm file (%s) \n", argv[1]), -1;

    
    return 0;
}
