#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <awnn_lib.h>
#include <math.h>
#include "deepspeech2_post_process.h"

int main(int argc, char **argv) {
    printf("%s nbg input\n", argv[0]);
    if(argc < 3)
    {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }
    const char* nbg = argv[1];
    const char* input = argv[2];
    printf("input tensor_file = %s\n", input);

    // npu init
    awnn_init();
    // create network
    Awnn_Context_t *context = awnn_create(nbg);
    // copy input
    unsigned int width = 756;
    unsigned int height = 161;
    unsigned int sz = width * height;
    float *plant_data = (float*) malloc(sz * sizeof(float));
    unsigned char* data = (unsigned char*) malloc(sz * sizeof(unsigned char));
    // Read tensor data from file
    FILE *file = fopen(input, "r");
    if (file == NULL) {
        printf("Failed to open tensor file.\n");
        exit(-1);
    }
    for (int i = 0; i < sz; i++) {
        fscanf(file, "%f", &plant_data[i]);
        plant_data[i] /= 0.009404;
        if (plant_data[i] < 0)
            data[i] = 0;
        else if (plant_data[i] > 255)
            data[i] = 255;
        else
            data[i] = (unsigned char)round(plant_data[i]);
    }
    fclose(file);

    unsigned char *input_buffers[1] = {data};
    awnn_set_input_buffers(context, input_buffers);

    // process network
    awnn_run(context);

    // get result
    float **results = awnn_get_output_buffers(context);

    // post process
    deepspeech2_post_process(results[0]);

    free(plant_data);
    free(data);

    // destroy network
    // awnn_destroy(context);

    // npu uninit
    // awnn_uninit();

    return 0;
}

