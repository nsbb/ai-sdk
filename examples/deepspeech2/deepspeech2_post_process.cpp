/*
 * Company:    AW
 * Author:     Zhang
 * Date:    2023/08/05
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "deepspeech2_post_process.h"

#define ALPHABET_SIZE 29
char alphabets[ALPHABET_SIZE] = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\''};

void removeDuplicateChars(char* str) {
    int len = strlen(str);
    if (len <= 1) {
        return;
    }
    char result[len];
    int resultIndex = 0;
    result[resultIndex++] = str[0];
    for (int i = 1; i < len; i++) {
        if (str[i] != str[i - 1]) {
            result[resultIndex++] = str[i];
        }
    }
    strcpy(str, result);
}

int deepspeech2_post_process(float *tensor_data)
{
    int rows = 378;
    int data_size = rows * ALPHABET_SIZE;  // 数据个数计数器

    // printf("data_size = %d \n", data_size);
    // printf("tensor[0] = %f \n", tensor_data[0]);

    float tensor[rows][ALPHABET_SIZE];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < ALPHABET_SIZE; j++) {
            tensor[i][j] = tensor_data[i * ALPHABET_SIZE + j];
        }
    }

    free(tensor_data);
    // // Print first row of tensor
    // for (int i = 0; i < ALPHABET_SIZE; i++) {
    //     printf("%f ", tensor[0][i]);
    // }
    // printf("\n");

    // Find maximum value index in each row
    int tensor_argmax[rows];
    for (int i = 0; i < rows; i++) {
        int max_index = 0;
        for (int j = 0; j < ALPHABET_SIZE; j++) {
            if (tensor[i][j] > tensor[i][max_index]) {
                max_index = j;
            }
        }
        tensor_argmax[i] = max_index;

    }

    // Convert maximum value indices to characters
    char results_1[rows + 1];
    char results[rows + 1];
    int a = 0;
    for (int i = 0; i < rows; i++) {
        if (tensor_argmax[i] < ALPHABET_SIZE) {
            results_1[i] = alphabets[tensor_argmax[i]];
            if (results_1[i] != '\0') {
                results[a] = results_1[i];
                // printf("%c", results[a]);
                a = a + 1;
            }
        } else {
            results_1[i] = '-';
        }
    }
    results[rows] = '\0';

    printf("Original array: %s\n", results);
    removeDuplicateChars(results);
    printf("Modified array : %s\n", results);

    return 0;
}