#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff)       |
           ((val >> 8)  & 0xff00)     |
           ((val << 8)  & 0xff0000)   |
           ((val << 24) & 0xff000000);
}

float* read_mnist_images(const char* filename, int* num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    uint32_t magic, n_imgs, n_rows, n_cols;
    (void)fread(&magic,  4, 1, file);
    (void)fread(&n_imgs, 4, 1, file);
    (void)fread(&n_rows, 4, 1, file);
    (void)fread(&n_cols, 4, 1, file);
    *num_images = swap_endian(n_imgs);
    int size = (*num_images) * swap_endian(n_rows) * swap_endian(n_cols);
    uint8_t* raw = (uint8_t*)malloc(size);
    (void)fread(raw, 1, size, file);
    fclose(file);
    float* norm = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) norm[i] = raw[i] / 255.0f;
    free(raw);
    return norm;
}

uint8_t* read_mnist_labels(const char* filename, int* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    uint32_t magic, n_lbls;
    (void)fread(&magic,  4, 1, file);
    (void)fread(&n_lbls, 4, 1, file);
    *num_labels = swap_endian(n_lbls);
    uint8_t* labels = (uint8_t*)malloc(*num_labels);
    (void)fread(labels, 1, *num_labels, file);
    fclose(file);
    return labels;
}

#endif
