#ifndef POSITIONALENCODING_H
#define POSITIONALENCODING_H

#include "matrix.h"
#include "error.h"
#include <stddef.h>

typedef struct {
    Matrix* Matrix;
} PosEncoding;

PosEncoding init_pos_encoding(const int dmodel, const int MAX_SEQ_LENGHT);

void free_pos_encoding(PosEncoding* PE);


#endif