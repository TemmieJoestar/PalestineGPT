#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "positional_encoding.h"
#include "matrix.h"
#include "error.h"

PosEncoding init_pos_encoding(const int dmodel, const int MAX_SEQ_LENGHT) {
    PosEncoding PE;
    PE.Matrix = create_matrix(MAX_SEQ_LENGHT, dmodel, false);
    float sin_value = 0;
    float cos_value = 0;

    for (int pos = 0; pos < MAX_SEQ_LENGHT; pos++) { 
        for (int i  = 0; i < dmodel; i+=2) { 
            float angle = (float)pos / powf(10000.0f, (2*i/(float)dmodel));
            sin_value = sinf(angle);
            set_value(PE.Matrix, pos, i, sin_value);

            if (i + 1 < dmodel){
                cos_value = cosf(angle);
                set_value(PE.Matrix, pos, i + 1, cos_value);
            }
        }
    }
    return PE;
}

void free_pos_encoding(PosEncoding* PE) {
    free_matrix(PE->Matrix);
}