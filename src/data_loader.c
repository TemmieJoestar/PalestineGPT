#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "error.h"

Matrix load_iris_features(const char* filename) {
    FILE *fp = fopen(filename, "r");
    char buffer[1024];
    
    if (!fp) {
        FATAL_ERROR("Could not open file %s.",filename);
    }

    Matrix Output = create_matrix(150, 4, false); 
    int current_row = 0;

    while (fgets(buffer, 1024, fp) && current_row < 150) {
        char *value = strtok(buffer, ",");
        int current_col = 0;

        while (value != NULL && current_col < 4) {
            float val = atof(value);
            set_value(Output, current_row, current_col, val);
            value = strtok(NULL, ",");
            current_col++;
        }
        current_row++;
    }

    fclose(fp);
    return Output;
}

Matrix load_iris_labels(const char* filename) {
    FILE *fp = fopen(filename, "r");
    char buffer[1024];
    
    if (!fp) {
        FATAL_ERROR("Could not open file %s.",filename); 
    }

    Matrix Output = create_matrix(150, 1, false); 
    int current_row = 0;

    while (fgets(buffer, sizeof(buffer), fp) && current_row < 150) {
        char *value = strtok(buffer, ",");
        
        for (int i = 0; i < 4 && value != NULL; i++) {
            value = strtok(NULL, ",");
        }

        if (value != NULL) {
            float label_val = atof(value);
            set_value(Output, current_row, 0, label_val);
        }
        current_row++;
    }

    fclose(fp);
    return Output;
}

Matrix label_to_onehot(int label, int num_classes) {
    Matrix row_vector = create_matrix(1, num_classes, false);

    for (int i = 0; i < num_classes; i++) {
        set_value(row_vector, 0, i, 0.0f);
    }

    if (label >= 0 && label < num_classes) {
        set_value(row_vector, 0, label, 1.0f);
    } else {
        ERROR("Label %d out of range for %d classes", label, num_classes);
    }

    return row_vector;
}

void shuffle_dataset(Matrix Features, Matrix Labels) {
    for (int i = Features.rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        matrix_swap_rows(Features, i, j);
        matrix_swap_rows(Labels, i, j);
    }
}