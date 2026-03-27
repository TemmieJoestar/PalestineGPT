#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

Matrix load_iris_features(const char* filename) {
    FILE *fp = fopen(filename, "r");
    char buffer[1024];
    
    if (!fp) {
        printf(RED_TEXT("Error: Could not open file %s\n"), filename);
        exit(1); 
    }

    Matrix Output = create_matrix(150, 4); 
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
        printf(RED_TEXT("Error: Could not open file %s\n"), filename);
        exit(1); 
    }

    Matrix Output = create_matrix(150, 1); 
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
    Matrix row_vector = create_matrix(1, num_classes);

    for (int i = 0; i < num_classes; i++) {
        set_value(row_vector, 0, i, 0.0f);
    }

    if (label >= 0 && label < num_classes) {
        set_value(row_vector, 0, label, 1.0f);
    } else {
        printf(RED_TEXT("Warning: Label %d out of range for %d classes\n"), label, num_classes);
    }

    return row_vector;
}

void shuffle_dataset(Matrix Features, Matrix Labels) {
    // Seed the random number generator if you haven't in main
    // srand(time(NULL)); 

    for (int i = Features.rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        matrix_swap_rows(Features, i, j);
        matrix_swap_rows(Labels, i, j);
    }
}