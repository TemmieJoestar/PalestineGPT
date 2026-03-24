#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "matrix.h"
#include "gradient_descent.h"

int main()
{
    Matrix Dataset = create_matrix(1,3);
    set_value(Dataset, 0, 0, 150.0f); set_value(Dataset, 0, 1, 15.0f); set_value(Dataset, 0, 2, 67.0f);
    Matrix normalize = matrix_normalize(Dataset);
    print_matrix(normalize);
    return 0;
}

