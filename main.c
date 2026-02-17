// main.c
#include <stdio.h>
#include "matrix.h"
#include "gradient_descent.h"
// Testing
int main() {

Matrix Input = create_matrix(1, 3);
set_value(Input, 0, 0, 1.0f);
set_value(Input, 0, 1, 2.0f);
set_value(Input, 0, 2, 3.0f);


Matrix Initial_weight = create_matrix(3, 1);
set_value(Initial_weight, 0, 0, 0.1f);
set_value(Initial_weight, 1, 0, 0.1f);
set_value(Initial_weight, 2, 0, 0.1f);

Matrix Target = create_matrix(1, 1);
set_value(Target, 0, 0, 14.0f);
    
train(Input, Initial_weight, Target, 10, 0.01f);
    
free_matrix(Input);
free_matrix(Initial_weight);
free_matrix(Target);
}