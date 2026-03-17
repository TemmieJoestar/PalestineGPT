#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "gradient_descent.h"

void train_xor();

int main() {
    train_xor();
    return 0;
}

void train_xor() {
    srand(time(NULL));

    Matrix input1 = create_matrix(1, 2); set_value(input1, 0, 0, 0.0f); set_value(input1, 0, 1, 0.0f);
    Matrix target1 = create_matrix(1, 1); set_value(target1, 0, 0, 0.0f);

    Matrix input2 = create_matrix(1, 2); set_value(input2, 0, 0, 0.0f); set_value(input2, 0, 1, 1.0f);
    Matrix target2 = create_matrix(1, 1); set_value(target2, 0, 0, 1.0f);

    Matrix input3 = create_matrix(1, 2); set_value(input3, 0, 0, 1.0f); set_value(input3, 0, 1, 0.0f);
    Matrix target3 = create_matrix(1, 1); set_value(target3, 0, 0, 1.0f);
    
    Matrix input4 = create_matrix(1, 2); set_value(input4, 0, 0, 1.0f); set_value(input4, 0, 1, 1.0f);
    Matrix target4 = create_matrix(1, 1); set_value(target4, 0, 0, 0.0f);

    Matrix Bias_1 = create_matrix(1, 2);
    set_value(Bias_1, 0, 0, 0.0f);
    set_value(Bias_1, 0, 1, 0.0f);

    Matrix Bias_2 = create_matrix(1, 1);
    set_value(Bias_2, 0, 0, 0.0f);

    Matrix Velocity_W1 = create_matrix(2, 2);
    set_value(Velocity_W1, 0, 0, 0.0f);
    set_value(Velocity_W1, 0, 1, 0.0f);
    set_value(Velocity_W1, 1, 0, 0.0f);
    set_value(Velocity_W1, 1, 1, 0.0f);

    Matrix Velocity_W2 = create_matrix(2, 1);
    set_value(Velocity_W2, 0, 0, 0.0f);
    set_value(Velocity_W2, 1, 0, 0.0f);

    Matrix Velocity_B1 = create_matrix(1, 2);
    set_value(Velocity_B1, 0, 0, 0.0f);
    set_value(Velocity_B1, 0, 1, 0.0f);

    Matrix Velocity_B2 = create_matrix(1, 1);
    set_value(Velocity_B2, 0, 0, 0.0f);

    Matrix Weights_1 = create_matrix(2, 2);
    for (int i = 0; i < 2; i++) 
        for (int j = 0; j < 2; j++) 
            set_value(Weights_1, i, j, ((float)rand()/(float)RAND_MAX) - 0.5f);
    
    Matrix Weights_2 = create_matrix(2, 1); 
    for (int i = 0; i < 2; i++) 
        for (int j = 0; j < 1; j++) 
            set_value(Weights_2, i, j, ((float)rand()/(float)RAND_MAX) - 0.5f);

    int epochs = 100000;
    float learning_rate = 0.1f;
    float momentum = 0.9f;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        Matrix hidden, hidden_raw, output;
        Matrix grad_w1, grad_w2, grad_b1, grad_b2;
        Matrix new_w1, new_w2, new_b1, new_b2;
        
        // Example 1
        output = forward_pass_2layer(input1, Weights_1, Weights_2, Bias_1, Bias_2, &hidden, &hidden_raw);
        total_loss += mean_squared_error(output, target1);
        backward_pass_2layer(output, hidden, hidden_raw, input1, target1, Weights_2, &grad_w1, &grad_w2, &grad_b1, &grad_b2);
        new_w1 = update_weights_momentum(Weights_1, grad_w1, &Velocity_W1, learning_rate, momentum);
        new_w2 = update_weights_momentum(Weights_2, grad_w2, &Velocity_W2, learning_rate, momentum);
        new_b1 = update_weights_momentum(Bias_1, grad_b1, &Velocity_B1, learning_rate, momentum);
        new_b2 = update_weights_momentum(Bias_2, grad_b2, &Velocity_B2, learning_rate, momentum);
        free_matrix(output); free_matrix(hidden); free_matrix(hidden_raw);
        free_matrix(grad_w1); free_matrix(grad_w2); free_matrix(grad_b1); free_matrix(grad_b2);
        free_matrix(Weights_1); free_matrix(Weights_2); free_matrix(Bias_1); free_matrix(Bias_2);
        Weights_1 = new_w1; Weights_2 = new_w2; Bias_1 = new_b1; Bias_2 = new_b2;

        // Example 2
        output = forward_pass_2layer(input2, Weights_1, Weights_2, Bias_1, Bias_2, &hidden, &hidden_raw);
        total_loss += mean_squared_error(output, target2);
        backward_pass_2layer(output, hidden, hidden_raw, input2, target2, Weights_2, &grad_w1, &grad_w2, &grad_b1, &grad_b2);
        new_w1 = update_weights_momentum(Weights_1, grad_w1, &Velocity_W1, learning_rate, momentum);
        new_w2 = update_weights_momentum(Weights_2, grad_w2, &Velocity_W2, learning_rate, momentum);
        new_b1 = update_weights_momentum(Bias_1, grad_b1, &Velocity_B1, learning_rate, momentum);
        new_b2 = update_weights_momentum(Bias_2, grad_b2, &Velocity_B2, learning_rate, momentum);
        free_matrix(output); free_matrix(hidden); free_matrix(hidden_raw);
        free_matrix(grad_w1); free_matrix(grad_w2); free_matrix(grad_b1); free_matrix(grad_b2);
        free_matrix(Weights_1); free_matrix(Weights_2); free_matrix(Bias_1); free_matrix(Bias_2);
        Weights_1 = new_w1; Weights_2 = new_w2; Bias_1 = new_b1; Bias_2 = new_b2;

        // Example 3
        output = forward_pass_2layer(input3, Weights_1, Weights_2, Bias_1, Bias_2, &hidden, &hidden_raw);
        total_loss += mean_squared_error(output, target3);
        backward_pass_2layer(output, hidden, hidden_raw, input3, target3, Weights_2, &grad_w1, &grad_w2, &grad_b1, &grad_b2);
        new_w1 = update_weights_momentum(Weights_1, grad_w1, &Velocity_W1, learning_rate, momentum);
        new_w2 = update_weights_momentum(Weights_2, grad_w2, &Velocity_W2, learning_rate, momentum);
        new_b1 = update_weights_momentum(Bias_1, grad_b1, &Velocity_B1, learning_rate, momentum);
        new_b2 = update_weights_momentum(Bias_2, grad_b2, &Velocity_B2, learning_rate, momentum);
        free_matrix(output); free_matrix(hidden); free_matrix(hidden_raw);
        free_matrix(grad_w1); free_matrix(grad_w2); free_matrix(grad_b1); free_matrix(grad_b2);
        free_matrix(Weights_1); free_matrix(Weights_2); free_matrix(Bias_1); free_matrix(Bias_2);
        Weights_1 = new_w1; Weights_2 = new_w2; Bias_1 = new_b1; Bias_2 = new_b2;

        // Example 4
        output = forward_pass_2layer(input4, Weights_1, Weights_2, Bias_1, Bias_2, &hidden, &hidden_raw);
        total_loss += mean_squared_error(output, target4);
        backward_pass_2layer(output, hidden, hidden_raw, input4, target4, Weights_2, &grad_w1, &grad_w2, &grad_b1, &grad_b2);
        new_w1 = update_weights_momentum(Weights_1, grad_w1, &Velocity_W1, learning_rate, momentum);
        new_w2 = update_weights_momentum(Weights_2, grad_w2, &Velocity_W2, learning_rate, momentum);
        new_b1 = update_weights_momentum(Bias_1, grad_b1, &Velocity_B1, learning_rate, momentum);
        new_b2 = update_weights_momentum(Bias_2, grad_b2, &Velocity_B2, learning_rate, momentum);
        free_matrix(output); free_matrix(hidden); free_matrix(hidden_raw);
        free_matrix(grad_w1); free_matrix(grad_w2); free_matrix(grad_b1); free_matrix(grad_b2);
        free_matrix(Weights_1); free_matrix(Weights_2); free_matrix(Bias_1); free_matrix(Bias_2);
        Weights_1 = new_w1; Weights_2 = new_w2; Bias_1 = new_b1; Bias_2 = new_b2;

        if (epoch % 10000 == 0) {  
            printf("Epoch %d: Average Loss = %.6f\n", epoch, total_loss / 4.0f);
        }
    }

    printf("\nFinal XOR Results\n");
    Matrix test_h, test_hr, test_o;

    test_o = forward_pass_2layer(input1, Weights_1, Weights_2, Bias_1, Bias_2, &test_h, &test_hr);
    printf("Input (0,0) -> Output: %.4f (Target: 0.0)\n", get_value(test_o, 0, 0));
    free_matrix(test_o); free_matrix(test_h); free_matrix(test_hr);

    test_o = forward_pass_2layer(input2, Weights_1, Weights_2, Bias_1, Bias_2, &test_h, &test_hr);
    printf("Input (0,1) -> Output: %.4f (Target: 1.0)\n", get_value(test_o, 0, 0));
    free_matrix(test_o); free_matrix(test_h); free_matrix(test_hr);

    test_o = forward_pass_2layer(input3, Weights_1, Weights_2, Bias_1, Bias_2, &test_h, &test_hr);
    printf("Input (1,0) -> Output: %.4f (Target: 1.0)\n", get_value(test_o, 0, 0));
    free_matrix(test_o); free_matrix(test_h); free_matrix(test_hr);

    test_o = forward_pass_2layer(input4, Weights_1, Weights_2, Bias_1, Bias_2, &test_h, &test_hr);
    printf("Input (1,1) -> Output: %.4f (Target: 0.0)\n", get_value(test_o, 0, 0));
    free_matrix(test_o); free_matrix(test_h); free_matrix(test_hr);

    /*
    printf("\nFinal Model State\n");
    printf("[Layer 1 Weights]\n"); print_matrix(Weights_1);
    printf("[Layer 1 Bias]\n");    print_matrix(Bias_1);
    printf("\n[Layer 2 Weights]\n"); print_matrix(Weights_2);
    printf("[Layer 2 Bias]\n");    print_matrix(Bias_2);
    */
    free_matrix(input1); free_matrix(input2); free_matrix(input3); free_matrix(input4);
    free_matrix(target1); free_matrix(target2); free_matrix(target3); free_matrix(target4);
    free_matrix(Weights_1); free_matrix(Weights_2); free_matrix(Bias_1); free_matrix(Bias_2);
    free_matrix(Velocity_W1); free_matrix(Velocity_W2); free_matrix(Velocity_B1); free_matrix(Velocity_B2);
}