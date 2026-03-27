#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"
#include "data_loader.h"
#include "gradient_descent.h"
#include <time.h>
void iris_dataset_train();

int main()
{
    srand(time(NULL));
    iris_dataset_train();
    return 0;
}


void iris_dataset_train(){
    // Initialize all values to random numbers
    Matrix Weights_1 = create_matrix(4, 8);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            set_value(Weights_1, i, j, ((float)rand()/(float)RAND_MAX) - 0.5f);

    Matrix Weights_2 = create_matrix(8, 3);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++)
            set_value(Weights_2, i, j, ((float)rand()/(float)RAND_MAX) - 0.5f);

    // Initialize all values to 0
    Matrix Bias_1 = create_matrix(1, 8);
    for (int i = 0; i < 8; i++)
        set_value(Bias_1, 0, i, 0.0f);

    Matrix Bias_2 = create_matrix(1, 3);
    for (int i = 0; i < 3; i++)
        set_value(Bias_2, 0, i, 0.0f);

    // Initialize all values to 0
    Matrix Velocity_W1 = create_matrix(4, 8);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            set_value(Velocity_W1, i, j, 0.0f);

    Matrix Velocity_W2 = create_matrix(8, 3);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++)
            set_value(Velocity_W2, i, j, 0.0f);

    Matrix Velocity_B1 = create_matrix(1, 8);
    for (int i = 0; i < 1; i++)
        for (int j = 0; j < 8; j++)
            set_value(Velocity_B1, i, j, 0.0f);

    Matrix Velocity_B2 = create_matrix(1, 3);
    for (int i = 0; i < 1; i++)
        for (int j = 0; j < 3; j++)
            set_value(Velocity_B2, i, j, 0.0f);
    
    Matrix Iris_Features = load_iris_features("others/iris.csv");
    Matrix Iris_Features_Normalized = matrix_normalize(Iris_Features);
    Matrix Iris_Labels = load_iris_labels("others/iris.csv");

    shuffle_dataset(Iris_Features_Normalized, Iris_Labels);

    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        float learning_rate = 0.001f;
        float momentum = 0.9f;
        
        for (int sample_idx = 0; sample_idx < 120; sample_idx++) {
            Matrix Input = get_row(Iris_Features_Normalized, sample_idx);
            Matrix Label_Matrix = get_row(Iris_Labels, sample_idx);
            int label_value = (int)get_value(Label_Matrix, 0, 0);
            Matrix Target = label_to_onehot(label_value, 3); 
            free_matrix(Label_Matrix);

            Matrix Hidden, Hidden_raw, Output;
            Matrix Grad_w1, Grad_w2, Grad_b1, Grad_b2;
            Matrix New_w1, New_w2, New_b1, New_b2;

            // Forward pass
            Output = forward_pass_2layer_softmax(Input, Weights_1, Weights_2, Bias_1, Bias_2, &Hidden, &Hidden_raw);
            float loss = cross_entropy_loss(Output, Target);
            total_loss += loss;
            
            // Backward Pass
            backward_pass_2layer(Output, Hidden, Hidden_raw, Input, Target, Weights_2, &Grad_w1, &Grad_w2, &Grad_b1, &Grad_b2);

            // Computing New Weights
            New_w1 = update_weights_momentum(Weights_1, Grad_w1, &Velocity_W1, learning_rate, momentum);
            New_w2 = update_weights_momentum(Weights_2, Grad_w2, &Velocity_W2, learning_rate, momentum);
            New_b1 = update_weights_momentum(Bias_1, Grad_b1, &Velocity_B1, learning_rate, momentum);
            New_b2 = update_weights_momentum(Bias_2, Grad_b2, &Velocity_B2, learning_rate, momentum);

            // Freeing matrices used in this sample
            free_matrix(Output); free_matrix(Hidden); free_matrix(Hidden_raw);
            free_matrix(Grad_w1); free_matrix(Grad_w2); free_matrix(Grad_b1); free_matrix(Grad_b2);
            free_matrix(Weights_1); free_matrix(Weights_2); free_matrix(Bias_1); free_matrix(Bias_2);
            free_matrix(Target);
            free_matrix(Input); // Fixed the leak here

            // Updating Old Weights using New Weights
            Weights_1 = New_w1; Weights_2 = New_w2; Bias_1 = New_b1; Bias_2 = New_b2;
        }
        
        if (epoch % 100 == 0) {
            int correct = 0;
            for (int sample_idx = 120; sample_idx < 150; sample_idx++) {
                Matrix Label_Matrix = get_row(Iris_Labels, sample_idx);
                int label_value = (int)get_value(Label_Matrix, 0, 0);
                Matrix Target = label_to_onehot(label_value, 3);
                Matrix Input = get_row(Iris_Features_Normalized, sample_idx);

                Matrix Test_Hidden, Test_Hidden_raw, Test_Output;
                Test_Output = forward_pass_2layer_softmax(Input, Weights_1, Weights_2, Bias_1, Bias_2, &Test_Hidden, &Test_Hidden_raw);

                int total = Test_Output.rows * Test_Output.cols;
                float max = Test_Output.data[0];
                int max_index = 0;

                for (int i = 1; i < total; i++) {
                    if (Test_Output.data[i] > max){
                        max = Test_Output.data[i];
                        max_index = i;
                    }
                }

                if (max_index == label_value) {
                    correct++;
                }

                free_matrix(Label_Matrix); free_matrix(Test_Hidden); 
                free_matrix(Test_Hidden_raw); free_matrix(Test_Output); 
                free_matrix(Input); free_matrix(Target);
            }
            float accuracy = (float)correct / 30.0f * 100.0f;
            printf("Epoch %d: Accuracy = %.2f%%\n", epoch, accuracy);
        }
    }

    // Final Cleanup
    free_matrix(Iris_Features);
    free_matrix(Iris_Features_Normalized);
    free_matrix(Iris_Labels);
    free_matrix(Weights_1);
    free_matrix(Weights_2);
    free_matrix(Bias_1);
    free_matrix(Bias_2);
    free_matrix(Velocity_W1);
    free_matrix(Velocity_W2);
    free_matrix(Velocity_B1);
    free_matrix(Velocity_B2);
}