#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <immintrin.h>

void generate_inference(int num_iterations, std::vector<float*>& data, float result[], float coeff[], float intercept) {
    for (int iteration = 0; iteration<num_iterations; iteration++)
    {
        #pragma omp parallel for
        for(int row_id=0; row_id < data.size(); row_id++){
            for (int i=0; i<8; i++) {
                // std::cout << row[i] *coeff[i] << std::endl;
                result[row_id] += data[row_id][i]*coeff[i];
            }
            result[row_id] += intercept;
            // std::cout << result[row_id] << std::endl;
            row_id++;
        }
    }
}

float hsum_double_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);

    __m128 high64 = _mm_unpackhi_ps(vlow, vlow);
    return  _mm_cvtss_f32(_mm_add_ps(vlow, high64));  // reduce to scalar
}

void print_register(__m256 __m256_reg){
    float buffer[8];
    _mm256_store_ps(&buffer[0], __m256_reg);
    for(int i=0; i<8; i++){
        std::cout << buffer[i] << ", ";
    }
    std::cout << "\n";
}

void generate_inference_intrinsics(int num_iterations, std::vector<float*>& data, float result[], float coeff[], float intercept) {
    float buffer[8];
    for (int iteration = 0; iteration<num_iterations; iteration++)
    {
        __m256 _mm_coeff = _mm256_load_ps(&coeff[0]);
        __m256 _mm_result, _mm_x, random_reg;
        #pragma omp parallel for
        for(int row_id=0; row_id < 5; row_id++){
            // std::cout << row[i] *coeff[i] << std::endl;
            _mm_x = _mm256_loadu_ps(&data[row_id][0]);
            _mm256_store_ps(&buffer[0], _mm_x);
            
            _mm_result = _mm256_mul_ps(_mm_coeff, _mm_x);
            _mm256_store_ps(&buffer[0], _mm_result);
            
            print_register(_mm_result);
            result[row_id] = buffer[0]+buffer[1]+buffer[2]+buffer[3]+
                             buffer[4]+buffer[5]+buffer[6]+buffer[7]+intercept;
            
            std::cout << result[row_id] << std::endl;
        }
    }
}

int main() {
    // Specify the CSV file path
    std::string filePath = "train_data.csv";

    // Open the CSV file
    std::ifstream inputFile(filePath);

    // Check if the file is successfully opened
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open the file.\n";
        return 1;
    }

    int num_interations = 1;

    alignas(32) float model_coefficients[8] = { 0.75568351,  1.12939616,  0.52521255, -0.13658123,  0.5836674 , -0.4185402 ,  0.06868902,  1.03548009};
    float model_intercept = -2214.75081678577;

    std::vector<float *> data;  // 2D vector to store the data
    float* result = new float[1000];
    for(int i=0; i<1000; i++){
        result[i] = 0;
    }

    // Skip the first line (column names)
    std::string line;
    std::getline(inputFile, line);

    while (std::getline(inputFile, line)) {
        std::istringstream ss(line);
        std::string cell;

        // Split the line by commas and process each column
        int i=0;
        float* row = new float[8];
        while (std::getline(ss, cell, ',')) {
            // Convert the column to float and add to the row
            // row.push_back(std::stof(cell));
            if(i == 8){
                continue;
            }
            row[i] = std::stof(cell);
            i++;
        }
        // Add the row to the data
        data.push_back(row);
    }

    // Close the file
    inputFile.close();

    auto start = std::chrono::high_resolution_clock::now();
    generate_inference(num_interations, data, result, model_coefficients, model_intercept);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();
    float inferences_per_second = (1000*num_interations)/(seconds*1e6);
    // Print the data
    std::cout << "Inferences per second: " << inferences_per_second << std::endl;
    std::cout << "Sample result: " << result[0] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    generate_inference_intrinsics(num_interations, data, result, model_coefficients, model_intercept);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    seconds = elapsed_seconds.count();
    inferences_per_second = (1000*num_interations)/(seconds*1e6);
    // Print the data
    std::cout << "\nIntrinsics ";
    std::cout << "Inferences per second: " << inferences_per_second << std::endl;
    std::cout << "Sample result: " << result[0] << std::endl;
    for (const auto& row : data) {
        // for (int i=0; i<8; i++) {
        //     std::cout << row[i] << ' ';
        // }
        delete[] row;
        // std::cout << '\n';
    }

    return 0;
}
