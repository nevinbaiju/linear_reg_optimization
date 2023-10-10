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
    __m256 _m_coeff_0 = _mm256_set1_ps(coeff[0]);
    __m256 _m_coeff_1 = _mm256_set1_ps(coeff[1]);
    __m256 _m_coeff_2 = _mm256_set1_ps(coeff[2]);
    __m256 _m_coeff_3 = _mm256_set1_ps(coeff[3]);
    __m256 _m_coeff_4 = _mm256_set1_ps(coeff[4]);
    __m256 _m_coeff_5 = _mm256_set1_ps(coeff[5]);
    __m256 _m_coeff_6 = _mm256_set1_ps(coeff[6]);
    __m256 _m_coeff_7 = _mm256_set1_ps(coeff[7]);

    __m256 _m_intercept = _mm256_set1_ps(intercept);

    
    for (int iteration = 0; iteration<num_iterations; iteration++)
    {
        #pragma omp parallel for schedule(static, 1000)
        for(int row_id=0; row_id < data.size(); row_id+=8){
            __m256 _m_x0, _m_x1, _m_x2, _m_x3, _m_x4, _m_x5, _m_x6, _m_x7, _m_result;
            _m_x0 = _mm256_setr_ps(data[row_id][0], data[row_id+1][0], data[row_id+2][0], data[row_id+3][0], 
                                   data[row_id+4][0], data[row_id+5][0], data[row_id+6][0], data[row_id+7][0]);
            _m_x1 = _mm256_setr_ps(data[row_id][1], data[row_id+1][1], data[row_id+2][1], data[row_id+3][1], 
                                   data[row_id+4][1], data[row_id+5][1], data[row_id+6][1], data[row_id+7][1]);
            _m_x2 = _mm256_setr_ps(data[row_id][2], data[row_id+1][2], data[row_id+2][2], data[row_id+3][2], 
                    data[row_id+4][2], data[row_id+5][2], data[row_id+6][2], data[row_id+7][2]);
            _m_x3 = _mm256_setr_ps(data[row_id][3], data[row_id+1][3], data[row_id+2][3], data[row_id+3][3], 
                        data[row_id+4][3], data[row_id+5][3], data[row_id+6][3], data[row_id+7][3]);
            _m_x4 = _mm256_setr_ps(data[row_id][4], data[row_id+1][4], data[row_id+2][4], data[row_id+3][4], 
                        data[row_id+4][4], data[row_id+5][4], data[row_id+6][4], data[row_id+7][4]);
            _m_x5 = _mm256_setr_ps(data[row_id][5], data[row_id+1][5], data[row_id+2][5], data[row_id+3][5], 
                        data[row_id+4][5], data[row_id+5][5], data[row_id+6][5], data[row_id+7][5]);
            _m_x6 = _mm256_setr_ps(data[row_id][6], data[row_id+1][6], data[row_id+2][6], data[row_id+3][6], 
                        data[row_id+4][6], data[row_id+5][6], data[row_id+6][6], data[row_id+7][6]);
            _m_x7 = _mm256_setr_ps(data[row_id][7], data[row_id+1][7], data[row_id+2][7], data[row_id+3][7], 
                        data[row_id+4][7], data[row_id+5][7], data[row_id+6][7], data[row_id+7][7]);       

            _m_x0 = _mm256_add_ps(_m_x0, _m_x1);
            _m_x2 = _mm256_add_ps(_m_x2, _m_x3);
            _m_x4 = _mm256_add_ps(_m_x4, _m_x5);
            _m_x6 = _mm256_add_ps(_m_x6, _m_x7); 

            _m_x0 = _mm256_add_ps(_m_x0, _m_x2);
            _m_x4 = _mm256_add_ps(_m_x4, _m_x6);

            _m_result = _mm256_add_ps(_m_x0, _m_result);
            _m_result = _mm256_add_ps(_m_x0, _m_result);
            _m_result = _mm256_add_ps(_m_x0, _m_intercept);

            _mm256_store_ps(&result[row_id], _m_result);
            // std::cout << "Ivide vare ethi" << std::endl;
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
