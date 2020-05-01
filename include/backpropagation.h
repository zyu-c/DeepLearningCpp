#ifndef DEEPLEARNING_BACKPROPAGATION_H
#define DEEPLEARNING_BACKPROPAGATION_H

#include <vector>
#include "vector_function.h"

class Affine{
private:
    std::vector<std::vector<double>> w;
    std::vector<double> b;

public:
    Affine(int input_size, int output_size):
            w(input_size, std::vector<double>(output_size)),
            b(output_size){
    }

    void updateParam(std::vector<std::vector<double>> w, std::vector<double> b){
        if(this->w.size() == w.size() && this->w.at(0).size() == w.at(0).size()){
            this->w = w;
        }
        if(this->b.size() == b.size()){
            this->b = b;
        }
        return;
    }

    std::vector<std::vector<double>>forward(std::vector<std::vector<double>> x){
        std::vector<std::vector<double>> ans;
        ans = multiplication(x, w);
        for(int i = 0; i < w.at(0).size(); i++){
            for(int j = 0; j < b.size(); j++){
                ans.at(i).at(j) += b.at(j);
            }
        }
        return ans;
    }
};

#endif //DEEPLEARNING_BACKPROPAGATION_H
