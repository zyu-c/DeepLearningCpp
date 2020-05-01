#ifndef DEEPLEARNING_VECTOR_FUNCTION_H
#define DEEPLEARNING_VECTOR_FUNCTION_H

#include <vector>

std::vector<std::vector<double>> multiplication(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y){
    int a = x.size();
    int b = x.at(0).size();
    int c = y.at(0).size();
    std::vector<std::vector<double>> ans(a, std::vector<double>(c));

    if(b != y.size()){
        return ans;
    }

    for(int i = 0; i < a; i++){
        for(int j = 0; j < c; j++){
            for(int k = 0; k < b; k++){
                ans.at(i).at(j) += x.at(i).at(k) * y.at(k).at(j);
            }
        }
    }

    return ans;
}

#endif //DEEPLEARNING_VECTOR_FUNCTION_H
