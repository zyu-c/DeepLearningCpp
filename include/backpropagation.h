#ifndef DEEPLEARNING_BACKPROPAGATION_H
#define DEEPLEARNING_BACKPROPAGATION_H

#include <vector>
#include "vector_function.h"

class Affine{
private:
    std::vector<std::vector<double>> w;
    std::vector<double> b;

    std::vector<std::vector<double>> dw;
    std::vector<double> db;

    std::vector<std::vector<double>> x;

public:
    Affine(int input_size, int output_size):
            w(input_size, std::vector<double>(output_size)),
            b(output_size),
            dw(input_size, std::vector<double>(output_size)),
            db(output_size){
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

    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> x){
        this->x = x;
        std::vector<std::vector<double>> ans;
        ans = multiplication(x, w);
        for(int i = 0; i < w.at(0).size(); i++){
            for(int j = 0; j < b.size(); j++){
                ans.at(i).at(j) += b.at(j);
            }
        }
        return ans;
    }

    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> dout){
        std::vector<std::vector<double>> xt(x.at(0).size(), std::vector<double>(x.size()));
        for(int i = 0; i < x.size(); i++){
            for(int j = 0; j < x.at(0).size(); j++){
                xt.at(j).at(i) = x.at(i).at(j);
            }
        }
        dw = multiplication(xt, dout);

        for(int i = 0; i < db.size(); i++){
            db.at(i) = 0.0;
            for(int j = 0; j < dout.size(); j++){
                db.at(i) += dout.at(j).at(i);
            }
        }

        std::vector<std::vector<double>> wt(w.at(0).size(), std::vector<double>(w.size()));
        for(int i = 0; i < w.size(); i++){
            for(int j = 0; j < w.at(0).size(); j++){
                wt.at(j).at(i) = w.at(i).at(j);
            }
        }
        return multiplication(dout, wt);
    }
};

class Relu{
private:
    std::vector<std::vector<double>> x;

public:
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> x){
        this->x = x;
        std::vector<std::vector<double>> ans(x.size(), std::vector<double>(x.at(0).size()));
        for(int i = 0; i < x.size(); i++){
            for(int j = 0; j < x.at(0).size(); j++){
                if(x.at(i).at(j) > 0){
                    ans.at(i).at(j) = x.at(i).at(j);
                }else{
                    ans.at(i).at(j) = 0.0;
                }
            }
        }
        return ans;
    }
};

#endif //DEEPLEARNING_BACKPROPAGATION_H
