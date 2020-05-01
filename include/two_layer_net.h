#ifndef DEEPLEARNING_TWO_LAYER_NET_H
#define DEEPLEARNING_TWO_LAYER_NET_H

#include <cmath>
#include <vector>
#include <random>

class TwoLayerNet{
private:
    std::random_device seed;
    std::default_random_engine  engine;
    std::normal_distribution<> w1_init_dist, w2_init_dist;

    int input_size, hidden_size, output_size;
    std::vector<std::vector<double>> w1;
    std::vector<std::vector<double>> w2;
    std::vector<double> b1;
    std::vector<double> b2;

public:
    TwoLayerNet(int input_size, int hidden_size, int output_size):
            engine(seed()),
            w1_init_dist(0.0, std::sqrt(2.0 / input_size)),
            w2_init_dist(0.0, std::sqrt(2.0 / hidden_size)),
            input_size(input_size),
            hidden_size(hidden_size),
            output_size(output_size),
            w1(input_size, std::vector<double>(hidden_size)),
            w2(hidden_size, std::vector<double>(output_size)),
            b1(hidden_size),
            b2(output_size){
        init();
    }

private:
    void init(){
        for(int i = 0; i < input_size; i++){
            for(int j = 0; j < hidden_size; j++){
                w1.at(i).at(j) = w1_init_dist(engine);
            }
        }
        for(int i = 0; i < hidden_size; i++){
            for(int j = 0; j < output_size; j++){
                w2.at(i).at(j) = w2_init_dist(engine);
            }
        }
        return;
    }
};

#endif //DEEPLEARNING_TWO_LAYER_NET_H
