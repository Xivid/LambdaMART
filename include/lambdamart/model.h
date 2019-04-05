#ifndef LAMBDAMART_MODEL_H
#define LAMBDAMART_MODEL_H

#include <lambdamart/dataset.h>
#include <lambdamart/config.h>
#include <lambdamart/treelearner.h>
#include <lambdamart/lambdarank.h>

namespace LambdaMART {
    class Model {
        std::vector<Tree*>        trees;
        std::vector<double>       tree_weights;
        uint64_t                  num_samples;

    public:
        void                 train(const LambdaMART::Dataset& dataset, const LambdaMART::Config& config);
        std::vector<double>* predict(Dataset* data);

    };
}
#endif //LAMBDAMART_MODEL_H