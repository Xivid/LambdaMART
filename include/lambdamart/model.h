#ifndef LAMBDAMART_MODEL_H
#define LAMBDAMART_MODEL_H

#include "dataset.h"
#include "config.h"
#include "treelearner.h"
#include "lambdarank.h"

namespace LambdaMART {
    class Model {
        std::vector<Tree*>        trees;
        std::vector<double>       tree_weights;
        uint64_t                  num_samples;

        Tree* train_one_iteration(
                const LambdaRank& calculator,
                const std::vector<double>& currentScores,
                const LambdaMART::Dataset& dataset,
                std::vector<double>& gradients,
                std::vector<double>& hessians,
                const LambdaMART::Config& param);

    public:
        void                 train(const LambdaMART::Dataset& dataset, const LambdaMART::Config& param);
        std::vector<double>* predict(Dataset* data);

    };
}
#endif //LAMBDAMART_MODEL_H