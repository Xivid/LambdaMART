#ifndef LAMBDAMART_MODEL_H
#define LAMBDAMART_MODEL_H

#include <lambdamart/dataset.h>
#include <lambdamart/config.h>
#include <lambdamart/treelearner.h>
#include <lambdamart/lambdarank.h>
#include <lambdamart/types.h>

namespace LambdaMART {
    class Model {
        friend class Booster;

        std::vector<Tree*>  trees;
        std::vector<double> tree_weights;

        void add_tree(Tree* tree, double tree_weight) { trees.push_back(tree); tree_weights.push_back(tree_weight); }
    public:
        double* predict(Dataset* data);
    };
}
#endif //LAMBDAMART_MODEL_H