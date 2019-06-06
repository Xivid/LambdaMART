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
        std::vector<float> tree_weights;

        void add_tree(Tree* tree, float tree_weight) { trees.push_back(tree); tree_weights.push_back(tree_weight); }
    public:
        vector<score_t> predict(RawDataset* data, const string& output_path);
        vector<score_t> predict(RawDataset* data);
    };
}
#endif //LAMBDAMART_MODEL_H
