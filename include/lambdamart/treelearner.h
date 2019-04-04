#ifndef LAMBDAMART_TREELEARNER_H
#define LAMBDAMART_TREELEARNER_H

#include <lambdamart/dataset.h>
#include <lambdamart/config.h>

namespace LambdaMART {

class TreeNode {
    friend class TreeLearner;
private:
    double threshold;
    TreeNode *left, *right;
};
typedef TreeNode Tree;

/**
 * This class contains necessary information used in building a new decision tree.
 */
class TreeLearner {

    // TODO
};

Tree* build_new_tree(const LambdaMART::Dataset& dataset,
                     const std::vector<double>& gradients,
                     const std::vector<double>& hessians,
                     const LambdaMART::Config& param);

}

#endif //LAMBDAMART_TREELEARNER_H
