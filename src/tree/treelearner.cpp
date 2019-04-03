#include <lambdamart/treelearner.h>

namespace LambdaMART {

Tree* build_new_tree(const LambdaMART::Dataset& dataset,
                     const std::vector<double>& gradients,
                     const std::vector<double>& hessians,
                     const LambdaMART::Config& param)
{
    TreeLearner* learner = new TreeLearner();

    return nullptr;
}

}
