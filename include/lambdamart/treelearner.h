#ifndef LAMBDAMART_TREELEARNER_H
#define LAMBDAMART_TREELEARNER_H

#include <lambdamart/types.h>
#include <lambdamart/config.h>
#include <lambdamart/dataset.h>
#include <lambdamart/histogram.h>

namespace LambdaMART {

class SplitInfo {
    feature_t feature_id;
    bin_t     bin_id;
};


class TreeNode {
    friend class TreeLearner;
private:
    double threshold;
    TreeNode *left, *right;
    SplitInfo splitInfo;
};
typedef TreeNode Tree;


Tree* build_new_tree(const LambdaMART::Dataset& dataset,
                     const std::vector<double>& gradients,
                     const std::vector<double>& hessians,
                     std::vector<double>&       node_to_score,
                     std::vector<unsigned int>& sample_to_node,
                     const LambdaMART::Config&  config);

void find_best_splits(std::vector<SplitInfo>& best_splits,
                      const LambdaMART::Dataset& dataset,
                      const std::vector<double>& gradients,
                      const std::vector<double>& hessians,
                      std::vector<unsigned int>& sample_to_node);


node_t perform_split(const std::vector<SplitInfo>& best_splits,
                     std::vector<double>&          node_to_score,
                     std::vector<unsigned int>&    sample_to_node);

double predict_score(Dataset* data, datasize_t idx, Tree* model);


}



#endif //LAMBDAMART_TREELEARNER_H
