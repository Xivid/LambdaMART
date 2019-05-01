#ifndef LAMBDAMART_TREELEARNER_H
#define LAMBDAMART_TREELEARNER_H

#include <lambdamart/types.h>
#include <lambdamart/config.h>
#include <lambdamart/dataset.h>
#include <lambdamart/histogram.h>

namespace LambdaMART {

class TreeNode {
    friend class TreeLearner;
    friend class Model;
    TreeNode() = default;
    TreeNode(double _t, TreeNode* _l, TreeNode* _r, SplitInfo _s): threshold(_t), left(_l), right(_r), splitInfo(_s) {}

private:
    double threshold;
    TreeNode *left, *right;
    SplitInfo splitInfo;
    double predict_score(LambdaMART::Dataset* data, sample_t idx);
};
typedef TreeNode Tree;


class TreeLearner {
    friend class Booster;

    Tree* build_new_tree(const LambdaMART::Dataset& dataset,
                         const std::vector<double>& gradients,
                         const std::vector<double>& hessians,
                         std::vector<Histogram>&    histograms,
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

};

}



#endif //LAMBDAMART_TREELEARNER_H
