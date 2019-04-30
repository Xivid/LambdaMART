#include <lambdamart/treelearner.h>

namespace LambdaMART {

Tree* build_new_tree(const LambdaMART::Dataset& dataset,
                     const std::vector<double>& gradients,
                     const std::vector<double>& hessians,
                     std::vector<Histogram>&    histograms,
                     std::vector<double>&       node_to_score,
                     std::vector<unsigned int>& sample_to_node,
                     const LambdaMART::Config&  config)
{
    TreeNode* root = new TreeNode();
    std::vector<SplitInfo> best_splits;
    node_t num_nodes_to_split = 1;

    for (int depth = 1; depth <= config.max_depth; ++depth) {
        histograms.resize(num_nodes_to_split);
        find_best_splits(best_splits, dataset, gradients, hessians, sample_to_node);
        num_nodes_to_split = perform_split(best_splits, node_to_score, sample_to_node);
    }

    return nullptr;
}

void find_best_splits(std::vector<SplitInfo>& best_splits,
                      const LambdaMART::Dataset& dataset,
                      const std::vector<double>& gradients,
                      const std::vector<double>& hessians,
                      std::vector<unsigned int>& sample_to_node)
{
    /*
     * for each feature in dataset
     *   for each sample in [0, dataset->num_samples)
     *     histograms[sample_to_node[sample]][feature[sample]].update(gradients[sample], hessians[sample]);
     */

    /*
     * for each histogram in histograms
     *   best_splits[histogram.node_id] = histogram.best_split_point()
     */
}

node_t perform_split(const std::vector<SplitInfo>& best_splits,
                     std::vector<double>&          node_to_score,
                     std::vector<unsigned int>&    sample_to_node)
{
    //update node_to_score, sample_to_node
    return 100;
}

// TODO
// Given data, index of data, and treenode, returns the score for this data
double predict_score(LambdaMART::Dataset* data, sample_t idx, LambdaMART::Tree* model) {
    return 0.0;    
}


}
