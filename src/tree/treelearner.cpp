#include <lambdamart/treelearner.h>
#include <numeric>

namespace LambdaMART {


Tree* TreeLearner::build_new_tree()
{
    Tree* root = new TreeNode(1);
    auto* topInfo = new NodeInfoStats(num_samples, std::accumulate(gradients, gradients + num_samples, 0.0));
    node_queue.emplace(root, topInfo);

    this->cur_depth = 1;
    while (cur_depth < config->max_depth) {
        num_nodes_to_split = select_split_candidates();
        if (num_nodes_to_split == 0) break;

        update_candidate_tracker();

        std::vector<SplitInfo>* best_splits = find_best_splits();
        perform_split(*best_splits, node_to_score, sample_to_node);
    }

    return root;
}

int TreeLearner::select_split_candidates() {
    nodeidx_t num_candidates = 0;
//    while (!node_queue.empty() && num_candidates < max_splits)
//    {
//        auto& candidate = node_queue.top();
//        node_queue.pop();
//        split_candidates.push_back(candidate);
//        nodeId2NodeNo[candidate.node->id] = 0;
//
//        ++numNodes;
//        ++curLeaves;  // TODO(99): Is this nodes and leaves counting method correct?
//    }
    return num_candidates;
}

void TreeLearner::update_candidate_tracker() {

}

std::vector<SplitInfo>* TreeLearner::find_best_splits()
{
    histograms.resize(num_nodes_to_split);

    //TODO: input histograms?
    /*
     * for each feature in dataset
     *   for each sample in [0, dataset->num_samples)
     *     histograms[sample_to_node[sample]][feature[sample]].update(gradients[sample], hessians[sample]);
     */

    /*
     * for each histogram in histograms
     *   best_splits[histogram.node_id] = histogram.best_split_point()
     */
    return nullptr;
}

nodeidx_t TreeLearner::perform_split(const std::vector<SplitInfo>& best_splits,
                     std::vector<double>&          node_to_score,
                     std::vector<unsigned int>&    sample_to_node)
{
    //update node_to_score, sample_to_node
    ++cur_depth;
    return 100;
}

}
