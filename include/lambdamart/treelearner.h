#ifndef LAMBDAMART_TREELEARNER_H
#define LAMBDAMART_TREELEARNER_H

#include <lambdamart/types.h>
#include <lambdamart/config.h>
#include <lambdamart/dataset.h>
#include <lambdamart/histogram.h>
#include <queue>

namespace LambdaMART {

// TODO: make it a array-based data structure Tree, without using TreeNodes and pointers
class TreeNode {
    friend class TreeLearner;
    friend class Model;

    explicit TreeNode(nodeidx_t id) :
        id(id), output(0), impurity(0), is_leaf(true),
        split(nullptr), left_child(nullptr), right_child(nullptr) {}

    TreeNode(nodeidx_t id, score_t output, score_t impurity, bool isLeaf) :
        id(id), output(output), impurity(impurity), is_leaf(isLeaf),
        split(nullptr), left_child(nullptr), right_child(nullptr) {}

private:
    nodeidx_t id;  // root node is 1, left child of x is (2x), right child of x is (2x+1)
    score_t output;
    score_t impurity;
    bool is_leaf;
    Split* split;
    TreeNode* left_child;
    TreeNode* right_child;

    std::string toString(const std::string& prefix = "")
    {
        return prefix + "id = " + std::to_string(id) + ", output = " + std::to_string(output) + ", impurity = " + std::to_string(impurity) + ", is_leaf = " + std::to_string(is_leaf)
               + ", split = " + (split != nullptr ? split->toString() : "none") + ", left_child = " + std::to_string(left_child != nullptr ? left_child->id : 0) + ", right_child = " + std::to_string(right_child != nullptr ? right_child->id : 0)
               + (is_leaf ? "\n" : ("\n" + left_child->toString(prefix + "  ") + right_child->toString(prefix + "  ")));
    }

    score_t predict_score(const std::vector<featval_t> &features)
    {
        TreeNode* p = this;
        while (!p->is_leaf) {
            if (features[p->split->feature] <= p->split->threshold) {
                p = p->left_child;
            } else {
                p = p->right_child;
            }
        }
        return p->output;
    }

    uint32_t get_level() {
        return static_cast<uint32_t>(std::ceil(std::log2(id+1)));
    }

    nodeidx_t get_left_child_index() {
        return id << 1;
    }

    nodeidx_t get_right_child_index() {
        return (id << 1) + 1;
    }


    /*
    // TODO: utility functions
    static TreeNode* newEmptyNode(nodeidx_t nodeIndex)
    {
        // make it leaf by default
        return new TreeNode(nodeIndex, std::numeric_limits<score_t>::min(), -1.0, false);
    }

    static uint32_t indexToLevel(nodeidx_t nodeIndex)
    {
        return static_cast<uint32_t>(std::floor(std::log2(nodeIndex)));
    }

    static bool isLeftChild(nodeidx_t nodeIndex)
    {
        return (nodeIndex > 1 && (nodeIndex % 2 == 0));
    }

    static nodeidx_t leftChildIndex(nodeidx_t nodeIndex)
    {
        return (nodeIndex << 1);
    }

    static nodeidx_t rightChildIndex(nodeidx_t nodeIndex)
    {
        return (nodeIndex << 1) + 1;
    }

    static nodeidx_t parentIndex(nodeidx_t nodeIndex)
    {
        return (nodeIndex >> 1);
    }

    static nodeidx_t siblingIndex(nodeidx_t nodeIndex)
    {
        return isLeftChild(nodeIndex) ? nodeIndex + 1 : nodeIndex - 1;
    }

    nodeidx_t numDescendants()
    {
        return is_leaf ? 0 : (2 + left_child->numDescendants() + right_child->numDescendants());
    }

    nodeidx_t internalNodes()
    {
        // DEBUG_ASSERT_EX(is_leaf || (left_child != nullptr && right_child != nullptr), "%u internalNodes counting: no leftnode or rightnode!", id);
        return is_leaf ? 0 : (1 + left_child->internalNodes() + right_child->internalNodes());
    }
    */
};
typedef TreeNode Tree;


class TreeLearner {
    friend class Booster;

public:
    TreeLearner() = delete;
    TreeLearner(const Dataset* _dataset, const double* _gradients, const double* _hessians, const Config* _config) :
        dataset(_dataset), gradients(_gradients), hessians(_hessians), config(_config)
    {

        tie(num_samples, num_features) = dataset->shape();
        num_feature_blocking = config->num_feature_blocking;
        max_splits = config->max_splits;
        min_data_in_leaf = config->min_data_in_leaf;
        node_to_output.resize(1<<(config->max_depth));
        sample_to_node.resize(num_samples, 0);
        node_to_candidate.resize(1<<(config->max_depth));
        histograms.init(config->max_splits*num_feature_blocking, config->max_bin);
    }

    // perf
    int64_t sum_cycles_cumulate = 0;
    int64_t sum_cycles_getbestsplits = 0;

private:
    struct SplitCandidate
    {
        TreeNode* node;
        NodeStats* info;
        nodeidx_t smallerSibling;  // id

        SplitCandidate() = delete;
        SplitCandidate(TreeNode* n, NodeStats* i) : node(n), info(i), smallerSibling(0) {}
        SplitCandidate(TreeNode* n, NodeStats* i, nodeidx_t s) : node(n), info(i), smallerSibling(s) {}

        bool operator<(const SplitCandidate& rhs) const
        { // TODO: make smallerSibling == 0 top priority
            return node->impurity < rhs.node->impurity;
        }
    };

    struct CmpCandidates
    {
        bool operator()(const SplitCandidate* lhs, const SplitCandidate* rhs) const
        {
            return *lhs < *rhs;
        }
    };

    // as input
    const Config*             config;
    const Dataset*            dataset;
    const double*             gradients;
    const double*             hessians;

    // as working set
    sample_t                            num_samples;
    feature_t                           num_features;
    feature_t                           num_feature_blocking;
    HistogramMatrixTrans                histograms;
    uint32_t                            cur_depth = 0;
    std::vector<SplitInfo>              best_splits;
    size_t                              max_splits;
    sample_t                            min_data_in_leaf;
    std::vector<double>                 node_to_output;
    std::vector<unsigned int>           sample_to_node;
    std::vector<SplitCandidate*>        split_candidates;
    std::vector<NodeStats*>             node_info;
    std::vector<int>                    node_to_candidate;
    std::vector<int>                    sample_to_candidate;  // -1: this sample doesn't exist in any candidate node
    nodeidx_t                           num_candidates = 0;
    std::priority_queue<SplitCandidate*, std::vector<SplitCandidate*>, CmpCandidates> node_queue;

    // tree building methods
    Tree*  build_new_tree();
    bool   select_split_candidates();
    void   find_best_splits();
    void   perform_split();
    double get_sample_score(sample_t sid) { return node_to_output[sample_to_node[sid]]; };

};

}



#endif //LAMBDAMART_TREELEARNER_H
