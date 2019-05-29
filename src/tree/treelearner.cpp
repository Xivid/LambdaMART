#include <lambdamart/treelearner.h>
#include <numeric>
#include <immintrin.h>

namespace LambdaMART {


Tree* TreeLearner::build_new_tree()
{
    std::fill(sample_to_node.begin(), sample_to_node.end(), 1);
    std::fill(node_to_output.begin(), node_to_output.end(), 0.0);
    best_splits.clear();
    split_candidates.clear();
    histograms.clear();

    Tree* root = new TreeNode(1);
    auto* topInfo = new NodeStats(num_samples, 0.0);  // sum_gradients is always 0 for root node
    node_queue.push(new SplitCandidate(root, topInfo));
    LOG_DEBUG("build_new_tree: initialized");

    cur_depth = 1;
    while (cur_depth <= config->max_depth) {  // TODO: check num_leaves
        if(!select_split_candidates()) break;  // no more nodes to split -> break
        find_best_splits();
        perform_split();
    }

    // mark remaining candidates as leaves
    while (!node_queue.empty()) {
        SplitCandidate* candidate = node_queue.top();
        node_queue.pop();
        candidate->node->is_leaf = true;
    }

    return root;
}

bool TreeLearner::select_split_candidates() {
    LOG_DEBUG("select_split_candidates");

    std::fill(node_to_candidate.begin(), node_to_candidate.end(), -1);
    sample_to_candidate.clear();
    sample_to_candidate.resize(num_samples);
    split_candidates.clear();
    node_info.clear();

    num_candidates = 0;
    while (!node_queue.empty() && num_candidates < max_splits)
    {
        auto* candidate = node_queue.top();
        node_queue.pop();
        split_candidates.push_back(candidate);
        node_info.push_back(candidate->info);
        node_to_candidate[candidate->node->id] = num_candidates++;
    }

    if (!num_candidates) return false;

    for (sample_t sample = 0; sample < num_samples; ++sample) {
        sample_to_candidate[sample] = node_to_candidate[sample_to_node[sample]];
    }
    return true;
}


/**
 * Find best splits and put them into `best_splits'
 */
void TreeLearner::find_best_splits() {
    LOG_DEBUG("find_best_splits");

    best_splits.clear();
    best_splits.resize(num_candidates);
    LOG_TRACE("%lu", num_feature_blocking);

    feature_t fid;
    const feature_t feature_rest = num_features % num_feature_blocking;

    sample_t sample_idx;
    const sample_t sample_rest = num_samples % num_sample_blocking;

    double* const hist_base_addr = (double*) (&histograms[0][0]);
    const int size_hist_one_row = 255*16; //sizeof(histograms[0]);
    const __m256i hist_offset = _mm256_set1_epi64x(size_hist_one_row);

    const int size_hist_one_bin = sizeof(histograms[0][0]);
    const __m256i bin_size_offset = _mm256_set1_epi64x(size_hist_one_bin);

    const __m256d ones = _mm256_set1_pd(1);
    const __m256i eights = _mm256_set1_epi64x(8);

    const __m256i upper_MASK_STORE = _mm256_set_epi64x(-1, -1, 0, 0);
    const __m256i lower_MASK_STORE = _mm256_set_epi64x(0, 0, -1, -1);
    const __m256i num_feat_block_vec = _mm256_set1_epi64x(num_feature_blocking);
    const __m256i feature_const_offset = _mm256_set_epi64x(3, 2, 1, 0);
    const int i1 = 0, i3 = 2, i2 = 1, i4 = 3;

    for (fid = 0; fid < num_features - feature_rest; fid += num_feature_blocking) {
        LOG_DEBUG("checking feature [%lu, %lu)", fid, fid+num_feature_blocking);
        histograms.clear(num_candidates * num_feature_blocking);
        const Feature &feat0 = dataset->get_data(fid);
        const Feature &feat1 = dataset->get_data(fid+1);
        const Feature &feat2 = dataset->get_data(fid+2);
        const Feature &feat3 = dataset->get_data(fid+3);

        for (sample_idx = 0; sample_idx < num_samples - sample_rest; sample_idx ++)
        {
            const int candidate = sample_to_candidate[sample_idx];
            if (candidate != -1) {
                const int bin0 = feat0.bin_index[sample_idx];
                const int bin1 = feat1.bin_index[sample_idx];
                const int bin2 = feat2.bin_index[sample_idx];
                const int bin3 = feat3.bin_index[sample_idx];

                const __m256i candidate_vec = _mm256_set1_epi64x(candidate);
                const __m256i bin_idx_vec = _mm256_set_epi64x(bin3, bin2, bin1, bin0);
                const __m256i simple_offset = _mm256_mul_epi32(candidate_vec, num_feat_block_vec);
                const __m256i net_offset = _mm256_add_epi64(simple_offset, feature_const_offset);

                const __m256i candidate_off = _mm256_mul_epi32(net_offset, hist_offset);
                const __m256i bin_off = _mm256_mul_epi32(bin_idx_vec, bin_size_offset);
                const __m256i total_offset = _mm256_add_epi64(candidate_off, bin_off);

                const __m256d gradients_vec = _mm256_set1_pd(gradients[sample_idx]);

                const __m256i sum_counts_ptr = total_offset;
                const __m256i sum_gradients_ptr = _mm256_add_epi64(total_offset, eights);

                const __m256d sum_counts_val = _mm256_i64gather_pd(hist_base_addr, sum_counts_ptr, 1);
                const __m256d sum_gradients_val = _mm256_i64gather_pd(hist_base_addr, sum_gradients_ptr, 1);

                const __m256d new_sum_counts_val = _mm256_add_pd(sum_counts_val, ones);
                const __m256d new_sum_gradients_val = _mm256_add_pd(sum_gradients_val, gradients_vec);

                const __m256d c1g1c3g3 = _mm256_shuffle_pd(new_sum_counts_val, new_sum_gradients_val, 0);
                const __m256d c2g2c4g4 = _mm256_shuffle_pd(new_sum_counts_val, new_sum_gradients_val, 0xF);

                int offset = _mm256_extract_epi64(total_offset, i1);
                double* a = hist_base_addr + offset/8;
                _mm256_maskstore_pd(a, lower_MASK_STORE, c1g1c3g3);

                offset = _mm256_extract_epi64(total_offset, i3);
                a = hist_base_addr + offset/8 - 2;
                _mm256_maskstore_pd(a, upper_MASK_STORE, c1g1c3g3);

                offset = _mm256_extract_epi64(total_offset, i2);
                a = hist_base_addr + offset/8;
                _mm256_maskstore_pd(a, lower_MASK_STORE, c2g2c4g4);

                 offset = _mm256_extract_epi64(total_offset, i4);
                a = hist_base_addr + offset/8 - 2;
                _mm256_maskstore_pd(a, upper_MASK_STORE, c2g2c4g4);
            }
        }
        for (; sample_idx < num_samples; ++sample_idx)
        {
            const int candidate = sample_to_candidate[sample_idx];
            if (candidate != -1) {
                const bin_t bin0 = feat0.bin_index[sample_idx];
                const bin_t bin1 = feat1.bin_index[sample_idx];
                const bin_t bin2 = feat2.bin_index[sample_idx];
                const bin_t bin3 = feat3.bin_index[sample_idx];

                const gradient_t grad = gradients[sample_idx];

                histograms[candidate*num_feature_blocking][bin0].update(1.0, grad);
                histograms[candidate*num_feature_blocking+1][bin1].update(1.0, grad);
                histograms[candidate*num_feature_blocking+2][bin2].update(1.0, grad);
                histograms[candidate*num_feature_blocking+3][bin3].update(1.0, grad);
            }
        }

        for (nodeidx_t candidate = 0; candidate < num_candidates; ++candidate) {
            histograms.cumulate(candidate*num_feature_blocking);
            histograms.cumulate(candidate*num_feature_blocking+1);
            histograms.cumulate(candidate*num_feature_blocking+2);
            histograms.cumulate(candidate*num_feature_blocking+3);

            auto local_best0 = histograms.get_best_split(candidate*num_feature_blocking, fid, feat0, node_info[candidate], min_data_in_leaf);
            auto local_best1 = histograms.get_best_split(candidate*num_feature_blocking+1, fid+1, feat1, node_info[candidate], min_data_in_leaf);
            auto local_best2 = histograms.get_best_split(candidate*num_feature_blocking+2, fid+2, feat2, node_info[candidate], min_data_in_leaf);
            auto local_best3 = histograms.get_best_split(candidate*num_feature_blocking+3, fid+3, feat3, node_info[candidate], min_data_in_leaf);

            if (local_best0 >= best_splits[candidate]) {
                best_splits[candidate] = local_best0;
            }
            if (local_best1 >= best_splits[candidate]) {
                best_splits[candidate] = local_best1;
            }
            if (local_best2 >= best_splits[candidate]) {
                best_splits[candidate] = local_best2;
            }
            if (local_best3 >= best_splits[candidate]) {
                best_splits[candidate] = local_best3;
            }
        }
    }

    for (; fid < num_features; ++fid) {
        LOG_DEBUG("checking feature %lu", fid);
        histograms.clear(num_candidates);
        const Feature &feat = dataset->get_data(fid);

        //TODO: unrolling
        for (sample_idx = 0; sample_idx < num_samples - sample_rest; sample_idx += num_sample_blocking) {
            const int candidate0 = sample_to_candidate[sample_idx];
            const int candidate1 = sample_to_candidate[sample_idx+1];
            const int candidate2 = sample_to_candidate[sample_idx+2];
            const int candidate3 = sample_to_candidate[sample_idx+3];

            if (candidate0 != -1) {
                const bin_t bin = feat.bin_index[sample_idx];
                histograms[candidate0][bin].update(1.0, gradients[sample_idx]);
            }
            if (candidate1 != -1) {
                const bin_t bin = feat.bin_index[sample_idx+1];
                histograms[candidate1][bin].update(1.0, gradients[sample_idx+1]);
            }
            if (candidate2 != -1) {
                const bin_t bin = feat.bin_index[sample_idx+2];
                histograms[candidate2][bin].update(1.0, gradients[sample_idx+2]);
            }
            if (candidate3 != -1) {
                const bin_t bin = feat.bin_index[sample_idx+3];
                histograms[candidate3][bin].update(1.0, gradients[sample_idx+3]);
            }
        }
        for (; sample_idx < num_samples; ++sample_idx) {
            const int candidate = sample_to_candidate[sample_idx];
            if (candidate != -1) {
                const bin_t bin = feat.bin_index[sample_idx];
                histograms[candidate][bin].update(1.0, gradients[sample_idx]);
            }
        }

        for (nodeidx_t candidate = 0; candidate < num_candidates; ++candidate) {
            histograms.cumulate(candidate);
            auto local_best = histograms.get_best_split(candidate, fid, feat, node_info[candidate], min_data_in_leaf);
            if (local_best >= best_splits[candidate]) {
                best_splits[candidate] = local_best;
            }
        }
    }
}

void TreeLearner::perform_split()
{
    LOG_DEBUG("perform_split");
    //update node_to_output, sample_to_node
    //update cur_depth;

    double min_gain_to_split = config->min_gain_to_split;
    vector<bool> do_split(num_candidates, false);

    // determine which nodes to really split
    for (nodeidx_t candidate = 0; candidate < num_candidates; ++candidate) {
        if (best_splits[candidate].gain > min_gain_to_split) {
            do_split[candidate] = true;
            split_candidates[candidate]->node->is_leaf = false;
        } else {
            // do not split this candidate
            split_candidates[candidate]->node->is_leaf = true;
        }
    }

    // update sample_to_node for affected samples, and aggregate their lambdas and weights for children nodes
    for (sample_t sample = 0; sample < num_samples; ++sample) {
        int candidate = sample_to_candidate[sample];
        if (candidate < 0 || !do_split[candidate]) continue;

        const feature_t feature = best_splits[candidate].split->feature;
        const bin_t bin = best_splits[candidate].bin;
        // TODO: is there a way to optimize this 2-level random access?
        if (dataset->get_data(feature).bin_index[sample] <= bin) {
            sample_to_node[sample] <<= 1;  // move to left child
            best_splits[candidate].update_children_stats(gradients[sample] * gradients[sample], hessians[sample], 0., 0.);
        } else {
            sample_to_node[sample] = (sample_to_node[sample] << 1) + 1;  // move to right child
            best_splits[candidate].update_children_stats(0., 0., gradients[sample] * gradients[sample], hessians[sample]);
        }
    }

    // create the children nodes
    for (nodeidx_t candidate = 0; candidate < num_candidates; ++candidate) {
        LOG_TRACE("do_split[%d]: %s", candidate, do_split[candidate]?"true":"false");
        if (do_split[candidate]) {
            auto& splitInfo = best_splits[candidate];
            LOG_TRACE("split on Node %lu = Candidate %lu: %s",split_candidates[candidate]->node->id, candidate, splitInfo.toString().c_str());
            TreeNode* candidate_node = split_candidates[candidate]->node;
            candidate_node->split = splitInfo.split;
            score_t left_impurity = splitInfo.get_left_impurity();
            score_t right_impurity = splitInfo.get_right_impurity();
            score_t left_output = splitInfo.get_left_output();
            score_t right_output = splitInfo.get_right_output();

            // depth of children is still smaller than max_depth -> they can be further split
            bool child_is_leaf = (candidate_node->get_level() + 1 >= config->max_depth);
            bool left_child_is_leaf = (child_is_leaf || left_impurity < config->min_impurity_to_split);
            bool right_child_is_leaf = (child_is_leaf || right_impurity < config->min_impurity_to_split);
            LOG_TRACE(" child_is_leaf: %d\n\t\t  left_child_is_leaf: %d\n\t\t  right_child_is_leaf: %d",
                    child_is_leaf, left_child_is_leaf, right_child_is_leaf);

            TreeNode* left_child = new TreeNode(candidate_node->get_left_child_index(), left_output, left_impurity, left_child_is_leaf);
            TreeNode* right_child = new TreeNode(candidate_node->get_right_child_index(), right_output, right_impurity, right_child_is_leaf);
            candidate_node->left_child = left_child;
            candidate_node->right_child = right_child;
            node_to_output[left_child->id] = left_output;
            node_to_output[right_child->id] = right_output;

            if (!left_child_is_leaf) {
                node_queue.push(new SplitCandidate(left_child, splitInfo.left_stats));
            }
            if (!right_child_is_leaf) {
                node_queue.push(new SplitCandidate(right_child, splitInfo.right_stats));
            }

            cur_depth = std::max(left_child->get_level(), cur_depth);
        }
    }
}

}
