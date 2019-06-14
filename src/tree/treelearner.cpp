#include <lambdamart/treelearner.h>
#include <numeric>
#include <lambdamart/perf.h>
#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
using namespace std;

namespace LambdaMART {
//    int64_t sum_cycles_update = 0;
//    int64_t sum_cycles_cumulate = 0;
//    int64_t sum_cycles_gbs = 0;
    uint64_t sample_count = 0;


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
        sample_count = 0;
        find_best_splits();
//        Log::Info("cur_depth, sample count, num cand: %d, %lld, %lld", cur_depth, sample_count, num_candidates);
        perform_split();
    }
//    Log::Info("update, cumulate, gbs, total cycles: %lld, %lld, %lld, %lld", sum_cycles_update,
//              sum_cycles_cumulate, sum_cycles_gbs, sum_cycles_update+sum_cycles_cumulate+sum_cycles_gbs);
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

    Bin* histograms_data = histograms.data();
    bin_t bin_cnt = histograms.bin_cnt;

    for (fid = 0; fid < num_features - feature_rest; fid += num_feature_blocking) {
        LOG_DEBUG("checking feature [%lu, %lu)", fid, fid+num_feature_blocking);
        histograms.clear(num_candidates * num_feature_blocking);
        const Feature &feat0 = dataset->get_data(fid);
        const Feature &feat1 = dataset->get_data(fid+1);
        const Feature &feat2 = dataset->get_data(fid+2);
        const Feature &feat3 = dataset->get_data(fid+3);

        const vector<bin_t> sample_to_bin0 = feat0.bin_index;
        const vector<bin_t> sample_to_bin1 = feat1.bin_index;
        const vector<bin_t> sample_to_bin2 = feat2.bin_index;
        const vector<bin_t> sample_to_bin3 = feat3.bin_index;

        cycles_count_start();
        for (sample_idx = 0; sample_idx < num_samples - sample_rest; sample_idx += num_sample_blocking)
        {
            const int candidate0 = sample_to_candidate[sample_idx];
            const int candidate1 = sample_to_candidate[sample_idx+1];
            const int candidate2 = sample_to_candidate[sample_idx+2];
            const int candidate3 = sample_to_candidate[sample_idx+3];

            if (candidate0 != -1) {
//                if (fid == 0) sample_count++;
                const bin_t bin0 = sample_to_bin0[sample_idx];
                const bin_t bin1 = sample_to_bin1[sample_idx];
                const bin_t bin2 = sample_to_bin2[sample_idx];
                const bin_t bin3 = sample_to_bin3[sample_idx];

                const gradient_t grad = gradients[sample_idx];

                Bin* base = histograms_data + candidate0 * num_feature_blocking * bin_cnt;

//                gradient_t *p = (gradient_t*) (base + bin0);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt + bin1);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));


                gradient_t *p = (gradient_t*) (base + bin0);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt + bin1);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
                *p += 1.0; *(p+1) += grad;

                //histograms[candidate0*num_feature_blocking][bin0].update(1.0, grad);
                //histograms[candidate0*num_feature_blocking+1][bin1].update(1.0, grad);
                //histograms[candidate0*num_feature_blocking+2][bin2].update(1.0, grad);
                //histograms[candidate0*num_feature_blocking+3][bin3].update(1.0, grad);
            }
            if (candidate1 != -1) {
//                if (fid == 0) sample_count++;
                const bin_t bin0 = sample_to_bin0[sample_idx+1];
                const bin_t bin1 = sample_to_bin1[sample_idx+1];
                const bin_t bin2 = sample_to_bin2[sample_idx+1];
                const bin_t bin3 = sample_to_bin3[sample_idx+1];

                const gradient_t grad = gradients[sample_idx+1];

                Bin* base = histograms_data + candidate1 * num_feature_blocking * bin_cnt;

//                gradient_t *p = (gradient_t*) (base + bin0);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt + bin1);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));


                gradient_t *p = (gradient_t*) (base + bin0);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt + bin1);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
                *p += 1.0; *(p+1) += grad;

                //histograms[candidate1*num_feature_blocking][bin0].update(1.0, grad);
                //histograms[candidate1*num_feature_blocking+1][bin1].update(1.0, grad);
                //histograms[candidate1*num_feature_blocking+2][bin2].update(1.0, grad);
                //histograms[candidate1*num_feature_blocking+3][bin3].update(1.0, grad);
            }
            if (candidate2 != -1) {
//                if (fid == 0) sample_count++;
                const bin_t bin0 = sample_to_bin0[sample_idx+2];
                const bin_t bin1 = sample_to_bin1[sample_idx+2];
                const bin_t bin2 = sample_to_bin2[sample_idx+2];
                const bin_t bin3 = sample_to_bin3[sample_idx+2];

                const gradient_t grad = gradients[sample_idx+2];

                Bin* base = histograms_data + candidate2 * num_feature_blocking * bin_cnt;

//                gradient_t *p = (gradient_t*) (base + bin0);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt + bin1);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));


                gradient_t *p = (gradient_t*) (base + bin0);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt + bin1);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
                *p += 1.0; *(p+1) += grad;

                //histograms[candidate2*num_feature_blocking][bin0].update(1.0, grad);
                //histograms[candidate2*num_feature_blocking+1][bin1].update(1.0, grad);
                //histograms[candidate2*num_feature_blocking+2][bin2].update(1.0, grad);
                //histograms[candidate2*num_feature_blocking+3][bin3].update(1.0, grad);
            }
            if (candidate3 != -1) {
//                if (fid == 0) sample_count++;
                const bin_t bin0 = sample_to_bin0[sample_idx+3];
                const bin_t bin1 = sample_to_bin1[sample_idx+3];
                const bin_t bin2 = sample_to_bin2[sample_idx+3];
                const bin_t bin3 = sample_to_bin3[sample_idx+3];

                const gradient_t grad = gradients[sample_idx+3];

                Bin* base = histograms_data + candidate3 * num_feature_blocking * bin_cnt;

//                gradient_t *p = (gradient_t*) (base + bin0);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt + bin1);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));


                gradient_t *p = (gradient_t*) (base + bin0);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt + bin1);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
                *p += 1.0; *(p+1) += grad;

                //histograms[candidate3*num_feature_blocking][bin0].update(1.0, grad);
                //histograms[candidate3*num_feature_blocking+1][bin1].update(1.0, grad);
                //histograms[candidate3*num_feature_blocking+2][bin2].update(1.0, grad);
                //histograms[candidate3*num_feature_blocking+3][bin3].update(1.0, grad);
            }
        }
        for (; sample_idx < num_samples; ++sample_idx)
        {
            const int candidate = sample_to_candidate[sample_idx];
            if (candidate != -1) {
                if (fid == 0) sample_count++;
                const bin_t bin0 = sample_to_bin0[sample_idx];
                const bin_t bin1 = sample_to_bin1[sample_idx];
                const bin_t bin2 = sample_to_bin2[sample_idx];
                const bin_t bin3 = sample_to_bin3[sample_idx];

                const gradient_t grad = gradients[sample_idx];

                Bin* base = histograms_data + candidate * num_feature_blocking * bin_cnt;

//                gradient_t *p = (gradient_t*) (base + bin0);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt + bin1);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));
//                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
//                a = _mm_set_ps(0.0f, 0.0f, grad, 1.0);
//                b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));



                gradient_t *p = (gradient_t*) (base + bin0);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt + bin1);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 2 + bin2);
                *p += 1.0; *(p+1) += grad;
                p = (gradient_t*) (base + bin_cnt * 3 + bin3);
                *p += 1.0; *(p+1) += grad;

                //histograms[candidate*num_feature_blocking][bin0].update(1.0, grad);
                //histograms[candidate*num_feature_blocking+1][bin1].update(1.0, grad);
                //histograms[candidate*num_feature_blocking+2][bin2].update(1.0, grad);
                //histograms[candidate*num_feature_blocking+3][bin3].update(1.0, grad);
            }
        }
        sum_cycles_update += cycles_count_stop();

        cycles_count_start();
        histograms.cumulate(num_candidates*num_feature_blocking);
        sum_cycles_cumulate += cycles_count_stop();

        for (nodeidx_t candidate = 0; candidate < num_candidates; ++candidate) {

            cycles_count_start();
            auto local_best0 = histograms.get_best_split(candidate*num_feature_blocking, fid, feat0, node_info[candidate], min_data_in_leaf);
            auto local_best1 = histograms.get_best_split(candidate*num_feature_blocking+1, fid+1, feat1, node_info[candidate], min_data_in_leaf);
            auto local_best2 = histograms.get_best_split(candidate*num_feature_blocking+2, fid+2, feat2, node_info[candidate], min_data_in_leaf);
            auto local_best3 = histograms.get_best_split(candidate*num_feature_blocking+3, fid+3, feat3, node_info[candidate], min_data_in_leaf);
            sum_cycles_gbs += cycles_count_stop();

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
        const vector<bin_t>& sample_to_bin = feat.bin_index;

        cycles_count_start();

        for (sample_idx = 0; sample_idx < num_samples - sample_rest; sample_idx += num_sample_blocking) {
            const int candidate0 = sample_to_candidate[sample_idx];
            const int candidate1 = sample_to_candidate[sample_idx+1];
            const int candidate2 = sample_to_candidate[sample_idx+2];
            const int candidate3 = sample_to_candidate[sample_idx+3];

            if (candidate0 != -1) {
                if (fid == 0) sample_count++;
                const bin_t bin = sample_to_bin[sample_idx];
                gradient_t *p = (gradient_t*) (histograms_data + candidate0 * bin_cnt + bin);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, gradients[sample_idx], 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));


                *p += 1.0;
                *(p+1) += gradients[sample_idx];

                //histograms[candidate0][bin].update(1.0, gradients[sample_idx]);
            }
            if (candidate1 != -1) {
                if (fid == 0) sample_count++;
                const bin_t bin = sample_to_bin[sample_idx+1];
                gradient_t *p = (gradient_t*) (histograms_data + candidate1 * bin_cnt + bin);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, gradients[sample_idx], 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));

                *p += 1.0;
                *(p+1) += gradients[sample_idx];
                //histograms[candidate1][bin].update(1.0, gradients[sample_idx+1]);
            }
            if (candidate2 != -1) {
//                if (fid == 0) sample_count++;
                const bin_t bin = sample_to_bin[sample_idx+2];
                gradient_t *p = (gradient_t*) (histograms_data + candidate2 * bin_cnt + bin);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, gradients[sample_idx], 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));

                *p += 1.0;
                *(p+1) += gradients[sample_idx];

                //histograms[candidate2][bin].update(1.0, gradients[sample_idx+2]);
            }
            if (candidate3 != -1) {
//                if (fid == 0) sample_count++;
                const bin_t bin = sample_to_bin[sample_idx+3];
                gradient_t *p = (gradient_t*) (histograms_data + candidate3 * bin_cnt + bin);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, gradients[sample_idx], 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));

                *p += 1.0;
                *(p+1) += gradients[sample_idx];

                //histograms[candidate3][bin].update(1.0, gradients[sample_idx+3]);
            }
        }
        for (; sample_idx < num_samples; ++sample_idx) {
            const int candidate = sample_to_candidate[sample_idx];
            if (candidate != -1) {
                if (fid == 0) sample_count++;
                const bin_t bin = sample_to_bin[sample_idx];
                gradient_t *p = (gradient_t*) (histograms_data + candidate * bin_cnt + bin);
//                __m128 a = _mm_set_ps(0.0f, 0.0f, gradients[sample_idx], 1.0);
//                __m128 b = _mm_load_ps(p);
//                _mm_store_ps(p, _mm_add_ps(a, b));

                *p += 1.0;
                *(p+1) += gradients[sample_idx];

                //histograms[candidate][bin].update(1.0, gradients[sample_idx]);
            }
        }
        sum_cycles_update += cycles_count_stop();

        cycles_count_start();
        histograms.cumulate(num_candidates);
        sum_cycles_cumulate += cycles_count_stop();

        for (nodeidx_t candidate = 0; candidate < num_candidates; ++candidate) {
            cycles_count_start();
            auto local_best = histograms.get_best_split(candidate, fid, feat, node_info[candidate], min_data_in_leaf);
            sum_cycles_gbs += cycles_count_stop();

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

    score_t min_gain_to_split = config->min_gain_to_split;
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
