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
        for (feature_t fid = 0; fid < num_features; ++fid) {
            LOG_TRACE("checking feature %lu", fid);
            histograms.clear(num_candidates);
            const Feature &feat = dataset->get_data()[fid];

            double* const hist_base_addr = (double*) (&histograms[0][0]);

            const int size_hist_one_row = 255*16; //sizeof(histograms[0]);
            const __m256i hist_offset = _mm256_set1_epi64x(size_hist_one_row);

            const int size_hist_one_bin = sizeof(histograms[0][0]);
            const __m256i bin_size_offset = _mm256_set1_epi64x(size_hist_one_bin);

            const __m256i mones = _mm256_set1_epi64x(-1);
            const __m256d zeros = _mm256_setzero_pd();
            const __m256d ones = _mm256_set1_pd(1);
            const __m256i fours = _mm256_set1_epi64x(4);
            const __m256i eights = _mm256_set1_epi64x(8);

            const __m256i upper_MASK_STORE = _mm256_set_epi64x(-1, -1, 0, 0);
            const __m256i lower_MASK_STORE = _mm256_set_epi64x(0, 0, -1, -1);

            const sample_t remaining = num_samples % 4;
            sample_t sample_idx = 0;
            for (; sample_idx < num_samples - remaining; sample_idx += 4) {
                const __m128i candidate_vec32 = _mm_load_si128((__m128i*) &(sample_to_candidate[sample_idx]));
                const __m128i bin_idx_vec32 = _mm_load_si128((__m128i*) &(feat.bin_index[sample_idx]));

                const __m256i candidate_vec = _mm256_cvtepu32_epi64(candidate_vec32);
                const __m256i bin_idx_vec = _mm256_cvtepu32_epi64(bin_idx_vec32);

                //cand*histoff + bin*binoffset
                const __m256i candidate_off = _mm256_mul_epi32(candidate_vec, hist_offset);
                const __m256i bin_off = _mm256_mul_epi32(bin_idx_vec, bin_size_offset);
                const __m256i total_offset = _mm256_add_epi64(candidate_off, bin_off);
//        const __m256i final_hist_ptr = _mm256_add_epi32(hist_base_vec, total_offset);

                const __m256i not_one_MASK = _mm256_cmpgt_epi64(candidate_vec, mones);
                const __m256d not_one_MASK_dbl = _mm256_castps_pd(not_one_MASK);

                const __m256d gradients_vec = _mm256_loadu_pd(&(gradients[sample_idx]));

                // Assuming sum_counts lies before sum_gradients
                const __m256i sum_counts_ptr = total_offset; //_mm256_add_epi32(total_offset, fours);
                const __m256i sum_gradients_ptr = _mm256_add_epi64(total_offset, eights);

                const __m256d sum_counts_val = _mm256_mask_i64gather_pd(zeros, hist_base_addr, sum_counts_ptr, not_one_MASK_dbl, 1);
                const __m256d sum_gradients_val = _mm256_mask_i64gather_pd(zeros, hist_base_addr, sum_gradients_ptr, not_one_MASK_dbl, 1);

                const __m256d new_sum_counts_val = _mm256_add_pd(sum_counts_val, _mm256_and_pd(ones, not_one_MASK_dbl));
                const __m256d new_sum_gradients_val = _mm256_add_pd(sum_gradients_val, _mm256_and_pd(gradients_vec, not_one_MASK_dbl));

                const __m256d c1g1c3g3 = _mm256_shuffle_pd(new_sum_counts_val, new_sum_gradients_val, 0);
                const __m256d c2g2c4g4 = _mm256_shuffle_pd(new_sum_counts_val, new_sum_gradients_val, 0xF);
                const __m256d store_mask13 = _mm256_cmp_pd(c1g1c3g3, zeros, _CMP_NEQ_UQ);
                const __m256d store_mask24 = _mm256_cmp_pd(c2g2c4g4, zeros, _CMP_NEQ_UQ);
                int offset;

                const int i1 = 0;
                offset = _mm256_extract_epi64(total_offset, i1);
                double* a = hist_base_addr + offset/8;
                __m256d final_mask = _mm256_and_pd(lower_MASK_STORE, store_mask13);
                _mm256_maskstore_pd(a, final_mask, c1g1c3g3);

                const int i3 = 2;
                offset = _mm256_extract_epi64(total_offset, i3);
                a = hist_base_addr + offset/8 - 2;
                final_mask = _mm256_and_pd(upper_MASK_STORE, store_mask13);
                _mm256_maskstore_pd(a, final_mask, c1g1c3g3);

                const int i2 = 1;
                offset = _mm256_extract_epi64(total_offset, i2);
                a = hist_base_addr + offset/8;
                final_mask = _mm256_and_pd(lower_MASK_STORE, store_mask24);
                _mm256_maskstore_pd(a, final_mask, c2g2c4g4);

                const int i4 = 3;
                offset = _mm256_extract_epi64(total_offset, i4);
                a = hist_base_addr + offset/8 - 2;
                final_mask = _mm256_and_pd(upper_MASK_STORE, store_mask24);
                _mm256_maskstore_pd(a, final_mask, c2g2c4g4);

//            cout<<"offset: "<<offset<<",\tnew addr[COUNT]:"<<hist_base_addr + offset/8 <<",\tnew addr[GRAD]:"<<hist_base_addr + offset/8 + 1<<",\tval[count]:"<<*(hist_base_addr + offset/8)
//            <<",\tval[grad]:"<<*(hist_base_addr + offset/8 + 1) <<",\tcorrect_addr[count]:"<<&(histograms[candidate][bin].sum_count)
//            <<",\tcorrect_addr[grad]:"<<&(histograms[candidate][bin].sum_gradients)<<",\tcand:"<<candidate<<",\tbin:"<<bin<<",\t correct val[count]:"<<histograms[candidate][bin].sum_count
//            <<"correct val[grad]:"<<histograms[candidate][bin].sum_gradients<<endl;
//            cout<<endl;

//            cout<<"new addr[COUNT]:"<<hist_base_addr + offset/8 <<",\tval[count]:"<<*(hist_base_addr + offset/8)
//                <<",\tval[grad]:"<<*(hist_base_addr + offset/8 + 1) <<",\tcorrect_addr[count]:"<<&(histograms[candidate][bin].sum_count)
//                <<",\tcorrect_addr[grad]:"<<&(histograms[candidate][bin].sum_gradients)<<",\tcand:"<<candidate<<",\tbin:"<<bin<<",\t correct val[count]:"<<histograms[candidate][bin].sum_count
//                <<"correct val[grad]:"<<histograms[candidate][bin].sum_gradients<<endl;
//            cout<<endl;
            }

            for (; sample_idx < num_samples; sample_idx++)
            {
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
            if (dataset->get_data()[feature].bin_index[sample] <= bin) {
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
