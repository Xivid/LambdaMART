#include <lambdamart/lambdarank.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

namespace LambdaMART {


void LambdaRank::get_derivatives(float* currentScores, float* gradients, float* hessians) {
    for (sample_t i = 0; i < num_queries_; ++i) {
        get_derivatives_one_query(currentScores, gradients, hessians, i);
    }

}

inline void LambdaRank::get_derivatives_one_query(float* scores, float* gradients,
                                        float* hessians, sample_t query_id) {

    const float kminscore = -std::numeric_limits<float>::infinity();

    const sample_t start = boundaries_[query_id];
    const sample_t count = boundaries_[query_id+1] - start;
    const float inverse_max_dcg = inverse_max_dcg_[query_id];
    //const label_t* label = label_ + start;
    scores += start;
    gradients += start;
    hessians += start;

    for (sample_t i = 0; i < count; ++i) {
        gradients[i] = 0.0f;
        hessians[i] = 0.0f;
    }

    // get sorted indices for scores;
    std::vector<sample_t> sindex;
    for (sample_t i = 0; i < count; ++i) {
        sindex.emplace_back(i);
    }
    std::stable_sort(sindex.begin(), sindex.end(), [scores](sample_t a, sample_t b) {return scores[a] > scores[b]; });
    float best_score = scores[sindex[0]];
    sample_t worst_idx = count - 1;
    if (worst_idx > 0 && scores[sindex[worst_idx]] == kminscore) worst_idx -= 1;
    const float worst_score = scores[sindex[worst_idx]];

    for (sample_t i = 0; i < count; ++i) {
        const sample_t high = sindex[i];
        const int high_label = static_cast<int>(label_[start + high]);
        const float high_score = scores[high];
        if (high_score == kminscore) {continue; }
        const float hl_gain = label_gain_[high_label];
        const float h_discount = get_discount(i);
        float high_sum_gradient = 0.0;
        float  high_sum_hessian = 0.0;
        for (sample_t j = 0; j < count; ++j) {
            if (i == j) continue;

            const sample_t low = sindex[j];
            const int low_label = static_cast<int>(label_[start + low]);
            const float low_score = scores[low];
            // only consider pairs with different labels
            if (high_label <= low_label || low_score == kminscore) continue;

            const float delta = high_score - low_score;
            const float ll_gain = label_gain_[low_label];
            const float l_discount = get_discount(j);
            const float dcg_gap = hl_gain - ll_gain;
            const float pair_discount = fabs(h_discount - l_discount);
            float delta_pair_ndcg = dcg_gap * pair_discount * inverse_max_dcg;
            //std::cout << "delta pair " << dcg_gap << " " << h_discount << " " << l_discount << std::endl;
            // regularize the pair ndcg by score distance
            if (high_label != low_label && best_score != worst_score) {
                delta_pair_ndcg /= (0.01f + fabs(delta));
            }
            // calculate gradient and hessian for this pair
            float p_gradient = get_sigmoid(delta);
            float p_hessian = p_gradient * (2.0f - p_gradient);

//            p_gradient *= -delta_pair_ndcg;
            p_gradient *= delta_pair_ndcg;
            p_hessian *= 2 * delta_pair_ndcg;
            high_sum_gradient += p_gradient;
            high_sum_hessian += p_hessian;
            gradients[low] -= p_gradient;
            hessians[low] += p_hessian;
        }
        gradients[high] += high_sum_gradient;
        hessians[high] += high_sum_hessian;

    }
    float accum = 0.;
    for (int i = 0; i < count; ++i) {
        accum += gradients[i] * gradients[i];
    }
    float norm = sqrt(accum);
    float haccum = 0.;
    for (int i = 0; i < count; ++i) {
        haccum += hessians[i] * hessians[i];
    }
    float hnorm = sqrt(haccum);
//    std::cout << "Query " << query_id << ": gradient norm: " << norm << " hessian norm: " << hnorm << std::endl;
//    //std::cout << "score " << scores[0] << " " << scores[1] << " " << scores[2] << " " << scores[3] << " " << scores[4] << std::endl;
//
//    std::cout << "score: ";
//    for (int i = 0; i < count; ++i) {
//        std::cout << scores[i] << " ";
//    }
//    std::cout << std::endl;

}

std::vector<float> LambdaRank::eval(float* scores) {
    std::vector<float> result(eval_at_.size());
    std::vector<float> tmp_dcg(eval_at_.size(), 0.0f);
    for (sample_t i = 0; i < num_queries_; ++i) {
        if (eval_inverse_max_dcg_[i][0] <= 0.0f) {
            for (int j = 0; j < result.size(); ++j) {
                result[j] += 1.0f;
            }
        } else {
            sample_t num_data = boundaries_[i+1] - boundaries_[i];
            cal_dcg(eval_at_, label_ + boundaries_[i], scores + boundaries_[i], num_data, &tmp_dcg);
            for (int j = 0; j < result.size(); ++j) {
                result[j] += tmp_dcg[j] * eval_inverse_max_dcg_[i][j];
            }
        }
    }

    for (int i = 0; i < eval_at_.size(); ++i) {
        result[i] /= num_queries_;
    }

    return result;
}

void LambdaRank::set_eval_rank(std::vector<sample_t>* eval_ranks) {
    if (eval_ranks->empty()) {
        for (int i = 1; i <= 5; ++i) {
            eval_ranks->push_back(i);
        }
    } else {
        for (int i = 0; i < eval_ranks->size(); ++i) {
            // TODO remove later
            if (!(eval_ranks->at(i) > 0)) {
                Log::Fatal("Check failed at lambdarank: DefaultEvalRanks");
                break;
            }
        }
    }
}
void LambdaRank::set_label_gain(int max_label) {
    // relevant gain for labels
    // label_gain = 2^i - 1, may overflow, so we use 31 here
    label_gain_.push_back(0.0f);
    for (int i = 1; i <= max_label; ++i) {
        label_gain_.push_back(static_cast<float>((1 << i) - 1));
    }
    label_gain_.resize(max_label+1);
}

void LambdaRank::set_discount() {
    discount_.resize(kMaxPosition);
    for (int i = 0; i < kMaxPosition; ++i) {
        discount_[i] = 1.0 / std::log2(2.0 + i);
    }
}

float LambdaRank::cal_dcg_k(int k, const label_t* label, const float* score, sample_t num_data) {
    std::vector<int> sorted_idx(num_data);
    for (int i = 0; i < num_data; ++i) sorted_idx[i] = i;
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(), [score](int a, int b)
            {return score[a] > score[b]; });

    if (k > num_data) k = num_data;
    float dcg = 0.0f;
    for (int i = 0; i < k; ++i) {
        int idx = sorted_idx[i];
        dcg += label_gain_[static_cast<int>(label[idx])] * discount_[i];
    }
    return dcg;
}

void LambdaRank::cal_dcg(const std::vector<int>& ks, const label_t* label,
        const float* score, sample_t num_data, std::vector<float>* out) {
    std::vector<uint16_t> sorted_idx(num_data);
    for (int i = 0; i < num_data; ++i) sorted_idx[i] = i;
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),[score](int a, int b)
        {return score[a] > score[b]; });

    float cur_result = 0.0f;
    int cur_left = 0;
    for (int i = 0; i < ks.size(); ++i) {
        int cur_k = ks[i];
        if (cur_k > num_data) cur_k = num_data;
        for (int j = cur_left; j < cur_k; ++j) {
            int idx = sorted_idx[j];
            cur_result += label_gain_[static_cast<int>(label[idx])] * discount_[j];
        }
        (*out)[i] = cur_result;
        cur_left = cur_k;
    }
}

float LambdaRank::cal_maxdcg_k(int k, sample_t start, sample_t num_data) {
    float ret = 0.0f;
    std::vector<int> label_counts(label_gain_.size(), 0);
    for(int i = 0; i < num_data; ++i) {
        label_counts[static_cast<int>(label_[start + i])]++;
    }
    // top_label has the highest label_gain_
    int top_label = static_cast<int>(label_gain_.size()) - 1;
    if (k > num_data) k = num_data;
    for (int j = 0; j < k; ++j) {
        while (top_label > 0 && label_counts[top_label] <= 0) {
            top_label -= 1;
        }
        if (top_label < 0) {
            break;
        }
        ret += discount_[j] * label_gain_[top_label];
        label_counts[top_label] -= 1;
    }
    return ret;
}

void LambdaRank::cal_maxdcg(const std::vector<int>& ks, const label_t* label,
                            sample_t num_data, std::vector<float>* out) {
    std::vector<int> label_counts(label_gain_.size(), 0);
    // get counts for all labels
    for (int i = 0; i < num_data; ++i) {
        ++label_counts[static_cast<int>(label[i])];
    }
    float cur_result = 0.0f;
    int cur_left = 0;
    int top_label = static_cast<int>(label_gain_.size()) - 1;
    for (int i = 0; i < ks.size(); ++i) {
        int cur_k = ks[i];
        if (cur_k > num_data) cur_k = num_data;
        for (int j = cur_left; j < cur_k; ++j) {
            while (top_label > 0 && label_counts[top_label] <= 0) {
                top_label -= 1;
            }
            if (top_label < 0) {
                break;
            }
            cur_result += discount_[j] * label_gain_[top_label];
            label_counts[top_label] -= 1;
        }
        (*out)[i] = cur_result;
        cur_left = cur_k;
    }
}

float LambdaRank::get_sigmoid(float score) const {
    if (score <= min_input_) return sigmoid_table_[0];
    else if (score >= max_input_) return sigmoid_table_[sigmoid_bins_-1];
    else return sigmoid_table_[static_cast<uint32_t>((score - min_input_) * sig_factor_)];
}

void LambdaRank::create_sigmoid_table() {
    min_input_ = min_input_ / sigmoid_ / 2;
    max_input_ = -min_input_;
    sigmoid_table_.resize(sigmoid_bins_);
    // score to bin factor
    sig_factor_ = sigmoid_bins_ / (max_input_ - min_input_);
    for (uint32_t i = 0; i < sigmoid_bins_; ++i) {
        const float score = i / sig_factor_ + min_input_;
        sigmoid_table_[i] = 2.0f / (1.0f + std::exp(2.0f * score * sigmoid_));
    }
}

}