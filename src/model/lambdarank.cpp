#include <lambdamart/lambdarank.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

namespace LambdaMART {


void LambdaRank::get_derivatives(const std::vector<double>& currentScores, std::vector<double>& gradients, std::vector<double>& hessians) const {

}

void NDCGCalculator::DefaultEvalRanks(std::vector<int>* eval_ranks) {
    if (eval_ranks->empty()) {
        for (int i = 1; i <= 5; ++i) {
            eval_ranks->push_back(i);
        }
    } else {
        for (int i = 0; i < eval_ranks->size(); ++i) {
            // TODO remove later
            if (!(eval_ranks->at(i) > 0)) {
                std::cout << "Check failed at lambdarank: DefaultEvalRanks"  << std::endl;
                break;
            }
        }
    }
}
void NDCGCalculator::SetLabelGain(std::vector<double>* label_gain) {
    // TODO: where to set max label
    // relevant gain for labels
    // label_gain = 2^i - 1, may overflow, so we use 31 here
    const int max_label = 31;
    label_gain->push_back(0.0f);
    for (int i = 1; i < max_label; ++i) {
        label_gain->push_back(static_cast<double>((1 << i) - 1));
    }
    label_gain_.resize(label_gain->size());
}

void NDCGCalculator::Init(const std::vector<double>& input_label_gain) {
    label_gain_.resize(input_label_gain.size());
    for (int i = 0; i < input_label_gain.size(); ++i) {
        label_gain_[i] = static_cast<double>(input_label_gain[i]);
    }
    discount_.resize(kMaxPosition);
    for (int i = 0; i < kMaxPosition; ++i) {
        discount_[i] = 1.0 / std::log2(2.0 + i);
    }
}

double NDCGCalculator::CalDCGAtK(int k, const int* label, const double* score, int num_data) {
    std::vector<int> sorted_idx(num_data);
    for (int i = 0; i < num_data; ++i) sorted_idx[i] = i;
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(), [score](int a, int b) 
            {return score[a] > score[b]; });

    if (k > num_data) k = num_data;
    double dcg = 0.0f;
    for (int i = 0; i < k; ++i) {
        int idx = sorted_idx[i];
        dcg += label_gain_[static_cast<int>(label[idx])] * discount_[i];
    }
    return dcg;
}

void NDCGCalculator::CalDCG(const std::vector<int>& ks, const int* label, 
        const double* score, int num_data, std::vector<double>* out) {
    std::vector<int> sorted_idx(num_data);
    for (int i = 0; i < num_data; ++i) sorted_idx[i] = i;
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),[score](int a, int b) 
        {return score[a] > score[b]; });

    double cur_result = 0.0f;
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

double NDCGCalculator::CalMaxDCGAtK(int k, const int* label, int num_data) {
    double ret = 0.0f;
    std::vector<int> label_counts(label_gain_.size(), 0);
    for(int i = 0; i < num_data; ++i) {
        label_counts[static_cast<int>(label[i])]++;
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

void NDCGCalculator::CalMaxDCG(const std::vector<int>& ks, const int* label, 
                            int num_data, std::vector<double>* out) {
    std::vector<int> label_counts(label_gain_.size(), 0);
    // get counts for all labels
    for (int i = 0; i < num_data; ++i) {
        ++label_counts[static_cast<int>(label[i])];
    }
    double cur_result = 0.0f;
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

}