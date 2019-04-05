#ifndef LAMBDAMART_LAMBDARANK_H
#define LAMBDAMART_LAMBDARANK_H
#include <vector>

namespace LambdaMART {

class NDCGCalculator {
public:

    NDCGCalculator() {
        // TODO
    }
    void DefaultEvalRanks(std::vector<int>* eval_ranks);
    void SetLabelGain(std::vector<double>* label_gain); // uses the default of 2^i-1

    void Init(const std::vector<double>& input_label_gain);

    // Calculates the DCG score at position k, given the rank label and score
    double CalDCGAtK(int k, const int* label,
        const double* score, int num_data);

    // Calculates the DCG score at multiple locations
    // the result is stored in out. label and score are pointers to
    // labels and scores respectively
    void CalDCG(const std::vector<int>& ks,
        const int* label, const double* score,
        int num_data, std::vector<double>* out);

    // calculates the max score (ideal DCG) at position k
    // returns: max score
    double CalMaxDCGAtK(int k,
        const int* label, int num_data);

    // calculates the max DCG (ideal DCG), result is stored in out
    void CalMaxDCG(const std::vector<int>& ks,
        const int* label, int num_data, std::vector<double>* out);

    // checks the label range
    void CheckLabel(const int* label, int num_data);

    // gets discount score at position k
    inline double GetDiscount(int k) { return discount_[k]; }

private:
    std::vector<double> label_gain_;
    std::vector<double> discount_;
    // max position of rank
    const int kMaxPosition = 10;
};

class LambdaRank {
    friend class Model;

    explicit LambdaRank(const std::vector</*uint64_t*/int>& query_boundaries) {
        boundaries = query_boundaries.data();
    }

private:
    NDCGCalculator calculator;
    const /*uint64_t*/ int* boundaries;
    void get_derivatives(const std::vector<double>& currentScores, std::vector<double>& gradients, std::vector<double>& hessians) const;
};



}

#endif //LAMBDAMART_LAMBDARANK_H
