#ifndef LAMBDAMART_LAMBDARANK_H
#define LAMBDAMART_LAMBDARANK_H
#include <vector>

namespace LambdaMART {

class NDCGCalculator {
public:

    NDCGCalculator() {
        // TODO
    }
    static void DefaultEvalRanks(std::vector<int>* eval_ranks);
    static void SetLabelGain(std::vector<double>* label_gain); // uses the default of 2^i-1

    static void Init(const std::vector<double>& input_label_gain);

    // Calculates the DCG score at position k, given the rank label and score
    static double CalDCGAtK(int k, const int* label,
        const double* score, int num_data);

    // Calculates the DCG score at multiple locations
    // the result is stored in out. label and score are pointers to
    // labels and scores respectively
    static void CalDCG(const std::vector<int>& ks,
        const int* label, const double* score,
        int num_data, std::vector<double>* out);

    // calculates the max score (ideal DCG) at position k
    // returns: max score
    static double CalMaxDCGAtK(int k,
        const int* label, int num_data);

    // calculates the max DCG (ideal DCG), result is stored in out
    static void CalMaxDCG(const std::vector<int>& ks,
        const int* label, int num_data, std::vector<double>* out);

    // checks the label range
    static void CheckLabel(const int* label, int num_data);

    // gets discount score at position k
    inline static double GetDiscount(int k) { return discount_[k]; }

private:
    static std::vector<double> label_gain_;
    static std::vector<double> discount_;
    // max position of rank
    static const int kMaxPosition;
};

class LambdaRank {
    friend class Model;

    explicit LambdaRank(const std::vector<uint64_t>& query_boundaries) {
        boundaries = query_boundaries.data();
    }

private:
    NDCGCalculator calculator;
    const uint64_t* boundaries;
    void get_derivatives(const std::vector<double>& currentScores, std::vector<double>& gradients, std::vector<double>& hessians) const;
};



}

#endif //LAMBDAMART_LAMBDARANK_H
