#ifndef LAMBDAMART_LAMBDARANK_H
#define LAMBDAMART_LAMBDARANK_H
#include <vector>

namespace LambdaMART {

class LambdaRank {
    friend class Model;

    explicit LambdaRank(const std::vector<uint64_t>& query_boundaries) {
        boundaries = query_boundaries.data();
    }

private:
    const uint64_t* boundaries;
    void get_derivatives(const std::vector<double>& currentScores, std::vector<double>& gradients, std::vector<double>& hessians) const;
};

}

#endif //LAMBDAMART_LAMBDARANK_H
