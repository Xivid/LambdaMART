#ifndef LAMBDAMART_BOOSTER_H
#define LAMBDAMART_BOOSTER_H

#include <lambdamart/model.h>

namespace LambdaMART {
    class Booster {
        Dataset*                  dataset;
        Config*                   config;
        uint64_t                  num_samples;
        std::vector<double>       current_scores;
        std::vector<double>       gradients;
        std::vector<double>       hessians;
        LambdaRank*               ranker;

        bool check_early_stopping();


    public:
        Booster() = delete;

        Booster(Dataset* _dataset, Config* _config) : dataset(_dataset), config(_config)
        {
            num_samples = dataset->num_samples();
            current_scores.resize(num_samples, 0.0);
            gradients.resize(num_samples);
            hessians.resize(num_samples);
            ranker = new LambdaRank(*dataset, *config);
        }

        Model* train();
    };
}
#endif //LAMBDAMART_BOOSTER_H
