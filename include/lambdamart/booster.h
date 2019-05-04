#ifndef LAMBDAMART_BOOSTER_H
#define LAMBDAMART_BOOSTER_H

#include <lambdamart/model.h>

namespace LambdaMART {
    class Booster {
        Dataset*                  train_dataset;
        RawDataset*               valid_dataset;
        Config*                   config;
        uint64_t                  num_samples;
        std::vector<double>       current_scores;
        std::vector<double>       gradients;
        std::vector<double>       hessians;
        LambdaRank*               train_ranker;
        LambdaRank*               valid_ranker;
        Model*                    model;

        inline string get_train_ndcg_string() {
            vector<double> result = train_ranker->eval(current_scores.data());
            string tmp;
            for (size_t i = 0 ; i < result.size(); ++i) {
                tmp += "\ttrain-ndcg@" + to_string(config->eval_at[i]) + ":" + to_string(result[i]);
            }
            return tmp;
        }

        inline string get_valid_ndcg_string() {
            vector<double> predictions = model->predict(valid_dataset);
            vector<double> result = valid_ranker->eval(predictions.data());
            string tmp;
            for (size_t i = 0 ; i < result.size(); ++i) {
                tmp += "\tvalid-ndcg@" + to_string(config->eval_at[i]) + ":" + to_string(result[i]);
            }
            return tmp;
        }

    public:
        Booster() = delete;

        Booster(Dataset* _train_dataset, RawDataset* _valid_dataset, Config* _config)
          : train_dataset(_train_dataset), valid_dataset(_valid_dataset), config(_config)
        {
            model = nullptr;
            num_samples = train_dataset->num_samples();
            current_scores.resize(num_samples, 0.0);
            gradients.resize(num_samples);
            hessians.resize(num_samples);
            train_ranker = new LambdaRank(*train_dataset, *config);
            valid_ranker = valid_dataset ? new LambdaRank(*valid_dataset, *config) : nullptr;
        }

        Model* train();
    };
}
#endif //LAMBDAMART_BOOSTER_H
