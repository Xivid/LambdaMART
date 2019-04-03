#ifndef LAMBDAMART_MODEL_H
#define LAMBDAMART_MODEL_H

#include "dataset.h"
#include "boosting.h"

namespace LambdaMART {
    class Model {

    public:
        void train(Dataset* data, Params* param) {
            for (int iter = 0; iter < param->num_iters; ++iter) {
                boosting::train_one_iter();
            }
        }

        std::vector<double>* predict(Dataset* data) {

        }

    };
}
#endif //LAMBDAMART_MODEL_H
