#ifndef LAMBDAMART_BOOSTER_H
#define LAMBDAMART_BOOSTER_H

#include <lambdamart/model.h>

namespace LambdaMART {
    class Booster {
        bool check_early_stopping();

    public:
        Model *train(const Dataset &dataset, const Config &config);
    };
}
#endif //LAMBDAMART_BOOSTER_H
