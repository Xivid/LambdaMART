//
// Created by Zhifei Yang on 25.03.19.
//

#ifndef LAMBDAMART_BIN_H
#define LAMBDAMART_BIN_H

#include <vector>

namespace lambdamart {

    class Dataset;

    class Binner {
        friend class Dataset;

        std::vector<double> thresholds;

        virtual Dataset* load(const char* path) = 0;
    };

    class PercentileBinner : Binner {
        friend class Dataset;

        Dataset* load(const char* path) override;
    };

}
#endif //LAMBDAMART_BIN_H
