//
// Created by Zhifei Yang on 25.03.19.
//

#ifndef LAMBDAMART_DATASET_H
#define LAMBDAMART_DATASET_H

#include <vector>
#include "bin.h"

namespace lambdamart {

    class Dataset {
        Binner* binner;
        std::vector<std::vector<uint8_t>> samples;
    public:
        static Dataset* load_dataset(const char* path, Binner* _binner = nullptr) {
            if (_binner) {
                return _binner->load(path);
            } else {
                return (new PercentileBinner())->load(path);
            }
        }

    public:
        Binner* get_binner() {
            return binner;
        }
    };

}



#endif //LAMBDAMART_DATASET_H
