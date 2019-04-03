#ifndef LAMBDAMART_DATASET_H
#define LAMBDAMART_DATASET_H

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <cstring>
#include "bin.h"

using namespace std;

namespace LambdaMART {

    class Dataset {
        Binner* binner;
        vector<vector<uint8_t>> samples;
        int d = 1024;

    public:
        Dataset(Binner* binner, vector<vector<uint8_t>>& samples){
            this->binner = binner;
            this->samples = samples;
        }

        void load_dataset(const char* path, Binner* _binner = nullptr) {
            if (_binner) {
                this->samples =  _binner->load(path, this->d);
            } else {
                return (new PercentileBinner())->load(path, d);
            }
        }

        Binner* get_binner() {
            return binner;
        }
    };

}
#endif //LAMBDAMART_DATASET_H
