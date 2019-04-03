#ifndef LAMBDAMART_DATASET_H
#define LAMBDAMART_DATASET_H

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <cstring>

using namespace std;

namespace LambdaMART {

class Dataset {
    vector<vector<uint8_t>> samples;
    vector<uint64_t> boundaries;

public:
    uint64_t                get_num_samples() const { return samples.size(); }
    const vector<uint64_t>& get_boundaries()  const { return boundaries; };
};

Dataset* load_dataset(const char* data_path, const char* query_path, Dataset* use_binning_from = nullptr);

}
#endif //LAMBDAMART_DATASET_H
