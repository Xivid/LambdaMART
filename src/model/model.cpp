#include <lambdamart/model.h>
#include <lambdamart/dataset.h>

namespace LambdaMART {


double* Model::predict(RawDataset* data) {
    size_t num_iter = trees.size();
    if (num_iter == 0) {
        Log::Fatal("Empty model!");
        return nullptr;
    }
    sample_t num_data = data->num_samples();
    double *out = new double[num_data];
    for (sample_t i = 0; i < num_data; ++i) {
        double score = 0.0f;
        for (size_t m = 0; m < num_iter; ++m) {
            score += trees[m]->predict_score(data->get_sample_row(i));
        }
        out[i] = score;
    }
    return out;
}

}