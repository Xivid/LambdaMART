#include <lambdamart/model.h>
#include <lambdamart/dataset.h>

namespace LambdaMART {


vector<double> Model::predict(RawDataset* data) {
    size_t num_iter = trees.size();
    sample_t num_data = data->num_samples();
    vector<double> predictions(num_data);
    for (sample_t i = 0; i < num_data; ++i) {
        score_t score = 0.0f;
        for (size_t m = 0; m < num_iter; ++m) {
            score += trees[m]->predict_score(data->get_sample_row(i)) * tree_weights[m];
        }
        predictions[i] = score;
    }
    return predictions;
}

vector<double> Model::predict(RawDataset* data, const string& output_path) {
    size_t num_iter = trees.size();
    sample_t num_data = data->num_samples();
    vector<double> predictions(num_data);
    for (sample_t i = 0; i < num_data; ++i) {
        score_t score = 0.0f;
        for (size_t m = 0; m < num_iter; ++m) {
            score += trees[m]->predict_score(data->get_sample_row(i)) * tree_weights[m];
        }
        predictions[i] = score;
    }

    Log::Info("Writing predictions to %s", output_path.c_str());
    ofstream fout(output_path);
    for (score_t score: predictions) {
        fout << score << '\n';
    }
    fout.close();

    return predictions;
}


}