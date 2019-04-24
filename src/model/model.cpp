#include <lambdamart/model.h>

namespace LambdaMART {


void Model::train(const Dataset& dataset, const Config& config) {
    uint64_t                  num_samples = dataset.get_num_samples();
    std::vector<double>       currentscores(num_samples, 0.0);
    std::vector<double>       node_to_score;
    std::vector<unsigned int> sample_to_node(num_samples);
    std::vector<double>       gradients(num_samples);
    std::vector<double>       hessians(num_samples);
    LambdaRank                ranker(dataset.query_boundaries(), dataset.num_queries(), dataset.label(), config);

    bool is_finished = false;
    int num_iter = config.num_iterations;
    for (int iter = 0; iter < num_iter && !is_finished; ++iter) {
        ranker.get_derivatives(currentscores.data(), gradients.data(), hessians.data());
        trees.push_back(build_new_tree(dataset, gradients, hessians, node_to_score, sample_to_node, config));
        for (size_t sid = 0; sid < num_samples; ++sid) {
            currentscores[sid] += config.learning_rate * node_to_score[sample_to_node[sid]];
        }
        is_finished = check_early_stopping();
    }
}

bool Model::train_one_itr(const double* gradients, const double* hessians) {
    return true;
}

double* Model::predict(Dataset* data) {
    int num_iter = trees.size();
    if (num_iter == 0) return nullptr;
    datasize_t num_data = data->get_num_samples();
    std::vector<double> out(num_data);
    for (datasize_t i = 0; i < num_data; ++i) {
        double score = 0.0f;
        for (int m = 0; m < num_iter; ++m) {
            score += predict_score(data, i, trees[m]);
        }
        out[i] = score;
    }
    return out.data();
}

// TODO
// Checks for early stopping:
// - if there are no more leaves meet the split requirement (should this check be done in build_new_tree?)
// - others?
bool Model::check_early_stopping() {
    bool early_stopping = false;
    // TODO
    return early_stopping;
}

}