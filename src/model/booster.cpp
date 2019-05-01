#include <lambdamart/booster.h>

namespace LambdaMART {

Model* Booster::train(const Dataset& dataset, const Config& config) {
    auto model = new Model();
    auto treeLearner = new TreeLearner();

    uint64_t                  num_samples = dataset.num_samples();
    std::vector<double>       currentscores(num_samples, 0.0);
    std::vector<double>       node_to_score;
    std::vector<unsigned int> sample_to_node(num_samples);
    std::vector<double>       gradients(num_samples);
    std::vector<double>       hessians(num_samples);
    std::vector<Histogram>    histograms;
    LambdaRank                ranker(dataset.query_boundaries(), dataset.num_queries(), dataset.label(), config);

    bool is_finished = false;
    int num_iter = config.num_iterations;
    for (int iter = 0; iter < num_iter && !is_finished; ++iter) {
        ranker.get_derivatives(currentscores.data(), gradients.data(), hessians.data());
        model->add_tree(treeLearner->build_new_tree(dataset, gradients, hessians, histograms, node_to_score, sample_to_node, config),
                        0.0);
        for (size_t sid = 0; sid < num_samples; ++sid) {
            currentscores[sid] += config.learning_rate * node_to_score[sample_to_node[sid]];
        }
        is_finished = check_early_stopping();
    }

    return model;
}


// TODO
// Checks for early stopping:
// - if there are no more leaves meet the split requirement (should this check be done in build_new_tree?)
// - others?
bool Booster::check_early_stopping() {
    bool early_stopping = false;
    // TODO
    return early_stopping;
}


}