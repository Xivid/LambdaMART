#include <lambdamart/model.h>

namespace LambdaMART {


void Model::train(const Dataset& dataset, const Config& config) {
    /* uint64_t */ int        num_samples = dataset.get_num_samples();
    std::vector<double>       currentScores(num_samples, 0.0);
    std::vector<double>       node_to_score;
    std::vector<unsigned int> sample_to_node(num_samples);
    std::vector<double>       gradients(num_samples);
    std::vector<double>       hessians(num_samples);
    LambdaRank                calculator(dataset.get_queries());

    for (int iter = 0; iter < config.num_iterations; ++iter) {
        calculator.get_derivatives(currentScores, gradients, hessians);
        trees.push_back(build_new_tree(dataset, gradients, hessians, node_to_score, sample_to_node, config));
        for (size_t sid = 0; sid < num_samples; ++sid) {
            currentScores[sid] += config.learning_rate * node_to_score[sample_to_node[sid]];
        }
    }
}

std::vector<double>* Model::predict(Dataset *data) {
    return nullptr;
}

}