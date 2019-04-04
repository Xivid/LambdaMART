#include <lambdamart/model.h>

namespace LambdaMART {

Tree* Model::train_one_iteration(
        const LambdaRank& calculator,
        const std::vector<double>& currentScores,
        const LambdaMART::Dataset& dataset,
        std::vector<double>& gradients,
        std::vector<double>& hessians,
        const LambdaMART::Config& param)
{
    calculator.get_derivatives(currentScores, gradients, hessians);
    return build_new_tree(dataset, gradients, hessians, param);
}

void Model::train(const LambdaMART::Dataset& dataset, const LambdaMART::Config& param) {
    uint64_t                  num_samples = dataset.get_num_samples();
    std::vector<double>       currentScores(num_samples);
    std::vector<double>       node_to_score;
    std::vector<unsigned int> sample_to_node(num_samples);
    std::vector<double>       gradients(num_samples);
    std::vector<double>       hessians(num_samples);
    LambdaRank                calculator(dataset.get_boundaries());

    for (int iter = 0; iter < param.num_iterations; ++iter) {
        trees.push_back(train_one_iteration(calculator, currentScores, dataset, gradients, hessians, param));
        for (size_t sid = 0; sid < num_samples; ++sid) {
            currentScores[sid] += param.learning_rate * node_to_score[sample_to_node[sid]];
        }
    }
}

std::vector<double>* Model::predict(LambdaMART::Dataset *data) {
    return nullptr;
}

}