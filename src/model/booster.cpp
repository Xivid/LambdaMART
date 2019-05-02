#include <lambdamart/booster.h>

namespace LambdaMART {

Model* Booster::train() {
    auto model = new Model();
    auto treeLearner = new TreeLearner(dataset, gradients.data(), hessians.data(), config);

    const int num_iter = config->num_iterations;
    const double learning_rate = config->learning_rate;
    Log::Debug("Train %d iterations with learning rate %lf", num_iter, learning_rate);

    for (int iter = 0; iter < num_iter; ++iter) {
        Log::Debug("Iteration %d: start", iter);
        ranker->get_derivatives(current_scores.data(), gradients.data(), hessians.data());

        double maxElement = *std::max_element(gradients.begin(), gradients.end());
        size_t maxElementIndex = std::max_element(gradients.begin(),gradients.end()) - gradients.begin();
        Log::Debug("max: %f, index: %u", maxElement, maxElementIndex);

        Tree* tree = treeLearner->build_new_tree();
        model->add_tree(tree, learning_rate);

        for (sample_t sid = 0; sid < num_samples; ++sid) {
            current_scores[sid] += learning_rate * treeLearner->get_sample_score(sid);
        }

        for (int eval_at: config->eval_at) {
            Log::Info("Iteration %d: NDCG@%d = %lf", iter, eval_at, 0.0);
        }
    }

    return model;
}

}
