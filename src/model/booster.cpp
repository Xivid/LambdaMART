#include <lambdamart/booster.h>

namespace LambdaMART {

Model* Booster::train() {
    auto model = new Model();
    Log::Debug("New model created");
    auto treeLearner = new TreeLearner(dataset, gradients.data(), hessians.data(), config);
    Log::Debug("New tree learner created");

    bool is_finished = false;
    const int num_iter = config->num_iterations;
    const double learning_rate = config->learning_rate;
    Log::Debug("Train %d iterations with learning rate %lf", num_iter, learning_rate);

    for (int iter = 0; iter < num_iter && !is_finished; ++iter) {
        ranker->get_derivatives(current_scores.data(), gradients.data(), hessians.data());

        Tree* tree = treeLearner->build_new_tree();
        model->add_tree(tree, 1.0);  // TODO: tree weight?

        for (size_t sid = 0; sid < num_samples; ++sid) {
            current_scores[sid] += learning_rate * treeLearner->get_sample_score(sid);
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