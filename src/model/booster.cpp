#include <lambdamart/booster.h>

namespace LambdaMART {

Model* Booster::train() {
    model = new Model();
    auto treeLearner = new TreeLearner(train_dataset, gradients.data(), hessians.data(), config);

    const int num_iter = config->num_iterations;
    const double learning_rate = config->learning_rate;
    LOG_DEBUG("Train %d iterations with learning rate %lf", num_iter, learning_rate);

    for (int iter = 1; iter <= num_iter; ++iter) {
        LOG_DEBUG("Iteration %d: start", iter);
        train_ranker->get_derivatives(current_scores.data(), gradients.data(), hessians.data());

        Tree* tree = treeLearner->build_new_tree();
        model->add_tree(tree, learning_rate);

        for (sample_t sid = 0; sid < num_samples; ++sid) {
            current_scores[sid] += learning_rate * treeLearner->get_sample_score(sid);
        }


        if( iter % 10 == 0 )
            Log::Info("[%d]%s%s", iter, get_train_ndcg_string().c_str(), valid_dataset ? get_valid_ndcg_string().c_str() : "");
    }

    return model;
}

}
