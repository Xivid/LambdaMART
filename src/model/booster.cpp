#include <lambdamart/booster.h>

namespace LambdaMART {

Model* Booster::train() {
    model = new Model();
    auto treeLearner = new TreeLearner(train_dataset, gradients.data(), hessians.data(), config);

    const int num_iter = config->num_iterations;
    const float learning_rate = config->learning_rate;
    LOG_DEBUG("Train %d iterations with learning rate %lf", num_iter, learning_rate);
    Log::Info("num_feature_blocking: %u, num_sample_blocking: %u", config->num_feature_blocking, config->num_sample_blocking);

    for (int iter = 1; iter <= num_iter; ++iter) {
        LOG_DEBUG("Iteration %d: start", iter);
        train_ranker->get_derivatives(current_scores.data(), gradients.data(), hessians.data());

        Tree* tree = treeLearner->build_new_tree();
        model->add_tree(tree, learning_rate);

        for (sample_t sid = 0; sid < num_samples; ++sid) {
            current_scores[sid] += learning_rate * treeLearner->get_sample_score(sid);
        }

        if (iter % config->eval_interval == 0)
            Log::Info("[%d]%s%s", iter, get_train_ndcg_string().c_str(), valid_dataset ? get_valid_ndcg_string().c_str() : "");
    }
    Log::Info("update, cumulate, gbs, total cycles: %lld, %lld, %lld, %lld", treeLearner->sum_cycles_update,
              treeLearner->sum_cycles_cumulate, treeLearner->sum_cycles_gbs, treeLearner->sum_cycles_update+treeLearner->sum_cycles_cumulate+treeLearner->sum_cycles_gbs);

    return model;
}

}
