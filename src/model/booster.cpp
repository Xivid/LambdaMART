#include <lambdamart/booster.h>

namespace LambdaMART {

Model* Booster::train() {
    auto model = new Model();
    auto treeLearner = new TreeLearner(dataset, gradients.data(), hessians.data(), config);

    const int num_iter = config->num_iterations;
    const double learning_rate = config->learning_rate;
    LOG_DEBUG("Train %d iterations with learning rate %lf", num_iter, learning_rate);

    vector<double> result = ranker->eval(current_scores.data());
    string tmp = "Initial";
    for (size_t i = 0 ; i < result.size(); ++i) {
        tmp += "\tNDCG@" + to_string(config->eval_at[i]) + "=" + to_string(result[i]);
    }
    Log::Info(tmp.c_str());

    for (int iter = 0; iter < num_iter; ++iter) {
        LOG_DEBUG("Iteration %d: start", iter);
        ranker->get_derivatives(current_scores.data(), gradients.data(), hessians.data());

        /* double maxElement = *std::max_element(gradients.begin(), gradients.end());
        size_t maxElementIndex = std::max_element(gradients.begin(),gradients.end()) - gradients.begin();
        LOG_DEBUG("max: %f, index: %u", maxElement, maxElementIndex); */

        Tree* tree = treeLearner->build_new_tree();
        model->add_tree(tree, learning_rate);

        for (sample_t sid = 0; sid < num_samples; ++sid) {
            current_scores[sid] += learning_rate * treeLearner->get_sample_score(sid);
        }
//        if (iter == 0) {
//            current_scores[0] = -0.05;
//            current_scores[1] = 0.038083;
//            current_scores[2] = 0.038083;
//            current_scores[3] = -0.0323499;
//            current_scores[4] = -0.05;
//            current_scores[5] = 0.05;
//            current_scores[6] = 0.05;
//            current_scores[7] = 0.0130296;
//            current_scores[8] = 0.0112381;
//            current_scores[9] = -0.0299174;
//        }
//        if (iter == 1) {
//            current_scores[0] = -0.095734;
//            current_scores[1] = 0.0300723;
//            current_scores[2] = 0.0300723;
//            current_scores[3] = -0.0771052;
//            current_scores[4] = -0.0957761;
//            current_scores[5] = 0.0973957;
//            current_scores[6] = 0.0973957;
//            current_scores[7] = -0.0234477;
//            current_scores[8] = -0.0296639;
//            current_scores[9] = -0.0735101;
//        }

        vector<double> result = ranker->eval(current_scores.data());
        string tmp = "Iteration " + to_string(iter) + ":";
        for (size_t i = 0 ; i < result.size(); ++i) {
            tmp += "\tNDCG@" + to_string(config->eval_at[i]) + "=" + to_string(result[i]);
        }
        Log::Info(tmp.c_str());
    }

    return model;
}

}
