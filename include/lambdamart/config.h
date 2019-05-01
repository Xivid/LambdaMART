#ifndef LAMBDAMART_CONFIG_H
#define LAMBDAMART_CONFIG_H

#include <lambdamart/common.h>
#include <lambdamart/log.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>

using namespace std;

namespace LambdaMART {

    enum TaskType {
        Train, Predict, KRefitTree
    };
    const int kDefaultNumLeaves = 31;

    class Config {
    public:
        string ToString() const;
        inline static bool GetString(
                const unordered_map<string, string>& params,
                const string& name, string* out);
        inline static bool GetInt(
                const unordered_map<string, string>& params,
                const string& name, int* out);
        inline static bool GetDouble(
                const unordered_map<string, string>& params,
                const string& name, double* out);
        inline static bool GetBool(
                const unordered_map<string, string>& params,
                const string& name, bool* out);

        static void KV2Map(unordered_map<string, string>& params, const char* kv);
        static unordered_map<string, string> Str2Map(const char* parameters);

#pragma region Parameters

#pragma region Core Parameters

        string config_file= "";
        TaskType task = TaskType::Train;

        // desc = ``lambdarank``, `lambdarank <https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf>`__ application
        // descl2 = label should be ``int`` type in lambdarank tasks, and larger number represents the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)
        // descl2 = `label_gain <#objective-parameters>`__ can be used to set the gain (weight) of ``int`` label
        // descl2 = all values in ``label`` must be smaller than number of elements in ``label_gain``
        string objective = "lambdarank";
        string boosting = "gbdt";
        string train_data = "";
        string valid_data = "";
        int num_iterations = 100;
        double learning_rate = 0.1;
        int num_leaves = kDefaultNumLeaves;

        // options = serial, feature, data, voting
        // desc = ``serial``, single machine tree learner
        // desc = ``feature``, feature parallel tree learner, aliases: ``feature_parallel``
        // desc = ``data``, data parallel tree learner, aliases: ``data_parallel``
        // desc = ``voting``, voting parallel tree learner, aliases: ``voting_parallel``
        // desc = refer to `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__ to get more details
        string tree_learner = "serial";

        const int num_threads = 0;
        const string device_type = "cpu";
        int seed = 0;

#pragma endregion

#pragma region Learning Control Parameters

        int max_depth = -1;
        int min_data_in_leaf = 20;
        double min_sum_hessian_in_leaf = 1e-3;
        double bagging_fraction = 1.0;

        // desc = frequency for bagging
        // desc = ``0`` means disable bagging; ``k`` means perform bagging at every ``k`` iteration
        // desc = **Note**: to enable bagging, ``bagging_fraction`` should be set to value smaller than ``1.0`` as well
        int bagging_freq = 0;
        int bagging_seed = 3;

        // desc = LightGBM will randomly select part of features on each iteration if ``feature_fraction`` smaller than ``1.0``. For example, if you set it to ``0.8``, LightGBM will select 80% of features before training each tree
        double feature_fraction = 1.0;
        int feature_fraction_seed = 2;
        int early_stopping_round = 0;

        // desc = used to limit the max output of tree leaves
        double max_delta_step = 0.0;
        // double lambda_l1 = 0.0;
        // double lambda_l2 = 0.0;

        // desc = the minimal gain to perform split
        double min_gain_to_split = 0.0;

        // desc = minimal number of data per categorical group
        int min_data_per_group = 100;

        // desc = used for the categorical features
        int max_cat_threshold = 32;

        // desc = L2 regularization in categorcial split
        double cat_l2 = 10.0;

        // desc = this can reduce the effect of noises in categorical features, especially for categories with few data
        double cat_smooth = 10.0;

        // check = >0
        // desc = when number of categories of one feature smaller than or equal to ``max_cat_to_onehot``, one-vs-other split algorithm will be used
        int max_cat_to_onehot = 4;

        // desc = used in `Voting parallel <./Parallel-Learning-Guide.rst#choose-appropriate-parallel-algorithm>`__
        // desc = set this to larger value for more accurate result, but it will slow down the training speed
        int top_k = 20;

        // desc = used for constraints of monotonic features
        // desc = ``1`` means increasing, ``-1`` means decreasing, ``0`` means non-constraint
        // desc = you need to specify all features in order. For example, ``mc=-1,0,1`` means decreasing for 1st feature, non-constraint for 2nd feature and increasing for the 3rd feature
        vector<int8_t> monotone_constraints;

        // desc = used to control feature's split gain, will use ``gain[i] = max(0, feature_contri[i]) * gain[i]`` to replace the split gain of i-th feature
        vector<double> feature_contri;

        // desc = path to a ``.json`` file that specifies splits to force at the top of every decision tree before best-first learning commences
        // desc = ``.json`` file can be arbitrarily nested, and each split contains ``feature``, ``threshold`` fields, as well as ``left`` and ``right`` fields representing subsplits
        // desc = categorical splits are forced in a one-hot fashion, with ``left`` representing the split containing the feature value and ``right`` representing other values
        // desc = **Note**: the forced split logic will be ignored, if the split makes gain worse
        // desc = see `this file <https://github.com/Microsoft/LightGBM/tree/master/examples/binary_classification/forced_splits.json>`__ as an example
        string forcedsplits_filename = "";

        // desc = decay rate of ``refit`` task, will use ``leaf_output = refit_decay_rate * old_leaf_output + (1.0 - refit_decay_rate) * new_leaf_output`` to refit trees
        // desc = used only in ``refit`` task in CLI version or as argument in ``refit`` function in language-specific package
        double refit_decay_rate = 0.9;

#pragma endregion

#pragma region IO Parameters

        // desc = ``< 0``: Fatal, ``= 0``: Error (Warning), ``= 1``: Info, ``> 1``: Debug
        int verbosity = 1;

        // desc = max number of bins that feature values will be bucketed in
        uint8_t max_bin = 255;
        int min_data_in_bin = 3;

        // desc = number of data that sampled to construct histogram bins
        // desc = setting this to larger value will give better training result, but will increase data loading time
        // desc = set this to larger value if data is very sparse
        int bin_construct_sample_cnt = 200000;

        // desc = max cache size in MB for historical histogram; ``< 0`` means no limit
        double histogram_pool_size = -1.0;

        // desc = random seed for data partition in parallel learning (excluding the ``feature_parallel`` mode)
        int data_random_seed = 1;

        string output_model = "LightGBM_model.txt";
        string input_model = "";
        int snapshot_freq = -1;
        string output_result = "LightGBM_predict_result.txt";

        // desc = path of file with training initial scores
        // desc = if ``""``, will use ``train_data_file`` + ``.init`` (if exists)
        string initscore_filename = "";

        // alias = valid_data_init_scores, valid_init_score_file, valid_init_score
        // default = ""
        // desc = path(s) of file(s) with validation initial scores
        // desc = if ``""``, will use ``valid_data_file`` + ``.init`` (if exists)
        // desc = separate by ``,`` for multi-validation data
        vector<string> valid_data_initscores;

        // alias = is_pre_partition
        // desc = used for parallel learning (excluding the ``feature_parallel`` mode)
        // desc = ``true`` if training data are pre-partitioned, and different machines use different partitions
        const bool pre_partition = false;

        // desc = set this to ``false`` to disable Exclusive Feature Bundling (EFB), which is described in `LightGBM: A Highly Efficient Gradient Boosting Decision Tree <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree>`__
        bool enable_bundle = true;

        // desc = max conflict rate for bundles in EFB
        // desc = set this to ``0.0`` to disallow the conflict and provide more accurate results
        // desc = set this to a larger value to achieve faster speed
        double max_conflict_rate = 0.0;

        // alias = is_sparse, enable_sparse, sparse
        // desc = used to enable/disable sparse optimization
        bool is_enable_sparse = true;

        // desc = the threshold of zero elements percentage for treating a feature as a sparse one
        double sparse_threshold = 0.8;

        // desc = set this to ``false`` to disable the special handle of missing value
        bool use_missing = true;

        // desc = set this to ``true`` to treat all zero as missing values (including the unshown values in libsvm/sparse matrices)
        // desc = set this to ``false`` to use ``na`` for representing missing values
        bool zero_as_missing = false;

        // desc = set this to ``true`` if data file is too big to fit in memory
        // desc = by default, LightGBM will map data file to memory and load features from memory. This will provide faster data loading speed, but may cause run out of memory error when the data file is very big
        bool two_round = false;

        // desc = if ``true``, LightGBM will save the dataset (including validation data) to a binary file. This speed ups the data loading for the next time
        bool save_binary = false;

        // desc = set this to ``true`` to enable autoloading from previous saved binary datasets
        bool enable_load_from_binary_file = true;

        // desc = set this to ``true`` if input data has header
        bool header = false;

        // desc = used to specify the label column
        // desc = use number for index, e.g. ``label=0`` means column\_0 is the label
        // desc = add a prefix ``name:`` for column name, e.g. ``label=name:is_click``
        string label_column = "";

        // desc = used to specify the weight column
        // desc = use number for index, e.g. ``weight=0`` means column\_0 is the weight
        // desc = add a prefix ``name:`` for column name, e.g. ``weight=name:weight``
        string weight_column = "";

        // desc = used to specify the query/group id column
        // desc = use number for index, e.g. ``query=0`` means column\_0 is the query id
        // desc = add a prefix ``name:`` for column name, e.g. ``query=name:query_id``
        // desc = **Note**: data should be grouped by query\_id
        // desc = **Note**: index starts from ``0`` and it doesn't count the label column when passing type is ``int``, e.g. when label is column\_0 and query\_id is column\_1, the correct parameter is ``query=0``
        string group_column = "";

        // desc = used to specify some ignoring columns in training
        // desc = use number for index, e.g. ``ignore_column=0,1,2`` means column\_0, column\_1 and column\_2 will be ignored
        // desc = add a prefix ``name:`` for column name, e.g. ``ignore_column=name:c1,c2,c3`` means c1, c2 and c3 will be ignored
        string ignore_column = "";

        // desc = used to specify categorical features
        // desc = use number for index, e.g. ``categorical_feature=0,1,2`` means column\_0, column\_1 and column\_2 are categorical features
        // desc = add a prefix ``name:`` for column name, e.g. ``categorical_feature=name:c1,c2,c3`` means c1, c2 and c3 are categorical features
        std::string categorical_feature = "";

        // desc = used only in ``prediction`` task
        // desc = set this to ``true`` to predict only the raw scores
        // desc = set this to ``false`` to predict transformed scores
        bool predict_raw_score = false;

        // desc = used only in ``prediction`` task
        // desc = set this to ``true`` to predict with leaf index of all trees
        bool predict_leaf_index = false;

        // desc = used only in ``prediction`` task
        // desc = set this to ``true`` to estimate `SHAP values <https://arxiv.org/abs/1706.06060>`__, which represent how each feature contributes to each prediction
        // desc = produces ``#features + 1`` values where the last value is the expected value of the model output over the training data
        bool predict_contrib = false;

        // desc = used only in ``prediction`` task
        // desc = used to specify how many trained iterations will be used in prediction
        // desc = ``<= 0`` means no limit
        int num_iteration_predict = -1;

        // desc = used only in ``prediction`` task
        // desc = if ``true``, will use early-stopping to speed up the prediction. May affect the accuracy
        bool pred_early_stop = false;

        // desc = used only in ``prediction`` task
        // desc = the frequency of checking early-stopping prediction
        int pred_early_stop_freq = 10;

        // desc = used only in ``prediction`` task
        // desc = the threshold of margin in early-stopping prediction
        double pred_early_stop_margin = 10.0;

        // desc = used only in ``convert_model`` task
        // desc = if ``convert_model_language`` is set and ``task=train``, the model will be also converted
        string convert_model_language = "";

        // desc = used only in ``convert_model`` task
        // desc = output filename of converted model
        string convert_model = "gbdt_prediction.cpp";

#pragma endregion

#pragma region Objective Parameters

        int num_class = 1;
        bool is_unbalance = false;

        // desc = used only in ``binary`` application
        // desc = weight of labels with positive class
        double scale_pos_weight = 1.0;

        // desc = used only in ``binary`` and ``multiclassova`` classification and in ``lambdarank`` applications
        // desc = parameter for the sigmoid function
        double sigmoid = 1.0;

        // desc = optimizes `NDCG <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`__ at this position
        int max_position = 20;

        int max_label = 5;
        // default = 0,1,3,7,15,31,63,...,2^30-1
        // desc = relevant gain for labels. For example, the gain of label ``2`` is ``3`` in case of default label gains
        // desc = separate by ``,``
        vector<double> label_gain;

#pragma endregion

#pragma region Metric Parameters

        // desc = metric(s) to be evaluated on the evaluation set(s)
        // descl2 = ``""`` (empty string or not specified) means that metric corresponding to specified ``objective`` will be used (this is possible only for pre-defined objective functions, otherwise no evaluation metric will be added)
        // descl2 = ``"None"`` (string, **not** a ``None`` value) means that no metric will be registered, aliases: ``na``, ``null``, ``custom``
        // descl2 = ``l1``, absolute loss, aliases: ``mean_absolute_error``, ``mae``, ``regression_l1``
        // descl2 = ``l2``, square loss, aliases: ``mean_squared_error``, ``mse``, ``regression_l2``, ``regression``
        // descl2 = ``l2_root``, root square loss, aliases: ``root_mean_squared_error``, ``rmse``
        // descl2 = ``quantile``, `Quantile regression <https://en.wikipedia.org/wiki/Quantile_regression>`__
        // descl2 = ``mape``, `MAPE loss <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`__, aliases: ``mean_absolute_percentage_error``
        // descl2 = ``huber``, `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`__
        // descl2 = ``fair``, `Fair loss <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520>`__
        // descl2 = ``poisson``, negative log-likelihood for `Poisson regression <https://en.wikipedia.org/wiki/Poisson_regression>`__
        // descl2 = ``gamma``, negative log-likelihood for **Gamma** regression
        // descl2 = ``gamma_deviance``, residual deviance for **Gamma** regression
        // descl2 = ``tweedie``, negative log-likelihood for **Tweedie** regression
        // descl2 = ``ndcg``, `NDCG <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`__, aliases: ``lambdarank``
        // descl2 = ``map``, `MAP <https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/>`__, aliases: ``mean_average_precision``
        // descl2 = ``auc``, `AUC <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`__
        // descl2 = ``binary_logloss``, `log loss <https://en.wikipedia.org/wiki/Cross_entropy>`__, aliases: ``binary``
        // descl2 = ``binary_error``, for one sample: ``0`` for correct classification, ``1`` for error classification
        // descl2 = ``multi_logloss``, log loss for multi-class classification, aliases: ``multiclass``, ``softmax``, ``multiclassova``, ``multiclass_ova``, ``ova``, ``ovr``
        // descl2 = ``multi_error``, error rate for multi-class classification
        // descl2 = ``xentropy``, cross-entropy (with optional linear weights), aliases: ``cross_entropy``
        // descl2 = ``xentlambda``, "intensity-weighted" cross-entropy, aliases: ``cross_entropy_lambda``
        // descl2 = ``kldiv``, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__, aliases: ``kullback_leibler``
        // desc = support multiple metrics, separated by ``,``
        std::vector<std::string> metric;

        // alias = output_freq
        int metric_freq = 1;

        // desc = set this to ``true`` to output metric result over training dataset
        bool is_provide_training_metric = false;

        // default = 1,2,3,4,5
        // desc = used only with ``ndcg`` and ``map`` metrics
        std::vector<int> eval_at;

#pragma endregion

#pragma region Network Parameters

        const int num_machines = 1;
        const int local_listen_port = 12400;
        const int time_out = 120;
        const string machine_list_filename = "";
        const string machines = "";

#pragma endregion

#pragma region GPU Parameters

        const int gpu_platform_id = -1;
        const int gpu_device_id = -1;
        const bool gpu_use_dp = false;

#pragma endregion

#pragma endregion

        const bool is_parallel = false;
        const bool is_parallel_find_bin = false;
//        LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params);
        static unordered_map<string, string> alias_table;
        static unordered_set<string> parameter_set;

    private:
        void CheckParamConflict();
        void GetMembersFromString(const unordered_map<string, string>& params);
        string SaveMembersToString() const;
    };

    inline bool Config::GetString(
            const unordered_map<string, string>& params,
            const string& name, string* out) {
        if (params.count(name) > 0) {
            *out = params.at(name);
            return true;
        }
        return false;
    }

    inline bool Config::GetInt(
            const unordered_map<string, string>& params,
            const string& name, int* out) {
        if (params.count(name) > 0) {
            if (!Common::AtoiAndCheck(params.at(name).c_str(), out)) {
                Log::Fatal("Parameter %s should be of type int, got \"%s\"",
                           name.c_str(), params.at(name).c_str());
            }
            return true;
        }
        return false;
    }

    inline bool Config::GetDouble(
            const std::unordered_map<std::string, std::string>& params,
            const std::string& name, double* out) {
        if (params.count(name) > 0) {
            if (!Common::AtofAndCheck(params.at(name).c_str(), out)) {
                Log::Fatal("Parameter %s should be of type double, got \"%s\"",
                           name.c_str(), params.at(name).c_str());
            }
            return true;
        }
        return false;
    }

    inline bool Config::GetBool(
            const std::unordered_map<std::string, std::string>& params,
            const std::string& name, bool* out) {
        if (params.count(name) > 0) {
            std::string value = params.at(name);
            std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
            if (value == std::string("false") || value == std::string("-")) {
                *out = false;
            } else if (value == std::string("true") || value == std::string("+")) {
                *out = true;
            } else {
                Log::Fatal("Parameter %s should be \"true\"/\"+\" or \"false\"/\"-\", got \"%s\"",
                           name.c_str(), params.at(name).c_str());
            }
            return true;
        }
        return false;
    }

    struct ParameterAlias {
        static void KeyAliasTransform(std::unordered_map<std::string, std::string>* params) {
            std::unordered_map<std::string, std::string> tmp_map;
            for (const auto& pair : *params) {
                auto alias = Config::alias_table.find(pair.first);
                if (alias != Config::alias_table.end()) {  // found alias
                    auto alias_set = tmp_map.find(alias->second);
                    if (alias_set != tmp_map.end()) {  // alias already set
                        // set priority by length & alphabetically to ensure reproducible behavior
                        if (alias_set->second.size() < pair.first.size() ||
                            (alias_set->second.size() == pair.first.size() && alias_set->second < pair.first)) {
                            Log::Warning("%s is set with %s=%s, %s=%s will be ignored. Current value: %s=%s",
                                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), params->at(alias_set->second).c_str());
                        } else {
                            Log::Warning("%s is set with %s=%s, will be overridden by %s=%s. Current value: %s=%s",
                                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), pair.second.c_str());
                            tmp_map[alias->second] = pair.first;
                        }
                    } else {  // alias not set
                        tmp_map.emplace(alias->second, pair.first);
                    }
                } else if (Config::parameter_set.find(pair.first) == Config::parameter_set.end()) {
                    Log::Warning("Unknown parameter: %s", pair.first.c_str());
                }
            }
            for (const auto& pair : tmp_map) {
                auto alias = params->find(pair.first);
                if (alias == params->end()) {  // not find
                    params->emplace(pair.first, params->at(pair.second));
                    params->erase(pair.second);
                } else {
                    Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s",
                                 pair.first.c_str(), alias->second.c_str(), pair.second.c_str(), params->at(pair.second).c_str(),
                                 pair.first.c_str(), alias->second.c_str());
                }
            }
        }
    };

}   // namespace LightGBM

#endif //LAMBDAMART_CONFIG_H
