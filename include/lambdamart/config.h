#ifndef LAMBDAMART_CONFIG_H
#define LAMBDAMART_CONFIG_H

#include <lambdamart/common.h>
#include <lambdamart/log.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <fstream>
#include <cstring>
#include <unordered_set>
#include <algorithm>
#include <memory>

using namespace std;

namespace LambdaMART {

    class Config {
    public:
        string ToString() const;
        inline bool GetString(const string& name, string* out);
        inline bool GetInt(const string& name, int* out);
        inline bool GetDouble(const string& name, double* out);
        inline bool GetFloat(const string& name, float* out);
        inline bool GetBool(const string& name, bool* out);
        inline bool GetIntVector(const string& name, vector<int>* out);

        explicit Config(const char* filepath = nullptr) {
            ifstream infile(filepath);
            string line;
            if (infile.is_open()) {
                while (getline(infile, line)) {
                    istringstream row(line);
                    string token = row.str();
                    size_t delimiter = token.find(':');
                    string key = Common::Trim(token.substr(0, delimiter));
                    string val = Common::Trim(token.substr(delimiter + 1, token.length()));
                    if (key.find("#") == 0) continue;  // ignore comments
                    this->properties[key] = val;
                }
            } else {
                Log::Fatal("Unable to read configuration file %s", filepath);
            }
            infile.close();

            GetString("train_data", &train_data);
            GetString("train_query", &train_query);
            GetString("train_label", &train_label);
            GetString("valid_data", &valid_data);
            GetString("valid_query", &valid_query);
            GetInt("num_iterations", &num_iterations);
            GetFloat("learning_rate", &learning_rate);
            GetInt("max_depth", &max_depth);
            if(max_depth < 2)
                Log::Fatal("Max_depth should not be less than 2");
            GetInt("max_splits", &max_splits);
            GetInt("min_data_in_leaf", &min_data_in_leaf);
            GetFloat("min_impurity_to_split", &min_impurity_to_split);
            GetFloat("min_gain_to_split", &min_gain_to_split);
            GetInt("verbosity", &verbosity);
            Log::ResetLogLevel(LogLevel(verbosity));
            { int t; GetInt("max_bin", &t) && (max_bin = t > 255 ? 255 : t); }
            GetInt("min_data_in_bin", &min_data_in_bin);
            GetString("output_model", &output_model);
            GetString("output_result", &output_result);
            GetFloat("sigmoid", &sigmoid);
            GetInt("max_position", &max_position);
            GetInt("max_label", &max_label);
            GetIntVector("eval_at", &eval_at);
            GetInt("eval_interval", &eval_interval);
        }

#pragma region Parameters
        unordered_map<string, string> properties;

#pragma region Core Parameters
        string train_data, train_query, train_label;
        string valid_data, valid_query;
        int num_iterations = 100;
        float learning_rate = 0.1;

#pragma endregion

#pragma region Learning Control Parameters

        int max_depth = 9;
        int max_splits = 256;
        int min_data_in_leaf = 1;
        float min_gain_to_split = 1e-6;
        float min_impurity_to_split = 1e-6;

#pragma endregion

#pragma region IO Parameters

        // desc = ``< 0``: Fatal, ``= 0``: Error (Warning), ``= 1``: Info, ``> 1``: Debug, ``> 2``: Trace
        int verbosity = 1;

        // desc = max number of bins that feature values will be bucketed in
        uint8_t max_bin = 255;
        int min_data_in_bin = 3; //unused now in v1

        // desc = max cache size in MB for historical histogram; ``< 0`` means no limit
//        float histogram_pool_size = -1.0;  // TODO: maybe useful later

        string output_model = "model.txt";
        string output_result = "predict_result.txt";

#pragma endregion

#pragma region Objective Parameters

        // desc = parameter for the sigmoid function
        float sigmoid = 1.0;

        // desc = optimizes `NDCG <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`__ at this position
        // We assume that the number of sample in each query is at most 10000
        int max_position = 20;

        int max_label = 5;
        // default = 0,1,3,7,15,31,63,...,2^30-1
        // desc = relevant gain for labels. For example, the gain of label ``2`` is ``3`` in case of default label gains
        // desc = separate by ``,``
        vector<double> label_gain = {0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575,
                                    2097151, 4194303, 8388607, 16777215, 33554431, 67108863, 134217727, 268435455, 536870911};

        // default = 1,3,5
        // desc = used only with ``ndcg`` and ``map`` metrics
        std::vector<int> eval_at = {1, 3, 5, 10};

        // desc = evaluate training and validation ndcg every ``eval_interval`` iterations
        int eval_interval = 1;

        int num_feature_blocking = 4;

#pragma endregion

#pragma endregion

    private:
        void CheckParamConflict();
        void GetMembersFromString(const unordered_map<string, string>& properties);
        string SaveMembersToString() const;
    };

    inline bool Config::GetString(const string& name, string* out) {
        if (properties.count(name) > 0) {
            *out = properties.at(name);
            return true;
        }
        return false;
    }

    inline bool Config::GetInt(const string& name, int* out) {
        if (properties.count(name) > 0) {
            if (!Common::AtoiAndCheck(properties.at(name).c_str(), out)) {
                Log::Fatal("Parameter %s should be of type int, got \"%s\"",
                           name.c_str(), properties.at(name).c_str());
            }
            return true;
        }
        return false;
    }

    inline bool Config::GetDouble(const std::string& name, double* out) {
        if (properties.count(name) > 0) {
            float fout;
            if (!Common::AtofAndCheck(properties.at(name).c_str(), &fout)) {
                Log::Fatal("Parameter %s should be of type double, got \"%s\"",
                           name.c_str(), properties.at(name).c_str());
            }
            *out = fout;
            return true;
        }
        return false;
    }

    inline bool Config::GetFloat(const std::string& name, float* out) {
        if (properties.count(name) > 0) {
            if (!Common::AtofAndCheck(properties.at(name).c_str(), out)) {
                Log::Fatal("Parameter %s should be of type float, got \"%s\"",
                           name.c_str(), properties.at(name).c_str());
            }
            return true;
        }
        return false;
    }
    
    inline bool Config::GetBool(const std::string& name, bool* out) {
        if (properties.count(name) > 0) {
            std::string value = properties.at(name);
            std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
            if (value == std::string("false") || value == std::string("-")) {
                *out = false;
            } else if (value == std::string("true") || value == std::string("+")) {
                *out = true;
            } else {
                Log::Fatal("Parameter %s should be \"true\"/\"+\" or \"false\"/\"-\", got \"%s\"",
                           name.c_str(), properties.at(name).c_str());
            }
            return true;
        }
        return false;
    }

    inline bool Config::GetIntVector(const string& name, vector<int>* out) {
        if (properties.count(name) > 0) {
            std::string value = properties.at(name);
            std::vector<int> ret = Common::StringToArray<int>(value, ',');
            if (!ret.empty()) {
                *out = ret;
                return true;
            }
        }
        return false;
    }

}

#endif //LAMBDAMART_CONFIG_H
