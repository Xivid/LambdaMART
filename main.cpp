#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <lambdamart/lambdamart.h>

using namespace std;
using namespace LambdaMART;

void demo(Config* config) {
    const char* train = config->train_data.c_str();
    const char* train_query = config->train_query.c_str();
    Log::Info("Loading training dataset %s and query boundaries %s", train, train_query);
    auto* X_train = new Dataset(config);
    X_train->load_dataset(train, train_query);

    RawDataset* X_valid = nullptr;
    if (!config->valid_data.empty()) {
        const char* vali = config->valid_data.c_str();
        const char* vali_query = config->valid_query.c_str();
        Log::Info("Loading validation dataset %s and query boundaries %s", vali, vali_query);
        X_valid = new RawDataset();
        X_valid->load_dataset(vali, vali_query);
    } else {
        Log::Info("No validation dataset");
    }

    Log::Info("Start training...");
    Model* model = (new Booster(X_train, X_valid, config))->train();
    Log::Info("Training finished.");

    Log::Info("Predicting with validation dataset and saving output to %s", config->output_result.c_str());
    vector<score_t> predictions = model->predict(X_valid, config->output_result);
}

int main(int argc, char** argv) {
    cout << version() << endl;

    if (argc <= 1) {
        cout << help() << endl;
        exit(0);
    }

    Log::Info("Using configuration file %s", argv[1]);
    auto* config = new Config(argv[1]);

    demo(config);

    return 0;
}
