#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <lambdamart/lambdamart.h>

using namespace std;

void demo() {
    auto* config = new LambdaMART::Config();

    LambdaMART::Log::Info("Loading training dataset %s and query boundaries %s", "data/mq2008/small.train", "data/mq2008/small.train.query");
    auto* X_train = new LambdaMART::Dataset(config);
    X_train->load_dataset("data/demo.train", "data/demo.train.query", 300);

    LambdaMART::Model* model = (new LambdaMART::Booster(X_train, config))->train();


    LambdaMART::Log::Info("Loading test dataset %s and query boundaries %s", "data/mq2008/small.vali", "data/mq2008/small.vali.query");
    // TODO: load raw dataset for testing data
//    auto* X_test = new LambdaMART::RawDataset();
//    X_test->load_raw_dataset("data/demo.test", "data/demo.test.query");
//    double* predictions = model->predict(X_test);
}

int main(int argc, char** argv) {
    cout << LambdaMART::version() << endl;

    if (argc <= 1)
        cout << LambdaMART::help() << endl;
    else {
        LambdaMART::Log::Info("Using configuration file %s", argv[1]);
    }

    demo();

    // TODO: support (1) commandline arguments (2) reading from configuration file

    return 0;
}