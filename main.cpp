#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <lambdamart/lambdamart.h>

using namespace std;

void demo() {
    LambdaMART::Dataset* X_train = new LambdaMART::Dataset();
    X_train->load_dataset("data/demo.train", "data/demo.train.query");
    LambdaMART::Binner* binner = X_train->get_binner();

    // TODO: testing data should be stored in sample-major format! Still using `Dataset` is inconvenient!
    LambdaMART::Dataset* X_test = new LambdaMART::Dataset();
    X_test->load_dataset("data/demo.test", "data/demo.test.query", -1, binner);
    LambdaMART::Config* config = new LambdaMART::Config();
    
    LambdaMART::Model* model = new LambdaMART::Model();
    model->train(*X_train, *config);
    std::vector<double>* predictions = model->predict(X_test);

}

int main() {
    std::cout << LambdaMART::version() << std::endl;

    demo();

    // TODO: support (1) commandline arguments (2) reading from configuration file

    return 0;
}