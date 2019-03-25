#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "lambdamart.h"

using namespace std;

void demo() {
    lambdamart::Dataset* X_train = lambdamart::Dataset::load_dataset("data/demo.train");
    lambdamart::Binner* binner = X_train->get_binner();
    lambdamart::Dataset* X_test = lambdamart::Dataset::load_dataset("data/demo.test", binner);

    // lambdamart::Model* model = lambdamart::train(X_train, params);
    // std::vector<double>* predictions = model->predict(X_test);

}

int main() {
    std::cout << lambdamart::version() << std::endl;

    demo();

    // TODO: support (1) commandline arguments (2) reading from configuration file

    return 0;
}