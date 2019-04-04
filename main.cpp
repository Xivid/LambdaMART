#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <lambdamart/lambdamart.h>

using namespace std;

void demo() {
    LambdaMART::Dataset* X_train = new LambdaMART::Dataset();
    X_train->load_dataset("data/demo.train");
    LambdaMART::Binner* binner = X_train->get_binner();
    LambdaMART::Dataset* X_test = new LambdaMART::Dataset();
    X_test->load_dataset("data/demo.test", nullptr, -1, binner);

    LambdaMART::Model* model = new Model();
    model->train(X_train, params);
    std::vector<double>* predictions = model->predict(X_test);

}

int main() {
    std::cout << LambdaMART::version() << std::endl;

    demo();

    // TODO: support (1) commandline arguments (2) reading from configuration file

    return 0;
}