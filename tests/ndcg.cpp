#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <cmath>

#include <lambdamart/lambdamart.h>


using namespace std;
using namespace LambdaMART;

void test1() {
    LambdaMART::Config* config = new LambdaMART::Config();
    
    LambdaMART::Model* model = new LambdaMART::Model();
    //model->train(*X_train, *config);
    //double* predictions = model->predict(X_test);
    
    int num_samples = 11;
    label_t* label = new label_t[num_samples];
    label[0] = (label_t)3;
    label[1] = (label_t)2;
    label[2] = (label_t)3;
    label[3] = (label_t)0;
    label[4] = (label_t)1;
    label[5] = (label_t)2;

    label[6] = (label_t)2;
    label[7] = (label_t)2;
    label[8] = (label_t)1;

    label[9] = (label_t)0;
    label[10] = (label_t)3;
    sample_t* bounds = new sample_t[4];
    bounds[0] = (sample_t) 0;
    bounds[1] = (sample_t) 6;
    bounds[2] = (sample_t) 9;
    bounds[3] = (sample_t) 11;   

    // int num_samples = 2;
    // label_t* label = new label_t[num_samples];
    // label[0] = (label_t)0;
    // label[1] = (label_t)3;

    // sample_t* bounds = new sample_t[2];
    // bounds[0] = (sample_t) 0;
    // bounds[1] = (sample_t) 2;

    // int num_samples = 8;
    // label_t* label = new label_t[num_samples];
    // label[0] = (label_t)3;
    // label[1] = (label_t)2;
    // label[2] = (label_t)3;
    // label[3] = (label_t)0;
    // label[4] = (label_t)1;
    // label[5] = (label_t)2;

    // label[6] = (label_t)0;
    // label[7] = (label_t)3;

    // sample_t* bounds = new sample_t[3];
    // bounds[0] = (sample_t) 0;
    // bounds[1] = (sample_t) 6;
    // bounds[2] = (sample_t) 8;

    std::vector<double> currentscores(num_samples, 0.0);
    std::vector<double> gradients(num_samples, 0.0);
    std::vector<double> hessians(num_samples, 0.0);
    config->max_position = 1000;
    config->max_label = 3;
    LambdaMART::LambdaRank* ranker = new LambdaMART::LambdaRank(bounds, (sample_t)3, label, *config);
    ranker->get_derivatives(currentscores.data(), gradients.data(), hessians.data());
    for (int i = 0; i < num_samples; ++i) {
        assert (gradients[i] != 0.0);
        assert (hessians[i] != 0.0);
        //cout << currentscores[i] << " " << gradients[i] << " " << hessians[i] << endl;
    }
    // Simple assert statements based on assumptions on the value of the gradients
    assert(gradients[9] > gradients[10]);
    assert(gradients[3] > gradients[4]);
    assert(gradients[8] > gradients[7]);
    assert(gradients[7] > gradients[6]);
}

void test2() {
    LambdaMART::Config* config = new LambdaMART::Config();
    LambdaMART::Model* model = new LambdaMART::Model();

    int num_samples = 2;
    label_t* label = new label_t[num_samples];
    label[0] = (label_t)0;
    label[1] = (label_t)3;

    sample_t* bounds = new sample_t[2];
    bounds[0] = (sample_t) 0;
    bounds[1] = (sample_t) 2;

    std::vector<double> currentscores(num_samples, 0.0);
    currentscores[0] = 0;
    currentscores[1] = 3;
    std::vector<double> gradients(num_samples, 0.0);
    std::vector<double> hessians(num_samples, 0.0);
    config->max_position = 1000;
    config->max_label = 3;
    LambdaMART::LambdaRank* ranker = new LambdaMART::LambdaRank(bounds, (sample_t)1, label, *config);
    ranker->get_derivatives(currentscores.data(), gradients.data(), hessians.data());
    for (int i = 0; i < num_samples; ++i) {
        assert (gradients[i] != 0.0 && fabs(gradients[i]) < 0.001);
        assert (hessians[i] != 0.0);
        //cout << currentscores[i] << " " << gradients[i] << " " << hessians[i] << endl;
    }
}

void test3() {
    LambdaMART::Model* model = new LambdaMART::Model();
    auto* config = new LambdaMART::Config();
    auto* X_train = new LambdaMART::Dataset(config);
    X_train->load_dataset("/Volumes/Data/chen/amaris/ethz/spr2019/fastcode/project/LambdaMART/data/rank.test", "/Volumes/Data/chen/amaris/ethz/spr2019/fastcode/project/LambdaMART/data/rank.test.query", 300);
    sample_t num_samples = X_train->num_samples();
    sample_t num_queries = X_train->num_queries();
    label_t* label = X_train->get_labels();
    const sample_t* bounds = X_train->query_boundaries();
    config->max_position = num_samples;
    config->max_label = 4;

    std::vector<double> currentscores(num_samples, 0.0);
    std::vector<double> gradients(num_samples, 0.0);
    std::vector<double> hessians(num_samples, 0.0);
    std::cout << num_samples << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << label[i] << std::endl;
    }
    LambdaMART::LambdaRank* ranker = new LambdaMART::LambdaRank(bounds, num_queries, label, *config);
    ranker->get_derivatives(currentscores.data(), gradients.data(), hessians.data());
    for (int i = 0; i < num_samples; ++i) {
        assert (gradients[i] != 0.0);
        assert (hessians[i] != 0.0);
        cout << currentscores[i] << " " << gradients[i] << " " << hessians[i] << endl;
    }
}

int main() {
    std::cout << LambdaMART::version() << std::endl;

    test1();
    test2();
    test3();
    return 0;
}