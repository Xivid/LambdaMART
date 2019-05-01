#ifndef LAMBDAMART_DATASET_H
#define LAMBDAMART_DATASET_H
#define BIN_CNT 64

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <cstring>
#include <array>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <cstdlib>
#include <iostream>

#include <lambdamart/types.h>

using namespace std;

namespace LambdaMART {

    struct Binner{
        // NOTE: Features are assumed to be arranged accordingly
        vector<vector<double>> thresholds;
    };

    class feature {
    public:
        vector<uint8_t> bin_index;
        vector<pair<double, int>> samples;
        vector<int> sample_index;
        vector<double> sample_data;
        vector<double> threshold;

        feature(){
            this->threshold.resize(BIN_CNT, -1);
        }

        //sorts and extracts samples and indexes from vector of pairs
        void sort() {
            std::sort(this->samples.begin(), this->samples.end());

            for(auto & sample: this->samples) {
                this->sample_data.emplace_back(sample.first);
                this->sample_index.emplace_back(sample.second);
            }
            // delete samples
            vector<pair<double, int>>().swap(this->samples);
        }

         //creates bins with sizes "bin_size" and also calculates threshold values that split the bins
         void bin(int bin_size, int n) {
             int curr_count = 0, bin_count = 0;
             // for first value
             bin_index.emplace_back(bin_count);

             for (int i = 1; i < n; i++) {
                 if (curr_count++ <= bin_size | abs(this->sample_data[i - 1] - this->sample_data[i]) < 0.00001)
                     bin_index.emplace_back(bin_count);
                 else {
                     curr_count = 0;
                     i--;
                     threshold[bin_count] = i < n-1 ? (this->sample_data[i] + this->sample_data[i + 1]) / 2.0 : this->sample_data[i] + 0.1;
                     bin_count++;
                 }
             }
         }

        //overloaded bin for using predefined threshold values, using binner class and index of feature being binned.
        void bin(int n, Binner *binner, int f) {
            this->bin_index.resize(n, -1);
            vector<double> thresholds = binner->thresholds[f];
            int bin_count = 0;
            for (int i = 0; i < n; i++) {
                if(sample_data[i] <= thresholds[bin_count])
                    bin_index[i] = bin_count;
                else{
                    bin_count++;
                    i--;
                }
            }
        }
    };


    class Dataset {
        vector<feature> data; // feature major; d rows, n columns
        int n, d, bin_size;
        Binner binner;
        vector<int> rank;
        vector<sample_t> query;
        vector<sample_t> query_boundaries_;

        static void load_data_from_file(const char* path, vector<vector<pair<int, double>>>& data, vector<int> &rank){
            ifstream infile(path);
            string line;
            if(infile.is_open()){
                while(getline(infile, line)){
                    vector<pair<int, double>> record;
                    istringstream row(line);
                    vector<string> tokens{istream_iterator<string>{row}, istream_iterator<string>{}};
                    for(auto & token: tokens){
                        int delimiter = token.find(':');
                        if(delimiter == string::npos) {
                            rank.emplace_back(stoi(token));
                            continue;
                        }
                        int index = stoi(token.substr(0, delimiter)) - 1;
                        double val = stof(token.substr(delimiter+1, token.length()));
                        record.emplace_back(make_pair(index, val));
                    }
                    data.emplace_back(record);
                }
            }
            infile.close();
        }

        static void load_query_from_file(const char* path, vector<sample_t>& q){
            ifstream infile(path);
            string line;
            if(infile.is_open()){
                while(getline(infile, line)){
                    q.emplace_back(stoi(line));
                }
            }
            infile.close();
        }

    public:
        void load_dataset(const char* data_path, const char* query_path = nullptr, int num_features = -1, Binner* binner = nullptr) {
            if(num_features == -1){
                // TODO: calculate num_features;
            }
            this->d = num_features;
            vector<vector<pair<int, double>>> raw_data;
            load_data_from_file(data_path, raw_data, this->rank);
            this->n = raw_data.size();
            this->bin_size = (int)(n/BIN_CNT);
            vector<feature> processed_data(d);
            init_data();

            load_query_from_file(query_path, this->query);

            int row_index = 0;
            for(auto & row: raw_data){
                for(auto & entry: row)
                    this->data[entry.first].samples[row_index].first = entry.second;
                row_index++;
            }

            // delete raw_data
            vector<vector<pair<int, double>>>().swap(raw_data);

            if(!binner)
                for(auto & feat: this->data) {
                    feat.sort();
                    feat.bin(this->bin_size, this->n);
                    this->binner.thresholds.emplace_back(feat.threshold);
                }
            else {
                int f = 0;
                for (auto & feat: this->data) {
                    feat.sort();
                    feat.bin(this->n, binner, f++);
                }
            }
        }

        // initialize a sample-major `n x d` matrix
        void init_data(){
            this->data.reserve(this->d);
            for(int i=0; i<this->d; i++){
                feature f = feature();
                for(int k=0; k<this->n; k++)
                    f.samples.emplace_back(make_pair(0.0, k));
                this->data.emplace_back(f);
            }
        }

        // returns dimensions of raw data
        pair<int, int> shape() const{
            return make_pair(this->n, this->d);
        }

        // return binner
        Binner* get_binner(){
            return &(this->binner);
        }

        // query boundaries (the first sample_id of each query)
        inline const sample_t* query_boundaries() const {
            if (!query_boundaries_.empty()) return query_boundaries_.data();
            else return nullptr;   
        }

        const vector<sample_t>& get_queries() const {
            return this->query;
        }

        sample_t num_queries() const {
            return this->query.size();
        }

        sample_t num_samples() const {
            return this->n;
        }

        // returns pointer to labels
        label_t* label() const {
//            return this->rank;
            return nullptr;
        }
    };
}
#endif //LAMBDAMART_DATASET_H
