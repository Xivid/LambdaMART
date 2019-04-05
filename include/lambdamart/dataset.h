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

        feature() : threshold(BIN_CNT) {}

        void sort(){
            std::sort(samples.begin(), samples.end());

            for(auto & sample: samples) {
                this->sample_data.emplace_back(sample.first);
                this->sample_index.emplace_back(sample.second);
            }

            // delete samples
            vector<pair<double, int>>().swap(samples);
        }

         //updates threshold & bin_index
        void bin(int bin_size, int n){
            int curr_count = 0, bin_count = 0;
            this->bin_index.resize(n, -1);

            for(int i=0; i<n; i++){
                if(curr_count++ <= bin_size | sample_data[i-1] == sample_data[i]){
                    bin_index.emplace_back(bin_count);
                }
                else{
                    curr_count = 0;
                    i--;
                    threshold[bin_count] = (sample_data[i] + sample_data[i+1]) / 2.0;
                    bin_count++;
                }
            }
        }

         void bin(int bin_size, int n, Binner* binner, int f){
             this->bin_index.resize(n, -1);
             vector<double>& thresholds = binner->thresholds[f];
             int curr_count = 0, bin_count = 0;
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
        vector<feature> data; // d rows, n columns
        int n, d, bin_size;
        Binner binner;
        vector<int> query, rank;

        static void load_data_from_file(const char* path, vector<vector<pair<int, double>>>& data, vector<int> &rank){
            ifstream infile(path);
            string line;
            if(infile.is_open()){
                vector<pair<int, double>> record;
                while(getline(infile, line)){
                    istringstream row(line);
                    vector<string> tokens{istream_iterator<string>{row}, istream_iterator<string>{}};
                    for(auto & token: tokens){
                        int delimiter = token.find(':');
                        if(delimiter == string::npos) {
                            rank.emplace_back(stoi(token));
                            continue;
                        }
                        int index = stoi(token.substr(0, delimiter));
                        double val = stof(token.substr(delimiter+1, token.length()));
                        record.emplace_back(make_pair(index, val));
                    }
                    data.push_back(record);
                }
            }
            infile.close();
        }

        static void load_query_from_file(const char* path, vector<int>& q){
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
                this->d = num_features;
            }

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
                for (auto &feat: this->data) {
                    feat.sort();
                    feat.bin(this->bin_size, this->n, binner, f++);
                }
            }
        }

        // initialize `n x d` matrix
        void init_data(){
            this->data.reserve(this->d);
            for(int i=0; i<this->d; i++){
                this->data[i].samples = vector<pair<double, int>>();
                for(int k=0; k<this->n; k++)
                    this->data[i].samples.emplace_back(make_pair(0.0, k));
            }
        }

        // returns dimensions of raw data
        pair<int, int> shape(){
            return make_pair(this->n, this->d);
        }

        // return binner
        Binner* get_binner(){
            return &(this->binner);
        }

        // query boundaries (the first sample_id of each query)
        const vector<int>& get_boundaries() const {
            return this->query;
        }

        int get_num_samples() const {
            return this->n;
        }
    };
}
#endif //LAMBDAMART_DATASET_H
