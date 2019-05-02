#ifndef LAMBDAMART_DATASET_H
#define LAMBDAMART_DATASET_H

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <cstring>
#include <array>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <cmath>

#include <lambdamart/types.h>
#include <lambdamart/config.h>

using namespace std;

namespace LambdaMART {

    struct Binner{
        // NOTE: Features are assumed to be arranged accordingly
        vector<vector<double>> thresholds;
    };

    class feature {
    public:
        vector<int> bin_index;
        vector<pair<double, int>> samples;
        vector<int> sample_index;
        vector<double> sample_data;
        vector<double> threshold;

        explicit feature(const uint8_t bin_cnt){
            this->threshold.resize(bin_cnt, -1);
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
             bin_index.resize(n, -1);

             for (int i = 0; i < n; i++) {
                 if (curr_count++ <= bin_size | fabs(this->sample_data[i - 1] - this->sample_data[i]) < 0.00001)
                     bin_index[this->sample_index[i]] = bin_count;
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

        vector<int> get_nonzero_bin_idx(){
            vector<int> nnz_bin_index;
            for(auto & i: this->sample_index)
                nnz_bin_index.emplace_back(this->bin_index[i]);
            return nnz_bin_index;
        }
    };

    class Dataset {
        vector<feature> data; // feature major; d rows, n columns
        int bin_size, bin_cnt;
        Binner binner;

    protected:
        void load_data_from_file(const char* path, vector<vector<pair<int, double>>>& data, vector<label_t> &rank, int& max_d){
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
                        max_d = max(max_d, index+1);
                    }
                    data.emplace_back(record);
                }
            } else {
                Log::Fatal("Cannot open file %s", path);
            }
            infile.close();
        }

        void load_query_from_file(const char* path){
            sample_t sum = 0;
            query_boundaries.emplace_back(0);
            ifstream infile(path);
            string line;
            if(infile.is_open()){
                while(getline(infile, line)){
                    sample_t query_size = stoi(line);
                    sum += query_size;
                    query_boundaries.emplace_back(sum);
                    if (query_size > max_query_size) max_query_size = query_size;
                }
            } else {
                Log::Fatal("Cannot open file %s", path);
            }
            infile.close();
        }

    public:
        int n, d;
        vector<label_t> rank;
        vector<sample_t> query_boundaries;
        sample_t max_query_size = 0;

        explicit Dataset(Config* config = nullptr){
            bin_cnt = config ? config->max_bin : 16;
        }

        auto& get_data() const{
            return data;
        }

        int max_label(){
            return *max_element(rank.begin(), rank.end());
        }

        void load_dataset(const char* data_path, const char* query_path) {
            vector<vector<pair<int, double>>> raw_data;
            load_data_from_file(data_path, raw_data, this->rank, this->d);
            this->n = raw_data.size();
            this->bin_size = (int)(n/bin_cnt);
            init_data();

            load_query_from_file(query_path);

            int row_index = 0;
            for(auto & row: raw_data){
                for(auto & entry: row)
                    this->data[entry.first].samples[row_index].first = entry.second;
                row_index++;
            }

            // delete raw_data
            vector<vector<pair<int, double>>>().swap(raw_data);

            for(auto & feat: this->data) {
                feat.sort();
                feat.bin(this->bin_size, this->n);
                this->binner.thresholds.emplace_back(feat.threshold);
            }
            Log::Info("Loaded dataset of size: %d samples x %d features", this->n, this->d);
        }

        // initialize a sample-major `n x d` matrix
        void init_data(){
            for(int i=0; i<this->d; i++){
                feature f = feature(bin_cnt);
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
        inline const sample_t* get_query_boundaries() const {
            return query_boundaries.data();
        }

        sample_t num_queries() const {
            return query_boundaries.size() - 1;
        }

        sample_t num_samples() const {
            return n;
        }

        label_t* get_labels() {
            return rank.data();
        }

        int num_bins() const {
            return this->bin_cnt;
        }
    };

    class RawDataset: public Dataset{
    private:
        vector<vector<pair<int, double>>> raw_data;
        vector<vector<double>> data;
    public:
        void load_dataset(const char* data_path, const char* query_path) {
            load_data_from_file(data_path, raw_data, this->rank, this->d);
            this->n = raw_data.size();
            init_data();

            load_query_from_file(query_path);

            for(int i=0; i<n; i++)
                data.emplace_back(vector<double>(this->d, 0));

            int row_index = 0;
            for (auto &row: raw_data) {
                for (auto &entry: row)
                    this->data[row_index][entry.first] = entry.second;
                row_index++;
            }
            Log::Info("Loaded dataset of size: %d samples x %d features", this->n, this->d);
        }

        const vector<double>& get_sample_row(sample_t id) {
            return data[id];
        }
    };
}
#endif //LAMBDAMART_DATASET_H
