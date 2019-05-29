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
#include <climits>

#include <lambdamart/types.h>
#include <lambdamart/config.h>

using namespace std;

namespace LambdaMART {

    class Feature {
        int bin_cnt;
    public:
        vector<int> bin_index;
        vector<pair<double, int>> samples;
        vector<int> sample_index;
        vector<double> sample_data;
        vector<double> threshold;

        explicit Feature(const uint8_t bin_cnt){
            this->threshold.resize(bin_cnt, -1);
            this->bin_cnt = 0;
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
                 if (curr_count++ <= bin_size || fabs(this->sample_data[i - 1] - this->sample_data[i]) < 0.00001)
                     bin_index[this->sample_index[i]] = bin_count;
                 else {
                     curr_count = 0;
                     i--;
                     threshold[bin_count] = i < n-1 ? (this->sample_data[i] + this->sample_data[i + 1]) / 2.0 : numeric_limits<double>::max();
                     bin_count++;
                 }
             }
             this->bin_cnt = bin_count+1;
         }

        int bin_count() const {
            return this->bin_cnt;
        }
    };

    class Dataset {
        vector<Feature> data; // feature major; d rows, n columns
        int bin_size, bin_cnt, max_lbl;

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
                            int val = stoi(token);
                            this->max_lbl = max(max_lbl, val);
                            rank.emplace_back(val);
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
            this->max_lbl = INT_MIN;
            this->d = INT_MIN;
        }

        auto& get_data(int i) const{
            return data[i];
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
            }
            Log::Info("Loaded dataset of size: %d samples x %d features", this->n, this->d);
        }

        void load_debug_dataset(const char* data_path, const char* label_path, const char* query_path, int num_feat){
            ifstream infile(data_path);
            string line;
            if(infile.is_open()){
                while(getline(infile, line)){
                    vector<string> tokens;
                    istringstream row(line);
                    while(row.good()){
                        string substring;
                        getline(row, substring, '#');
                        tokens.emplace_back(substring);
                    }
                    if(tokens.size() != 3){
                        Log::Fatal("File syntax read incorrectly.");
                        exit(0);
                    }

                    vector<int> bins;
                    vector<double> thresholds;
                    stringstream proc(tokens[1]);
                    while(proc.good()){
                        string bin;
                        getline(proc, bin, ',');
                        bins.emplace_back(stoi(bin));
                    }

                    stringstream proc_(tokens[2]);
                    while(proc_.good()){
                        string thres;
                        getline(proc_, thres, ',');
                        thresholds.emplace_back(stod(thres));
                    }

                    Feature f = Feature(num_feat);
                    f.bin_index = bins;
                    f.threshold = thresholds;

                    this->data.emplace_back(f);
                }
            } else {
                Log::Fatal("Cannot open (data) file %s", data_path);
            }
            infile.close();

            ifstream infile_(label_path);
            if(infile_.is_open()){
                while(getline(infile_, line)){
                    istringstream row(line);
                    while(row.good()){
                        string label;
                        getline(row, label, ',');
                        this->rank.emplace_back(stoi(label));
                    }
                }
            } else {
                Log::Fatal("Cannot open (label) file %s", label_path);
            }

            ifstream infile__(query_path);
            if(infile__.is_open()){
                while(getline(infile__, line)){
                    istringstream row(line);
                    while(row.good()){
                        string qb;
                        getline(row, qb, ',');
                        this->query_boundaries.emplace_back(stoi(qb));
                    }
                }
            } else {
                Log::Fatal("Cannot open (query) file %s", query_path);
            }
            infile.close();
        }
        // initialize a sample-major `n x d` matrix
        void init_data(){
            for(int i=0; i<this->d; i++){
                Feature f = Feature(bin_cnt);
                for(int k=0; k<this->n; k++)
                    f.samples.emplace_back(make_pair(0.0, k));
                this->data.emplace_back(f);
            }
        }

        // returns dimensions of raw data
        pair<int, int> shape() const{
            return make_pair(this->n, this->d);
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
        vector<vector<pair<int, featval_t>>> raw_data;
        vector<vector<featval_t>> data;
    public:
        void load_dataset(const char* data_path, const char* query_path) {
            load_data_from_file(data_path, raw_data, this->rank, this->d);
            this->n = raw_data.size();
            init_data();

            load_query_from_file(query_path);

            for(int i=0; i<n; i++)
                data.emplace_back(this->d, 0);

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
