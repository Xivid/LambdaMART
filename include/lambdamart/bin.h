#ifndef LAMBDAMART_BIN_H
#define LAMBDAMART_BIN_H

#include <vector>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>

using namespace std;

namespace lambdamart {

    class Dataset;

    class Binner {
//        friend class Dataset;

        std::vector<double> thresholds;
        static const uint8_t bin_count = 64;

        virtual Dataset* load(const char* path) = 0;

    public:

        static void _load_data_from_file(const char* path, int d, vector<vector<double>>& data){
            ifstream infile(path);
            string line;
            if(infile.is_open()){
                vector<double> record;
                record.reserve(d);
                while(getline(infile, line)){
                    istringstream row(line);
                    vector<string> tokens{istream_iterator<string>{row}, istream_iterator<string>{}};
                    for(auto & token: tokens){
                        int delimiter = token.find(':');
                        int index = stoi(token.substr(0, delimiter));
                        double val = stof(token.substr(delimiter+1, token.length()));
                        record[index] = val;
                    }
                    data.push_back(record);
                }
            }
        }

        vector<vector<uint8_t>> load(const char* path, int d){
            vector<vector<double>> raw_data;
            const int n = raw_data.size();
            const int bin_size = (int)(n/bin_count);

            _load_data_from_file(path, d, raw_data);
            vector<vector<uint8_t>> processed_data(n);

            for(int i=0; i<d; i++){
                vector<pair<double, int>> feature(n);
                for(int row=0; row<n; row++)
                    feature[row] = make_pair(raw_data[row][i], row);
                sort(feature.begin(), feature.end());

                int curr_count = 0, bin_index = 0;
                for(int entry=0; entry<n; entry++){
                    pair<double, int> val = feature[entry];
                    processed_data[val.second].push_back(bin_index);
                    if(++curr_count > bin_size) {
                        bin_index++;
                        curr_count = 0;
                    }
                }
            }
            return processed_data;
        }

    };

    class PercentileBinner : Binner {
        friend class Dataset;

        Dataset* load(const char* path) override;
    };

}
#endif //LAMBDAMART_BIN_H
