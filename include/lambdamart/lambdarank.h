#ifndef LAMBDAMART_LAMBDARANK_H
#define LAMBDAMART_LAMBDARANK_H
#include <vector>
#include <lambdamart/types.h>
#include <lambdamart/config.h>


namespace LambdaMART {

class LambdaRank {
    friend class Booster;

    public:
    explicit LambdaRank(const sample_t* query_boundaries, const sample_t num_queries, label_t* label, const Config& config) {
        boundaries_ = query_boundaries;
        num_queries_ = num_queries;
        label_ = label;
        kMaxPosition = config.max_position;
        set_eval_rank(&eval_ranks_);
        set_label_gain(config.max_label);
        set_discount();

        inverse_max_dcg_.resize(num_queries_);
        for (sample_t i = 0; i < num_queries_; ++i) {
            inverse_max_dcg_[i] = cal_maxdcg_k(config.max_position, label_ + boundaries_[i], boundaries_[i+1] - boundaries_[i]);
            if (inverse_max_dcg_[i] > 0.0) {
                inverse_max_dcg_[i] = 1.0f / inverse_max_dcg_[i];
            }
        }
        create_sigmoid_table();
    }

    // TODO change back to private
    //private:
        const sample_t* boundaries_;
        sample_t  num_queries_;
        label_t* label_;
        std::vector<double> inverse_max_dcg_;
        void get_derivatives(double* currentScores, double* gradients, double* hessians);
        void get_derivatives_one_query(const double* scores, double* gradients, 
                                        double* hessians, sample_t query_id);

        // NDCG related fields
        std::vector<double> label_gain_;
        std::vector<sample_t> eval_ranks_;
        std::vector<double> discount_;
        // max position of rank
        int kMaxPosition;

        std::vector<double> sigmoid_table_;
        double min_input_ = -50;
        double max_input_ = 50;
        uint32_t sigmoid_bins_ = 1024*1024;
        double sigmoid_ = 1.0;
        double sig_factor_;
            
        void set_eval_rank(std::vector<sample_t>* eval_ranks);
        void set_label_gain(int max_label); // uses the default of 2^i-1
        void set_discount();

        //void Init(const std::vector<double>& input_label_gain);

        // Calculates the DCG score at position k, given the rank label and score
        double cal_dcg_k(int k, const label_t* label, const double* score, sample_t num_data);

        // Calculates the DCG score at multiple locations
        // the result is stored in out. label and score are pointers to
        // labels and scores respectively
        void cal_dcg(const std::vector<int>& ks, const label_t* label, const double* score,
                                            sample_t num_data, std::vector<double>* out);

        // calculates the max score (ideal DCG) at position k
        // returns: max score
        double cal_maxdcg_k(int k,
            const label_t* label, sample_t num_data);

        // calculates the max DCG (ideal DCG), result is stored in out
        void cal_maxdcg(const std::vector<int>& ks,
            const label_t* label, sample_t num_data, std::vector<double>* out);

        // checks the label range
        void check_label(const label_t* label, sample_t num_data);

        // gets discount score at position k
        inline double get_discount(int k) { return discount_[k]; }

        double get_sigmoid(double score) const;

        void create_sigmoid_table();
    };



}

#endif //LAMBDAMART_LAMBDARANK_H
