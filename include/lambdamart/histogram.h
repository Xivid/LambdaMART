#ifndef LAMBDAMART_HISTOGRAM_H
#define LAMBDAMART_HISTOGRAM_H

#include <lambdamart/dataset.h>
#include <immintrin.h>
#include <mm_malloc.h>

#if defined(_MSC_VER)
#define ALIGNED(x) __declspec(align(x))
#else
#define ALIGNED(x) __attribute__ ((aligned(x)))
#endif

namespace LambdaMART {

struct Split
{
    feature_t feature;
    featval_t threshold;

    Split() {}
    Split(feature_t feature, featval_t threshold) : feature(feature), threshold(threshold) {}

    std::string toString(const std::string& prefix = "")
    {
        return prefix + "(feature = " + std::to_string(feature) + ", threshold = " + std::to_string(threshold) + ")";
    }
};

// aligned on 8-byte boundary
struct ALIGNED(sizeof(gradient_t)*2) Bin
{
    gradient_t sum_count, sum_gradients;

    Bin() { sum_count = sum_gradients = 0.0f; }

    Bin(gradient_t _sum_counts, gradient_t _sum_gradients)
    {
        sum_count = _sum_counts;
        sum_gradients = _sum_gradients;
    }

    inline void clear() { sum_count = sum_gradients = 0.0f; }

    inline void update(gradient_t count, gradient_t gradient)
    {
//		    __m256 lhs = _mm256_load_pd((double *) this);
//			__m256 rhs = _mm256_set_pd(0, 0, gradient, count);
//			lhs = _mm256_add_pd(lhs, rhs);
//			_mm256_store_pd((double *) this, lhs);
        sum_count += count;
        sum_gradients += gradient;
    }

    inline gradient_t getLeafSplitGain() const
    {
        return sum_gradients * sum_gradients / sum_count;
    }

    std::string toString()
    {
        return "(sum_count = " + std::to_string(sum_count) + ", sum_gradients = " + std::to_string(sum_gradients) + ")";
    }

    inline Bin& operator+=(const Bin& rhs)
    {
        this->sum_count += rhs.sum_count;
        this->sum_gradients += rhs.sum_gradients;
        return *this;
    }

    inline void getComplement(const Bin& lhs, const Bin& rhs)
    {
        this->sum_count = lhs.sum_count - rhs.sum_count;
        this->sum_gradients = lhs.sum_gradients - rhs.sum_gradients;
    }

    inline Bin& operator^=(const Bin& rhs)
    {
        // get complement from rhs, not really arithmetic ^=
        this->sum_count = rhs.sum_count - this->sum_count;
        this->sum_gradients = rhs.sum_gradients - this->sum_gradients;
        return *this;
    }
};

typedef Bin NodeStats;

inline Bin operator-(Bin lhs, const Bin& rhs)
{
    lhs.sum_count -= rhs.sum_count;
    lhs.sum_gradients -= rhs.sum_gradients;
    return lhs;
}

struct SplitInfo {
    Split* split;
    bin_t bin;
    score_t gain;
    NodeStats* left_stats;
    NodeStats* right_stats;
    gradient_t left_sum_squares = 0., left_sum_hessians = 0.,  // sum_squares is sum of **gradient squares**
               right_sum_squares = 0., right_sum_hessians = 0.;

    SplitInfo() = default;

    SplitInfo(Split* _s, bin_t _b, score_t _g, NodeStats* _l, NodeStats* _r)
        : split(_s), bin(_b), gain(_g), left_stats(_l), right_stats(_r),
          left_sum_squares(0.), left_sum_hessians(0.), right_sum_squares(0.), right_sum_hessians(0.) {}

    inline void update_children_stats(gradient_t ls, gradient_t lw, gradient_t rs, gradient_t rw) {
        left_sum_squares += ls;
        left_sum_hessians += lw;
        right_sum_squares += rs;
        right_sum_hessians += rw;
    }

    inline score_t get_left_impurity() {
        return (left_sum_squares - left_stats->sum_gradients * left_stats->sum_gradients / left_stats->sum_count) / left_stats->sum_count;
    }

    inline score_t get_right_impurity() {
        return (right_sum_squares - right_stats->sum_gradients * right_stats->sum_gradients / right_stats->sum_count) / right_stats->sum_count;
    }

    inline score_t get_left_output() {
        return calc_leaf_output(left_stats->sum_count, left_stats->sum_gradients, left_sum_hessians);
    }

    inline score_t get_right_output() {
        return calc_leaf_output(right_stats->sum_count, right_stats->sum_gradients, right_sum_hessians);
    }

    inline string toString() {
        return "(split: " + (split == nullptr ? "null" : split->toString()) + ", bin: " + to_string(bin) + ", gain: " + to_string(gain)
                + ", left_stats: " + (left_stats == nullptr ? "null" : left_stats->toString())
                + ", right_stats: " + (right_stats == nullptr ? "null" : right_stats->toString()) + ")";
    }

    inline bool operator >=(const SplitInfo& other) {
        return gain >= other.gain;
    }

    inline score_t calc_leaf_output(gradient_t totalCount, gradient_t sumGradients, gradient_t sumHessians)
    {
        const score_t epsilon = 1.1e-38, maxOutput = 100;
        gradient_t leafValue = (sumGradients / totalCount + epsilon) / (2 * sumHessians / totalCount + epsilon);
        return (leafValue > maxOutput) ? maxOutput : ((leafValue < -maxOutput) ? -maxOutput : leafValue);
    }
};

class HistogramMatrix
{
private:
    nodeidx_t num_nodes;
    bin_t bin_cnt;  // unified max num of bins
    Bin** _head;
    Bin* _data;

public:
    HistogramMatrix() : num_nodes(0), bin_cnt(0), _head(nullptr), _data(nullptr) {}

    HistogramMatrix(nodeidx_t nodes, bin_t bins)
    {
        init(nodes, bins);
    }

    HistogramMatrix(HistogramMatrix const &) = delete;
    HistogramMatrix& operator=(HistogramMatrix const &) = delete;

    void init(nodeidx_t nodes, bin_t bins)
    {
        if (_data != nullptr)
            _mm_free(_data);
        if (_head != nullptr)
            _mm_free(_head);

        num_nodes = nodes;
        bin_cnt = bins;
        _head = (Bin**) _mm_malloc(sizeof(Bin*) * nodes, sizeof(Bin*));
        _data = (Bin*) _mm_malloc(sizeof(Bin) * nodes * bins, sizeof(Bin));
        for (nodeidx_t i = 0; i < nodes; ++i)
        {
            _head[i] = _data + i * bins;
        }
    }

    ~HistogramMatrix()
    {
        _mm_free(_data);
        _mm_free(_head);
    }


    inline void clear(nodeidx_t nodes)
    {
        memset(_data, 0, sizeof(Bin) * nodes * bin_cnt);
    }

    inline void clear()
    {
        clear(num_nodes);
    }

    inline Bin* operator[](nodeidx_t node)
    {
        return _head[node];
    }

    inline Bin* data()
    {
        return _data;
    }

    void cumulate(nodeidx_t node)
    {
        if (bin_cnt <= 1)
        {
            return;
        }

        Bin* bins = _head[node];

        for (bin_t bin = bin_cnt-2; bin > 0; --bin)
        {
            bins[bin] += bins[bin+1];
        }
        bins[0] += bins[1];

        //LOG_TRACE(" Bin # ");
        //for (bin_t bin = 0; bin < bin_cnt; bin++)
        //{
        //    LOG_TRACE("\t%i\t%s", bin, bins[bin].toString().c_str());
        //}
    }

    SplitInfo get_best_split(nodeidx_t node, feature_t fid,
                             const Feature &feat,
                             const NodeStats *nodeInfo,
                             const sample_t minInstancesPerNode = 1)
    {
        const Bin* bins = _head[node];
        const vector<featval_t>& temp_threshold = feat.threshold;

        score_t totalGain = nodeInfo->getLeafSplitGain();

        NodeStats bestRightInfo;
        score_t bestShiftedGain = 0.0;
        featval_t bestThreshold = 0.0;
        bin_t bestThresholdBin = 0;
        size_t temp_threshold_size = temp_threshold.size();

        LOG_TRACE(" Threshold #:");

        for (bin_t i = 1; i < feat.bin_count(); ++i)
        {
            bin_t threshLeft = i;
            NodeStats gt(bins[threshLeft]), lte(bins[0] - bins[threshLeft]);
            bin_t threshBin = i - 1;
            LOG_TRACE("\t%d\tlte: %s\n\t\t\t\tgt: %s\tGain: %lf", i, lte.toString().c_str(), gt.toString().c_str(), lte.getLeafSplitGain() + gt.getLeafSplitGain());
            if (lte.sum_count >= minInstancesPerNode && gt.sum_count >= minInstancesPerNode)
            {
                score_t currentShiftedGain = lte.getLeafSplitGain() + gt.getLeafSplitGain();
                if (currentShiftedGain > bestShiftedGain)
                {
                    bestRightInfo = gt;
                    bestShiftedGain = currentShiftedGain;
                    LOG_TRACE("\tbestShiftGain updated to: %lf", bestShiftedGain);
                    bestThreshold = temp_threshold[threshBin];
                    bestThresholdBin = threshBin;
                }
            }
        }

        auto* bestSplit = new Split(fid, bestThreshold);
        double splitGain = bestShiftedGain - totalGain;

        LOG_TRACE("bestRightInfo: %s", bestRightInfo.toString().c_str());
        LOG_TRACE("bestShiftedGain: %lf", bestShiftedGain);
        LOG_TRACE("bestThreshold: %lf", bestThreshold);
        LOG_TRACE("bestThresholdBin: %d", bestThresholdBin);
        LOG_TRACE("totalGain: %lf", totalGain);
        LOG_TRACE("splitGain: %lf", splitGain);

        return SplitInfo(bestSplit, bestThresholdBin, splitGain, new NodeStats(*nodeInfo - bestRightInfo), new NodeStats(bestRightInfo));
    }

};

class HistogramMatrixTrans
{
private:
    nodeidx_t num_nodes;
    bin_t bin_cnt;  // unified max num of bins
    Bin** _head;
    Bin* _data;

public:
    HistogramMatrixTrans() : num_nodes(0), bin_cnt(0), _head(nullptr), _data(nullptr) {}

    HistogramMatrixTrans(nodeidx_t nodes, bin_t bins)
    {
        init(nodes, bins);
    }

    HistogramMatrixTrans(HistogramMatrixTrans const &) = delete;
    HistogramMatrixTrans& operator=(HistogramMatrixTrans const &) = delete;

    void init(nodeidx_t nodes, bin_t bins)
    {
        if (_data != nullptr)
            _mm_free(_data);
        if (_head != nullptr)
            _mm_free(_head);

        num_nodes = nodes;
        bin_cnt = bins;

        _head = (Bin**) _mm_malloc(sizeof(Bin*) * bins, sizeof(Bin*));
        _data = (Bin*) _mm_malloc(sizeof(Bin) * nodes * bins, sizeof(Bin));
        for (bin_t i = 0; i < bins; ++i)
        {
            _head[i] = _data + i * nodes;
        }
    }

    ~HistogramMatrixTrans()
    {
        _mm_free(_data);
        _mm_free(_head);
    }


    inline void clear(nodeidx_t nodes)
    {
        for(bin_t i = 0; i < bin_cnt; ++i)
        {
            memset(_head[i], 0, sizeof(Bin) * nodes);
        }
        //TODO try simply clear()
    }

    inline void clear()
    {
        memset(_data, 0, sizeof(Bin) * num_nodes * bin_cnt);
    }

    inline Bin* operator[](bin_t bin)
    {
        return _head[bin];
    }

    inline Bin* data()
    {
        return _data;
    }


    void cumulate_noavx(nodeidx_t num_candidates) {
        Bin* bins_high = _head[bin_cnt-1];
        for (bin_t bin = bin_cnt - 1; bin > 0; --bin)
        {
            Bin* bins_low = _head[bin - 1];

            for (nodeidx_t i = 0; i < num_candidates; ++i)
            {
                bins_low[i] += bins_high[i];
            }

            bins_high = bins_low;
        }
    }

    void cumulate(nodeidx_t num_candidates)
    {
        if (bin_cnt <= 1)
        {
            return;
        }

        const nodeidx_t simd_blocking = 4;
        const nodeidx_t bins_per_register = 32 / sizeof(Bin);
        const nodeidx_t overall_blocking = simd_blocking * bins_per_register;
        const nodeidx_t node_rest = num_candidates % overall_blocking;

        nodeidx_t node;
        for (node = 0; node < num_candidates - node_rest; node += overall_blocking)
        {
            Bin* bins_high = _head[bin_cnt - 1] + node;

            // get last bin
            for (bin_t bin = bin_cnt - 1; bin > 0; --bin)
            {
                Bin* bins_low = _head[bin - 1] + node;

                // do `simd_blocking` simds, each simd adds `bins_per_register` bins to the lower row
                Bin* doubles_high = bins_high;
                Bin* doubles_low = bins_low;
                for (int i = 0; i < simd_blocking; i++) {
                    __m256d high = _mm256_load_pd((gradient_t*) doubles_high);
                    __m256d low = _mm256_load_pd((gradient_t*) doubles_low);
                    _mm256_store_pd((gradient_t*) doubles_low, _mm256_add_pd(low, high));

                    doubles_high += bins_per_register;
                    doubles_low += bins_per_register;
                }

                bins_high = bins_low;
            }
        }

        if (node_rest > 0) {
            const nodeidx_t node_rest = num_candidates % bins_per_register;
            for (; node < num_candidates - node_rest; node += bins_per_register)
            {
                // get last bin
                Bin* bins_high = _head[bin_cnt - 1] + node;

                for (bin_t bin = bin_cnt - 1; bin > 0; --bin)
                {
                    Bin* bins_low = _head[bin - 1] + node;

                    __m256d high = _mm256_load_pd((gradient_t*) bins_high);
                    __m256d low = _mm256_load_pd((gradient_t*) bins_low);
                    _mm256_store_pd((gradient_t*) bins_low, _mm256_add_pd(low, high));

                    bins_high = bins_low;
                }
            }
            if (node_rest > 0) {
                Bin* bins_high = _head[bin_cnt-1] + node;
                for (bin_t bin = bin_cnt - 1; bin > 0; --bin)
                {
                    Bin* bins_low = _head[bin - 1] + node;

                    for (nodeidx_t i = 0; i < node_rest; ++i)
                    {
                        bins_low[i] += bins_high[i];
                    }

                    bins_high = bins_low;
                }
            }
        }

    }

    // TODO: avx
    void get_best_splits(nodeidx_t num_candidates,
                         feature_t fid,
                         const int num_feature_blocking,
                         const vector<Feature>& feats,
                         vector<NodeStats*>& nodeInfo,
                         vector<SplitInfo>& best_splits,
                         const sample_t minInstancesPerNode = 1)
    {
        LOG_TRACE("HistogramMatrixTrans transpose get_best_splits...");

        // Warning: every feature can have different number of bins! Here we are taking the maximum.
        for (nodeidx_t node = 0; node < num_candidates; ++node)
        {
            size_t node_offset = node * num_feature_blocking;

            gradient_t totalGain = nodeInfo[node]->getLeafSplitGain();
            Bin* bin0_base = _data + node_offset;

            NodeStats bestRightInfo;
            score_t bestShiftedGain = 0.0;
            featval_t bestThreshold = 0.0;
            bin_t bestThresholdBin = 0;
            int best_fid = 0;

            for (bin_t bin = 1; bin < bin_cnt; bin++) {
                Bin* bin_base = _head[bin] + node * num_feature_blocking;

                for (int i = 0; i < num_feature_blocking; i++) {
                    NodeStats gt(bin_base[i]), lte(bin0_base[i] - bin_base[i]);
                    if (lte.sum_count >= minInstancesPerNode && gt.sum_count >= minInstancesPerNode) {
                        score_t currentShiftedGain = lte.getLeafSplitGain() + gt.getLeafSplitGain();
                        if (currentShiftedGain > bestShiftedGain)
                        {
                            bestRightInfo = gt;
                            bestShiftedGain = currentShiftedGain;
                            best_fid = fid + i;
                            bestThreshold = feats[best_fid].threshold[bin - 1];
                            bestThresholdBin = bin - 1;
                        }
                    }
                }
            }

            double splitGain = bestShiftedGain - totalGain;
            if (splitGain >= best_splits[node].gain) {
                best_splits[node] = SplitInfo(
                        new Split(best_fid, bestThreshold),
                        bestThresholdBin,
                        splitGain,
                        new NodeStats(*nodeInfo[node] - bestRightInfo),
                        new NodeStats(bestRightInfo));
            }
        }

    }

};

}

#endif //LAMBDAMART_HISTOGRAM_H
