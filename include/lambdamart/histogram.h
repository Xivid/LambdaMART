#ifndef LAMBDAMART_HISTOGRAM_H
#define LAMBDAMART_HISTOGRAM_H

#include <lambdamart/dataset.h>
#include <immintrin.h>
#include <xmmintrin.h>
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
    struct ALIGNED(sizeof(gradient_t)*4) Bin
	//struct Bin
	{
		gradient_t sum_count, sum_gradients;

		Bin() { sum_count = sum_gradients = 0.0f; }

		Bin(gradient_t _sum_counts, gradient_t _sum_gradients)
		{
			sum_count = _sum_counts;
			sum_gradients = _sum_gradients;
		}

		inline void clear() { sum_count = sum_gradients = 0.0f; }

		//inline void update(gradient_t count, gradient_t gradient)
		//{
		//	//__m128 tmp = _mm_setzero_ps(), rhs = _mm_set_ps(0, 0, gradient, count);
		//	//tmp = _mm_add_ps(_mm_loadl_pi(tmp, (__m64 const *) this), rhs);
		//	//_mm_storel_pi((__m64 *) this, tmp);
		//	sum_count += count;
		//	sum_gradients += gradient;
		//}

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

	inline Bin operator+(Bin lhs, const Bin& rhs)
	{
		lhs.sum_count += rhs.sum_count;
		lhs.sum_gradients += rhs.sum_gradients;
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

	struct HistogramMatrix
	{
		nodeidx_t num_nodes;
		bin_t bin_cnt;  // unified max num of bins
		Bin** _head;
		Bin* _data;

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
                //free(_data);
                _mm_free(_data);
            //_aligned_free(_data);
            if (_head != nullptr)
                //free(_head);
                _mm_free(_head);

			num_nodes = nodes;
			bin_cnt = bins;
			//_head = (Bin**)malloc(sizeof(Bin*) * nodes);
			//_data = (Bin*)malloc(sizeof(Bin) * nodes * bins);
			_head = (Bin**)_mm_malloc(sizeof(Bin*) * nodes, sizeof(Bin*));
			_data = (Bin*)_mm_malloc(sizeof(Bin) * nodes * bins, sizeof(Bin));
			for (nodeidx_t i = 0; i < nodes; ++i)
			{
				_head[i] = _data + i * bins;
			}
		}

		~HistogramMatrix()
		{
			//free(_data);
			//free(_head);
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

        void cumulate(nodeidx_t num_candidates)
        {
            if (bin_cnt <= 1)
            {
                return;
            }

            const nodeidx_t node_unrolloing = 4;
            const nodeidx_t node_rest = num_candidates % node_unrolloing;

            nodeidx_t node = 0;
            for (;node < num_candidates - node_rest; node += node_unrolloing)
            {
                Bin* bins0 = _head[node];
                Bin* bins1 = _head[node+1];
                Bin* bins2 = _head[node+2];
                Bin* bins3 = _head[node+3];

                Bin bin0 = bins0[0];
                Bin bin1 = bins1[0];
                Bin bin2 = bins2[0];
                Bin bin3 = bins3[0];

                __m128 rhs0 = _mm_load_ps((gradient_t*) (bins0+bin_cnt-2));
                __m128 rhs1 = _mm_load_ps((gradient_t*) (bins1+bin_cnt-2));
                __m128 rhs2 = _mm_load_ps((gradient_t*) (bins2+bin_cnt-2));
                __m128 rhs3 = _mm_load_ps((gradient_t*) (bins3+bin_cnt-2));

                __m128 mhs0 = _mm_load_ps((gradient_t*) (bins0+bin_cnt-3));
                __m128 mhs1 = _mm_load_ps((gradient_t*) (bins1+bin_cnt-3));
                __m128 mhs2 = _mm_load_ps((gradient_t*) (bins2+bin_cnt-3));
                __m128 mhs3 = _mm_load_ps((gradient_t*) (bins3+bin_cnt-3));

                __m128 lhs0;
                __m128 lhs1;
                __m128 lhs2;
                __m128 lhs3;

                //bins0[bin_cnt-2] += bins0[bin_cnt-1];
                //bins1[bin_cnt-2] += bins1[bin_cnt-1];
                //bins2[bin_cnt-2] += bins2[bin_cnt-1];
                //bins3[bin_cnt-2] += bins3[bin_cnt-1];

                for (bin_t bin = bin_cnt-3; bin > 0; --bin)
                {
                    lhs0 = _mm_load_ps((gradient_t*) (bins0+bin-1));
                    lhs1 = _mm_load_ps((gradient_t*) (bins1+bin-1));
                    lhs2 = _mm_load_ps((gradient_t*) (bins2+bin-1));
                    lhs3 = _mm_load_ps((gradient_t*) (bins3+bin-1));

                    mhs0 = _mm_add_ps(mhs0, rhs0);
                    mhs1 = _mm_add_ps(mhs1, rhs1);
                    mhs2 = _mm_add_ps(mhs2, rhs2);
                    mhs3 = _mm_add_ps(mhs3, rhs3);

                    _mm_store_ps((gradient_t*) (bins0+bin), mhs0);
                    _mm_store_ps((gradient_t*) (bins1+bin), mhs1);
                    _mm_store_ps((gradient_t*) (bins2+bin), mhs2);
                    _mm_store_ps((gradient_t*) (bins3+bin), mhs3);

                    rhs0 = mhs0;
                    rhs1 = mhs1;
                    rhs2 = mhs2;
                    rhs3 = mhs3;

                    mhs0 = lhs0;
                    mhs1 = lhs1;
                    mhs2 = lhs2;
                    mhs3 = lhs3;

                    //bins0[bin] += bins0[bin+1];
                    //bins1[bin] += bins1[bin+1];
                    //bins2[bin] += bins2[bin+1];
                    //bins3[bin] += bins3[bin+1];
                }
                bins0[0] = bin0 + bins0[1];
                bins1[0] = bin1 + bins1[1];
                bins2[0] = bin2 + bins2[1];
                bins3[0] = bin3 + bins3[1];
            }
            for (; node < num_candidates; ++node)
            {
                Bin* bins = _head[node];
                Bin bin0 = bins[0];
                __m128 rhs = _mm_load_ps((gradient_t*) (bins+bin_cnt-2));
                __m128 mhs = _mm_load_ps((gradient_t*) (bins+bin_cnt-3));
                __m128 lhs;
                //bins[bin_cnt-2] += bins[bin_cnt-1];
                for (bin_t bin = bin_cnt-3; bin > 0; --bin)
                {
                    lhs = _mm_load_ps((gradient_t*) (bins+bin-1));
                    mhs = _mm_add_ps(mhs, rhs);
                    _mm_store_ps((gradient_t*) (bins+bin), mhs);
                    rhs = mhs;
                    mhs = lhs;
                    //bins[bin] += bins[bin+1];
                }
                bins[0] = bin0 + bins[1];
            }
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
			score_t splitGain = bestShiftedGain - totalGain;

            LOG_TRACE("bestRightInfo: %s", bestRightInfo.toString().c_str());
            LOG_TRACE("bestShiftedGain: %lf", bestShiftedGain);
            LOG_TRACE("bestThreshold: %lf", bestThreshold);
            LOG_TRACE("bestThresholdBin: %d", bestThresholdBin);
            LOG_TRACE("totalGain: %lf", totalGain);
            LOG_TRACE("splitGain: %lf", splitGain);

			return SplitInfo(bestSplit, bestThresholdBin, splitGain, new NodeStats(*nodeInfo - bestRightInfo), new NodeStats(bestRightInfo));
		}

	};

}

#endif //LAMBDAMART_HISTOGRAM_H
