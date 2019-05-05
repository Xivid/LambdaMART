#ifndef LAMBDAMART_HISTOGRAM_H
#define LAMBDAMART_HISTOGRAM_H

#include <lambdamart/dataset.h>
#include <mm_malloc.h>

#if defined(_MSC_VER)
#define ALIGNED(x) __declspec(align(x))
#else
#define ALIGNED(x) __attribute__ ((aligned(x)))
#endif

namespace LambdaMART {

    /*
	class FeatureColumn
	{
	public:
		feature_t fid = 0;

		std::vector<sample_t> indices;
		std::vector<bin_t> values;
		sample_t used = 0;  // number of non-default elements
		sample_t size = 0;  // total number of elements
		bin_t default_val = 0;  // most occurred value in this array
		std::vector<featval_t> splits; // thresholds

		void setDefault(bin_t def)
		{
			default_val = def;
		}

		void setSize(sample_t datasize)
		{
			size = datasize;
		}

		void NonDefaultResize(size_t size)
		{
			indices.resize(0);
			indices.reserve(size);
			values.resize(0);
			values.reserve(size);
			used = 0;
		}

		void push_back(sample_t index, bin_t value)
		{
			indices.push_back(index);
			values.push_back(value);
			++used;
		}

		void toArray(std::vector<bin_t> &vec)
		{
			std::fill(vec.begin(), vec.end(), default_val);
			for (size_t i = 0; i < used; ++i)
			{
				vec[indices[i]] = values[i];
			}
		}
	};
	*/

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
	struct ALIGNED(8) Bin
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
			//__m128 tmp = _mm_setzero_ps(), rhs = _mm_set_ps(0, 0, gradient, count);
			//tmp = _mm_add_ps(_mm_loadl_pi(tmp, (__m64 const *) this), rhs);
			//_mm_storel_pi((__m64 *) this, tmp);
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

	struct Histogram
	{
		bin_t bin_cnt;
		std::vector<Bin> bins;

		Histogram() = default;

		Histogram(bin_t numbins) : bin_cnt(numbins)
		{
			bins.resize(numbins);
		}

		Histogram(const Bin* hist, bin_t numbins) : bin_cnt(numbins)
		{
			bins.assign(hist, hist + numbins);
		}

		inline void update(bin_t bin, gradient_t count, gradient_t gradient)
		{
			bins[bin].update(count, gradient);
		}

		void cumulate()
		{
			if (bin_cnt <= 1)
			{
				return;
			}

			// cumulate from right to left
			for (bin_t bin = bin_cnt -2; bin >= 0; --bin)
			{
				bins[bin] += bins[bin+1];
			}
		}

		//TODO: defaultBin - part of optimization
		//void cumulate(const Bin& info, bin_t defaultBin)
		//{
		//	if (bin_cnt <= 1)
		//	{
		//		return;
		//	}

		//	// cumulate from right to left
		//	for (bin_t bin = bin_cnt - 2; bin > defaultBin; --bin)
		//	{
		//		bins[bin] += bins[bin + 1];
		//	}

		//	if (defaultBin != 0)
		//	{
		//		// put statistics of bin #1 ~ default in bin #0 ~ #(default - 1) temporarily
		//		// bins[0].val = _mm_sub_pd(info.val, bins[0].val);
		//		bins[0] ^= info;
		//		for (bin_t bin = 1; bin < defaultBin; ++bin)
		//		{
		//			bin_t binLeft = bin - 1;
		//			// bins[bin].val = _mm_sub_pd(bins[binLeft].val, bins[bin].val);
		//			bins[bin] ^= bins[binLeft];
		//		}

		//		// shift right to the correct place
		//		for (bin_t bin = defaultBin; bin > 0; --bin)
		//		{
		//			bin_t binLeft = bin - 1;
		//			bins[bin] = bins[binLeft];
		//		}
		//	}

		//	bins[0] = info;
		//}

		inline void GetFromDifference(const Histogram& parent, const Histogram& sibling)
		{
			const std::vector<Bin>& pbins = parent.bins;
			const std::vector<Bin>& sbins = sibling.bins;

			for (int i = 0; i < bin_cnt; ++i)
			{
				bins[i] = pbins[i] - sbins[i];
			}

			//unrolling
			//int remaining = bin_cnt % 4;
			//for (i = 0; i < bin_cnt - remaining; i += 4)
			//{
			//	bins[i] = pbins[i] - sbins[i];
			//	bins[i + 1] = pbins[i + 1] - sbins[i + 1];
			//	bins[i + 2] = pbins[i + 2] - sbins[i + 2];
			//	bins[i + 3] = pbins[i + 3] - sbins[i + 3];
			//}
			//for (; i < bin_cnt; ++i)
			//{
			//	bins[i] = pbins[i] - sbins[i];
			//}
		}

		SplitInfo BestSplit(const feature_t& fid,
						    Feature& feat,
						    const NodeStats* nodeInfo,
						    sample_t minInstancesPerNode = 1)
		{
			score_t totalGain = nodeInfo->getLeafSplitGain();
			NodeStats bestRightInfo;
			score_t bestShiftedGain = 0.0l;
			featval_t bestThreshold = 0.0l;
			bin_t bestThresholdBin = 0;
			size_t temp_threshold_size = feat.threshold.size();

			for (bin_t i = 1; i < temp_threshold_size; ++i)
			{
				bin_t threshLeft = i;
				NodeStats gt(bins[threshLeft]), lte(bins[0] - bins[threshLeft]);
				bin_t th = i - 1;

				if (lte.sum_count >= minInstancesPerNode && gt.sum_count >= minInstancesPerNode)
				{
					score_t currentShiftedGain = lte.getLeafSplitGain() + gt.getLeafSplitGain();

					if (currentShiftedGain > bestShiftedGain)
					{
						bestRightInfo = gt;
						bestShiftedGain = currentShiftedGain;
						bestThreshold = feat.threshold[th];
						bestThresholdBin = th;
					}
				}
			}

			Split* bestSplit = new Split(fid, bestThreshold);
			score_t splitGain = bestShiftedGain - totalGain;
			NodeStats bestLeftInfo(*nodeInfo - bestRightInfo);

			return SplitInfo(bestSplit, bestThresholdBin, splitGain, new NodeStats(*nodeInfo - bestRightInfo), new NodeStats(bestRightInfo));
		}
	};

	//TODO: HistogramCacheByNode - part of optimization
	/*!
	* HistogramCacheByNode (size of pool < max number of nodes in tree)
	*/
	/*struct HistogramCacheByNode
	{
		int used_slots;
		std::vector<Histogram> pool;
		std::vector<int> NodeIDToSlot, SlotToNodeID;

		HistogramCacheByNode() : used_slots(0)
		{
			// in default, max 1024 nodes (10 levels of depth), 64 bins per histogram
			NodeIDToSlot.clear();
			NodeIDToSlot.resize(1024, -1);
			SlotToNodeID.clear();
			SlotToNodeID.resize(1024, -1);
			pool.clear();
			pool.reserve(1024);
		}

		HistogramCacheByNode(int maxnodes) : used_slots(0)
		{
			NodeIDToSlot.clear();
			NodeIDToSlot.resize(maxnodes, -1);
			SlotToNodeID.clear();
			SlotToNodeID.resize(maxnodes, -1);
			pool.clear();
			pool.reserve(maxnodes);
		}

		void init(int maxnodes = 1024)
		{
			used_slots = 0;
			NodeIDToSlot.clear();
			NodeIDToSlot.resize(maxnodes, -1);
			SlotToNodeID.clear();
			SlotToNodeID.resize(maxnodes, -1);
			pool.clear();
			pool.reserve(maxnodes);
		}

		~HistogramCacheByNode()
		{
			NodeIDToSlot.resize(0);
			NodeIDToSlot.shrink_to_fit();
			SlotToNodeID.resize(0);
			SlotToNodeID.shrink_to_fit();
			pool.resize(0);
			pool.shrink_to_fit();
		}

		void put(nodeidx_t nodeID, const Histogram& hist)
		{
			NodeIDToSlot[nodeID] = used_slots;
			SlotToNodeID[used_slots] = nodeID;
			pool.push_back(hist);
			++used_slots;
		}

		void put(nodbin_cntnodeID, bin_t numBins, const Bin* hist)
		{
			NodeIDToSlot[nodeID] = used_slots;
			SlotToNodeID[used_slots] = nodeID;
			pool.push_back(Histogram(hist, numBins));
			++used_slots;
		}

		bool exist(nodeidx_t nodeID)
		{
			return (NodeIDToSlot[nodeID] >= 0);
		}

		const Histogram& get(nodeidx_t nodeID)
		{
			//ERROR_HANDLING_ASSERT_EX(NodeIDToSlot[nodeID] >= 0, "Accessing non-existent histogram pool item (node %u)!", nodeID);
			return pool[NodeIDToSlot[nodeID]];
		}

		void remove(nodeidx_t myID)
		{
			// put the last node's histogram into this node's slot
			int mySlot = NodeIDToSlot[myID];
			nodeidx_t nodeAtEnd = SlotToNodeID[used_slots - 1];

			NodeIDToSlot[nodeAtEnd] = mySlot;
			SlotToNodeID[mySlot] = nodeAtEnd;

			pool[mySlot] = pool.back();
			pool.pop_back();
			--used_slots;
		}
	}; */

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
            // TODO: use aligned malloc and free (different functions in macos/linux/windows)

			if (_data != nullptr)
				_mm_free(_data);
				//_aligned_free(_data);
			if (_head != nullptr)
				_mm_free(_head);
				//_aligned_free(_head);

			num_nodes = nodes;
			bin_cnt = bins;
            //_head = (Bin**)malloc(sizeof(Bin*) * nodes);
            //_data = (Bin*)malloc(sizeof(Bin) * nodes * bins);
            _head = (Bin**) _mm_malloc(sizeof(Bin*) * nodes, sizeof(Bin*));
            _data = (Bin*) _mm_malloc(sizeof(Bin) * nodes * bins, sizeof(Bin));
			//_head = (Bin**)_aligned_malloc(sizeof(Bin*) * nodes, sizeof(Bin*));
			//_data = (Bin*)_aligned_malloc(sizeof(Bin) * nodes * bins, 8);
			for (nodeidx_t i = 0; i < nodes; ++i)
			{
				_head[i] = _data + i * bins;
			}
		}

		~HistogramMatrix()
		{
            _mm_free(_data);
            _mm_free(_head);

            //free(_data);
            //free(_head);
			//_aligned_free(_data);
			//_aligned_free(_head);
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
		//TODO: defaultBin - part of optimization
		//inline void cumulate(nodeidx_t node, const NodeStats* info, bin_t defaultBin)
		//{
		//	// TODO: loop unrolling

		//	if (bin_cnt <= 1)
		//	{
		//		return;
		//	}

		//	Bin* bins = _head[node];
		//	const Bin total = *info;

		//	// cumulate from right to left
		//	for (bin_t bin = bin_cnt - 2; bin > defaultBin; --bin)
		//	{
		//		bin_t binRight = bin + 1;
		//		bins[bin] += bins[binRight];
		//	}

		//	if (defaultBin != 0)
		//	{
		//		// put statistics of bin #1 ~ default in bin #0 ~ #(default - 1) temporarily
		//		bins[0] ^= total;
		//		for (bin_t bin = 1; bin < defaultBin; ++bin)
		//		{
		//			bin_t binLeft = bin - 1;
		//			bins[bin] ^= bins[binLeft];
		//		}

		//		// shift right to the correct place
		//		for (bin_t bin = defaultBin; bin > 0; --bin)
		//		{
		//			bin_t binLeft = bin - 1;
		//			bins[bin] = bins[binLeft];
		//		}
		//	}

		//	bins[0] = total;
		//}

		inline void GetFromDifference(nodeidx_t node, const Histogram& parent_hist, Bin* sibling)
		{
			Bin* bins = _head[node];
			const std::vector<Bin>& pbins = parent_hist.bins;

			for (int i = 0; i < bin_cnt; ++i)
			{
				bins[i] = pbins[i] - sibling[i];
			}

			//TODO: unrolling
			//int remaining = bin_cnt % 4;
			//for (i = 0; i < bin_cnt - remaining; i += 4)
			//{
			//	bins[i] = pbins[i] - sibling[i];
			//	bins[i + 1] = pbins[i + 1] - sibling[i + 1];
			//	bins[i + 2] = pbins[i + 2] - sibling[i + 2];
			//	bins[i + 3] = pbins[i + 3] - sibling[i + 3];
			//}
			//for (; i < bin_cnt; ++i)
			//{
			//	bins[i] = pbins[i] - sibling[i];
			//}
		}

		inline void GetFromDifference(nodeidx_t node, bin_t bin_cnt, const Histogram& parent_hist, const Histogram& sibling_hist)
		{
			auto bins = _head[node];
			const std::vector<Bin>& pbins = parent_hist.bins;
			const std::vector<Bin>& sbins = sibling_hist.bins;

			int i;
			for (i = 0; i < bin_cnt; ++i)
			{
				bins[i] = pbins[i] - sbins[i];
			}

			//TODO: unrolling
			//int remaining = bin_cnt % 4;
			//for (i = 0; i < bin_cnt - remaining; i += 4)
			//{
			//	bins[i] = pbins[i] - sibling[i];
			//	bins[i + 1] = pbins[i + 1] - sibling[i + 1];
			//	bins[i + 2] = pbins[i + 2] - sibling[i + 2];
			//	bins[i + 3] = pbins[i + 3] - sibling[i + 3];
			//}
			//for (; i < bin_cnt; ++i)
			//{
			//	bins[i] = pbins[i] - sibling[i];
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

}

#endif //LAMBDAMART_HISTOGRAM_H
