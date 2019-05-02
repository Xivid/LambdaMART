#ifndef LAMBDAMART_HISTOGRAM_H
#define LAMBDAMART_HISTOGRAM_H

#include <lambdamart/dataset.h>

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
	//typedef struct __declspec(align(8)) Bin
	struct Bin
	{
		gradient_t sumCount, sumScores;

		Bin() { sumCount = sumScores = 0.0f; }

		Bin(gradient_t sumCounts, gradient_t sumTarget)
		{
			sumCount = sumCounts;
			sumScores = sumTarget;
		}

		inline void clear() { sumCount = sumScores = 0.0f; }

		inline void update(gradient_t count, gradient_t score)
		{
			//__m128 tmp = _mm_setzero_ps(), rhs = _mm_set_ps(0, 0, score, count);
			//tmp = _mm_add_ps(_mm_loadl_pi(tmp, (__m64 const *) this), rhs);
			//_mm_storel_pi((__m64 *) this, tmp);
			sumCount += count;
			sumScores += score;
		}

		inline gradient_t getLeafSplitGain() const
		{
			return sumScores * sumScores / sumCount;
		}

		std::string toString()
		{
			return "(sumCount = " + std::to_string(sumCount) + ", sumScores = " + std::to_string(sumScores) + ")";
		}

		inline Bin& operator+=(const Bin& rhs)
		{
			this->sumCount += rhs.sumCount;
			this->sumScores += rhs.sumScores;
			return *this;
		}

		inline void getComplement(const Bin& lhs, const Bin& rhs)
		{
			this->sumCount = lhs.sumCount - rhs.sumCount;
			this->sumScores = lhs.sumScores - rhs.sumScores;
		}

		inline Bin& operator^=(const Bin& rhs)
		{
			// get complement from rhs, not really arithmetic ^=
			this->sumCount = rhs.sumCount - this->sumCount;
			this->sumScores = rhs.sumScores - this->sumScores;
			return *this;
		}
	};

	typedef Bin NodeStats;

	inline Bin operator-(Bin lhs, const Bin& rhs)
	{
		lhs.sumCount -= rhs.sumCount;
		lhs.sumScores -= rhs.sumScores;
		return lhs;
	}

	struct SplitInfo {
		Split split;
		bin_t bin;
		score_t gain;
		NodeStats left_stats;
		NodeStats right_stats;

		SplitInfo() = default;
		SplitInfo(Split& _s, bin_t _b, score_t _g, NodeStats& _l, NodeStats& _r)
			: split(_s), bin(_b), gain(_g), left_stats(_l), right_stats(_r) {}

		string toString() {
			return "(split: " + split.toString() + ", bin: " + to_string(bin) + ", gain: " + to_string(gain)
					+ ", left_stats: " + left_stats.toString() + ", right_stats: " + right_stats.toString() + ")";
		}

		bool operator >=(const SplitInfo& other) {
			return gain >= other.gain;
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

		inline void update(bin_t bin, gradient_t sampleWeight, gradient_t score)
		{
			bins[bin].update(sampleWeight, score);
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
						    feature& feat,
						    const NodeStats* nodeInfo,
						    sample_t minInstancesPerNode = 1)
		{
			//DEBUG_ASSERT_EX(feat.splits.size() > 0, "empty splits!");
			// DLogTrace("[thread %u] binsToBestSplit: nodeInfo = %s", thread_get_id(), nodeInfo.toString().c_str());

			//feature_t feature = feat.fid;
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

				if (lte.sumCount >= minInstancesPerNode && gt.sumCount >= minInstancesPerNode)
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

			Split bestSplit(fid, bestThreshold);
			score_t splitGain = bestShiftedGain - totalGain;
			NodeStats bestLeftInfo(*nodeInfo - bestRightInfo);

			return SplitInfo(bestSplit, bestThresholdBin, splitGain, bestLeftInfo, bestRightInfo);
		}
	};

	//TODO: HistogramCacheByNode - part of optimization
	/*!
	* HistogramCacheByNode (size of pool < max number of nodes in tree)
	*/
	//struct HistogramCacheByNode
	//{
	//	int used_slots;
	//	std::vector<Histogram> pool;
	//	std::vector<int> NodeIDToSlot, SlotToNodeID;

	//	HistogramCacheByNode() : used_slots(0)
	//	{
	//		// in default, max 1024 nodes (10 levels of depth), 64 bins per histogram
	//		NodeIDToSlot.clear();
	//		NodeIDToSlot.resize(1024, -1);
	//		SlotToNodeID.clear();
	//		SlotToNodeID.resize(1024, -1);
	//		pool.clear();
	//		pool.reserve(1024);
	//	}

	//	HistogramCacheByNode(int maxnodes) : used_slots(0)
	//	{
	//		NodeIDToSlot.clear();
	//		NodeIDToSlot.resize(maxnodes, -1);
	//		SlotToNodeID.clear();
	//		SlotToNodeID.resize(maxnodes, -1);
	//		pool.clear();
	//		pool.reserve(maxnodes);
	//	}

	//	void init(int maxnodes = 1024)
	//	{
	//		used_slots = 0;
	//		NodeIDToSlot.clear();
	//		NodeIDToSlot.resize(maxnodes, -1);
	//		SlotToNodeID.clear();
	//		SlotToNodeID.resize(maxnodes, -1);
	//		pool.clear();
	//		pool.reserve(maxnodes);
	//	}

	//	~HistogramCacheByNode()
	//	{
	//		NodeIDToSlot.resize(0);
	//		NodeIDToSlot.shrink_to_fit();
	//		SlotToNodeID.resize(0);
	//		SlotToNodeID.shrink_to_fit();
	//		pool.resize(0);
	//		pool.shrink_to_fit();
	//	}

	//	void put(nodeidx_t nodeID, const Histogram& hist)
	//	{
	//		NodeIDToSlot[nodeID] = used_slots;
	//		SlotToNodeID[used_slots] = nodeID;
	//		pool.push_back(hist);
	//		++used_slots;
	//	}

	//	void put(nodbin_cntnodeID, bin_t numBins, const Bin* hist)
	//	{
	//		NodeIDToSlot[nodeID] = used_slots;
	//		SlotToNodeID[used_slots] = nodeID;
	//		pool.push_back(Histogram(hist, numBins));
	//		++used_slots;
	//	}

	//	bool exist(nodeidx_t nodeID)
	//	{
	//		return (NodeIDToSlot[nodeID] >= 0);
	//	}

	//	const Histogram& get(nodeidx_t nodeID)
	//	{
	//		//ERROR_HANDLING_ASSERT_EX(NodeIDToSlot[nodeID] >= 0, "Accessing non-existent histogram pool item (node %u)!", nodeID);
	//		return pool[NodeIDToSlot[nodeID]];
	//	}

	//	void remove(nodeidx_t myID)
	//	{
	//		// put the last node's histogram into this node's slot
	//		int mySlot = NodeIDToSlot[myID];
	//		nodeidx_t nodeAtEnd = SlotToNodeID[used_slots - 1];

	//		NodeIDToSlot[nodeAtEnd] = mySlot;
	//		SlotToNodeID[mySlot] = nodeAtEnd;

	//		pool[mySlot] = pool.back();
	//		pool.pop_back();
	//		--used_slots;
	//	}
	//};

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
				free(_data);
				//_aligned_free(_data);
			if (_head != nullptr)
				free(_head);
				//_aligned_free(_head);
			
			num_nodes = nodes;
			bin_cnt = bins;
			_head = (Bin**)malloc(sizeof(Bin*) * nodes);
			_data = (Bin*)malloc(sizeof(Bin) * nodes * bins);
			//_head = (Bin**)_aligned_malloc(sizeof(Bin*) * nodes, sizeof(Bin*));
			//_data = (Bin*)_aligned_malloc(sizeof(Bin) * nodes * bins, 8);
			for (nodeidx_t i = 0; i < nodes; ++i)
			{
				_head[i] = _data + i * bins;
			}
		}

		~HistogramMatrix()
		{
			free(_data);
			free(_head);
			//_aligned_free(_data);
			//_aligned_free(_head);
		}

		inline void clear()
		{
			for (size_t i = 0; i < (size_t)num_nodes * bin_cnt; ++i)
			{
				_data[i].clear();
			}
		}

		inline void clear(nodeidx_t nodes)
		{
			for (size_t i = 0; i < (size_t)nodes * bin_cnt; ++i)
			{
				_data[i].clear();
			}
		}

		inline Bin* operator[](nodeidx_t node)
		{
			return _head[node];
		}

		inline Bin* data()
		{
			return _data;
		}

		inline bin_t numBins()
		{
			return bin_cnt;
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

		SplitInfo BestSplit(nodeidx_t node, feature_t fid,
						   const feature& feat,
						   const NodeStats* nodeInfo,
						   const sample_t minInstancesPerNode = 1)
		{
			//DEBUG_ASSERT_EX(feat.splits.size() > 0, "empty splits!");
			// DLogTrace("[thread %u] binsToBestSplit: nodeInfo = %s", thread_get_id(), nodeInfo.toString().c_str());

			const Bin* bins = _head[node];
			const vector<featval_t>& temp_threshold = feat.threshold;

			score_t totalGain = nodeInfo->getLeafSplitGain();
			NodeStats bestRightInfo;
			score_t bestShiftedGain = 0.0l;
			featval_t bestThreshold = 0.0l;
			bin_t bestThresholdBin = 0;
			size_t temp_threshold_size = temp_threshold.size();

			for (bin_t i = 1; i < temp_threshold_size; ++i)
			{
				bin_t threshLeft = i;
				NodeStats gt(bins[threshLeft]), lte(bins[0] - bins[threshLeft]);
				bin_t th = i - 1;

				if (lte.sumCount >= minInstancesPerNode && gt.sumCount >= minInstancesPerNode)
				{
					score_t currentShiftedGain = lte.getLeafSplitGain() + gt.getLeafSplitGain();

					if (currentShiftedGain > bestShiftedGain)
					{
						bestRightInfo = gt;
						bestShiftedGain = currentShiftedGain;
						bestThreshold = temp_threshold[th];
						bestThresholdBin = th;
					}
				}
			}

			Split bestSplit(fid, bestThreshold);
			double splitGain = bestShiftedGain - totalGain;
			NodeStats bestLeftInfo(*nodeInfo - bestRightInfo);

			return SplitInfo(bestSplit, bestThresholdBin, splitGain, bestLeftInfo, bestRightInfo);
		}

	};

}

#endif //LAMBDAMART_HISTOGRAM_H
