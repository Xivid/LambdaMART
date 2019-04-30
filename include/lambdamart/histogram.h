#ifndef LAMBDAMART_HISTOGRAM_H
#define LAMBDAMART_HISTOGRAM_H

namespace LambdaMART {

	// aligned on 8-byte boundary
	//typedef struct __declspec(align(8)) BinInfo
	typedef struct BinInfo
	{
	public:
		gradient_t sumCount, sumScores;

		BinInfo() { sumCount = sumScores = 0.0f; }

		BinInfo(gradient_t sumCounts, gradient_t sumTarget)
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

		inline BinInfo& operator+=(const BinInfo& rhs)
		{
			this->sumCount += rhs.sumCount;
			this->sumScores += rhs.sumScores;
			return *this;
		}

		inline void getComplement(const BinInfo& lhs, const BinInfo& rhs)
		{
			this->sumCount = lhs.sumCount - rhs.sumCount;
			this->sumScores = lhs.sumScores - rhs.sumScores;
		}

		inline BinInfo& operator^=(const BinInfo& rhs)
		{
			// get complement from rhs, not really arithmetic ^=
			this->sumCount = rhs.sumCount - this->sumCount;
			this->sumScores = rhs.sumScores - this->sumScores;
			return *this;
		}
	} NodeInfo;

	inline BinInfo operator-(BinInfo lhs, const BinInfo& rhs)
	{
		lhs.sumCount -= rhs.sumCount;
		lhs.sumScores -= rhs.sumScores;
		return lhs;
	}

	typedef std::tuple<SplitInfo, binidx_t, score_t, NodeInfoStats, NodeInfoStats> splitTup;

    class Histogram
    {
    };
}

#endif //LAMBDAMART_HISTOGRAM_H
