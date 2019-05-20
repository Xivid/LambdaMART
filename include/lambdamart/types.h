#ifndef LAMBDAMART_TYPES_H
#define LAMBDAMART_TYPES_H

#include <cstdint>

namespace LambdaMART {
    typedef uint32_t sample_t;  // this is the former datasize_t, 32bit is enough
    typedef uint32_t feature_t;
    typedef uint16_t  bin_t;
    typedef uint32_t label_t;
    typedef uint32_t nodeidx_t;
    typedef float   score_t; // score of a node
    typedef float   gradient_t;
    typedef float   featval_t;
}

#endif //LAMBDAMART_TYPES_H
