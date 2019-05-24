//
// Created by Zhifei Yang on 25.03.19.
//

#ifndef LAMBDAMART_LAMBDAMART_H
#define LAMBDAMART_LAMBDAMART_H
// #define DEBUG_OUTPUT

#include <lambdamart/config.h>
#include <lambdamart/common.h>
#include <lambdamart/dataset.h>
#include <lambdamart/booster.h>
#include <lambdamart/model.h>
#include <lambdamart/log.h>
#include <lambdamart/perf.h>

namespace LambdaMART {

    const char* version()
    {
        return "LambdaMART 0.1";
    }

    const char* help() {
        return "usage: lambdamart config_file";
    }

    inline void* aligned_malloc(size_t size, size_t align) {
        void *result;
#ifdef _MSC_VER
        result = _aligned_malloc(size, align);
#else
        if(posix_memalign(&result, align, size)) result = 0;
#endif
        return result;
    }

    inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        free(ptr);
#endif

    }
}

#endif //LAMBDAMART_LAMBDAMART_H
