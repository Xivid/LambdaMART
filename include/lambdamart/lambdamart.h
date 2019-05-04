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

namespace LambdaMART {

    const char* version()
    {
        return "LambdaMART 0.1";
    }

    const char* help() {
        return "usage: lambdamart config_file";
    }
}

#endif //LAMBDAMART_LAMBDAMART_H
