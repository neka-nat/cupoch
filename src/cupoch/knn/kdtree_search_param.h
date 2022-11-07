/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/
#pragma once

namespace cupoch {
namespace knn {

static const int NUM_MAX_NN = 100;

class KDTreeSearchParam {
public:
    enum class SearchType {
        Knn = 0,
        Radius = 1,
    };

public:
    virtual ~KDTreeSearchParam() {}

protected:
    KDTreeSearchParam(SearchType type) : search_type_(type) {}

public:
    SearchType GetSearchType() const { return search_type_; }

private:
    SearchType search_type_;
};

class KDTreeSearchParamKNN : public KDTreeSearchParam {
public:
    KDTreeSearchParamKNN(int knn = 30)
        : KDTreeSearchParam(SearchType::Knn), knn_(knn) {}

public:
    int knn_;
};

class KDTreeSearchParamRadius : public KDTreeSearchParam {
public:
    KDTreeSearchParamRadius(float radius, int max_nn)
        : KDTreeSearchParam(SearchType::Radius),
          radius_(radius),
          max_nn_(max_nn) {}

public:
    float radius_;
    int max_nn_;
};

}  // namespace knn
}  // namespace cupoch