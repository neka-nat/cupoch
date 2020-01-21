#pragma once

namespace cupoch {
namespace geometry {

static const int NUM_MAX_NN = 100;
typedef Eigen::Matrix<int, NUM_MAX_NN, 1, Eigen::DontAlign> KNNIndices;

class KDTreeSearchParam {
public:
    enum class SearchType {
        Knn = 0,
        Radius = 1,
        Hybrid = 2,
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
    KDTreeSearchParamRadius(float radius)
        : KDTreeSearchParam(SearchType::Radius), radius_(radius) {}

public:
    float radius_;
};

class KDTreeSearchParamHybrid : public KDTreeSearchParam {
public:
    KDTreeSearchParamHybrid(float radius, int max_nn)
        : KDTreeSearchParam(SearchType::Hybrid),
          radius_(radius),
          max_nn_(max_nn) {}

public:
    float radius_;
    int max_nn_;
};

}  // namespace geometry
}  // namespace cupoch