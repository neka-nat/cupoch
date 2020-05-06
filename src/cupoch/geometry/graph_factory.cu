#include "cupoch/geometry/graph.h"
#include "cupoch/geometry/trianglemesh.h"

namespace cupoch {
namespace geometry {

std::shared_ptr<Graph> Graph::CreateFromTriangleMesh(const TriangleMesh &input) {
    auto out = std::make_shared<Graph>();
    out->points_ = input.vertices_;
    if (input.HasEdgeList()) {
        out->lines_ = input.edge_list_;
    } else {
        TriangleMesh tmp;
        tmp.triangles_ = input.triangles_;
        tmp.ComputeEdgeList();
        out->lines_ = tmp.edge_list_;
    }
    out->ConstructGraph();
    return out;
}

}
}