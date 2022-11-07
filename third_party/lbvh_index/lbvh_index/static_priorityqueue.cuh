#pragma once
#include <cuda/std/cmath>
#include <cuda/std/limits>

#define _min(x, y) (x < y ? x : y)
#define _max(x, y) (x < y ? y : x)
#define CAS(x, y) { auto tmp = _min(x,y); x = _max(x,y); y = tmp; }
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

namespace lbvh {


    template<typename KEY>
    struct KeyType {
        unsigned int id;
        KEY key;

        __device__ KeyType(KEY max_value)
            : id(UINT_MAX), key(max_value) {}

        __device__ KeyType()
            : KeyType(FLT_MAX) {}

        __device__ bool operator <(const KeyType &b) const {
            return key < b.key;
        }
    };

    template<typename KEY, unsigned int SIZE>
    struct StaticPriorityQueue {
        unsigned int _size;

        KeyType<KEY> _k[SIZE];

        __device__ StaticPriorityQueue(KEY max_value)
            : _size(0) {
            KeyType<KEY> kt(max_value);
            #pragma unroll (SIZE)
            for(int i=0; i<SIZE; ++i) {
                _k[i] = kt;
            }
        }

        __device__ StaticPriorityQueue()
            : StaticPriorityQueue(FLT_MAX) {};

        __device__ KEY max_distance() const{
            return _k[0].key;
        }

        __device__ unsigned int size() const{
            return _size;
        }

        __device__ unsigned int max_size() const {
            return SIZE;
        }

        __device__ void operator()(const float3& point, unsigned int index, const KEY& key) {
            KeyType<KEY> k;
            k.key = key;
            k.id = index;
            _k[0] = k;
            _size = min(++_size, SIZE);

            sort();
        }
        __device__ const KeyType<KEY>& operator [](unsigned int index) const {
            return _k[(SIZE-index)-1];
        }

        __device__ void write_results(unsigned int* indices, KEY* distances, unsigned int* n_neighbors) {
            for(int i=0; i<_size; ++i) {
                auto k = operator[](i);
                indices[i] = k.id;
                distances[i] = k.key;
            }
            n_neighbors[0] = _size;
        }

        // sort both arrays
        __device__ void sort() {
            #pragma unroll
            for (int i = 0; i < (SIZE - 1); ++i) {
                CAS(_k[i], _k[i+1]);
            }
        }
    };
}


#undef CAS
#undef min
#undef max
