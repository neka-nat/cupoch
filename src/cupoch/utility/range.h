#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace thrust {

// examples:
//   repeated_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   repeated_range([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   repeated_range([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
//   ...

template <typename Iterator>
class repeated_range {
public:
    typedef typename thrust::iterator_difference<Iterator>::type
            difference_type;

    struct repeat_functor
        : public thrust::unary_function<difference_type, difference_type> {
        difference_type repeats;

        repeat_functor(difference_type repeats) : repeats(repeats) {}

        __host__ __device__ difference_type
        operator()(const difference_type &i) const {
            return i / repeats;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>
            CountingIterator;
    typedef typename thrust::transform_iterator<repeat_functor,
                                                CountingIterator>
            TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
            PermutationIterator;

    // type of the repeated_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    repeated_range(Iterator first, Iterator last, difference_type repeats)
        : first(first), last(last), repeats(repeats) {}

    iterator begin(void) const {
        return PermutationIterator(first,
                                   TransformIterator(CountingIterator(0),
                                                     repeat_functor(repeats)));
    }

    iterator end(void) const { return begin() + repeats * (last - first); }

protected:
    Iterator first;
    Iterator last;
    difference_type repeats;
};

// this example illustrates how to make strided access to a range of values
// examples:
//   strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6]
//   strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
//   strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
//   ...

template <typename Iterator>
class strided_range {
public:
    typedef typename thrust::iterator_difference<Iterator>::type
            difference_type;

    struct stride_functor
        : public thrust::unary_function<difference_type, difference_type> {
        difference_type stride;

        stride_functor(difference_type stride) : stride(stride) {}

        __host__ __device__ difference_type
        operator()(const difference_type &i) const {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>
            CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor,
                                                CountingIterator>
            TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
            PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const {
        return PermutationIterator(
                first,
                TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

// examples:
//   tiled_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 2) -> [0, 1, 2, 3, 0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 3) -> [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
//   ...

template <typename Iterator>
class tiled_range {
public:
    typedef typename thrust::iterator_difference<Iterator>::type
            difference_type;

    struct tile_functor
        : public thrust::unary_function<difference_type, difference_type> {
        difference_type tiles;

        tile_functor(difference_type tiles) : tiles(tiles) {}

        __host__ __device__ difference_type
        operator()(const difference_type &i) const {
            return i % tiles;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>
            CountingIterator;
    typedef typename thrust::transform_iterator<tile_functor,
                                                CountingIterator>
            TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
            PermutationIterator;

    // type of the tiled_range iterator
    typedef PermutationIterator iterator;

    // construct tiled_range for the range [first,last)
    tiled_range(Iterator first, Iterator last, difference_type tiles)
        : first(first), last(last), tiles(tiles) {}

    iterator begin(void) const {
        return PermutationIterator(first,
                                   TransformIterator(CountingIterator(0),
                                                     tile_functor(tiles)));
    }

    iterator end(void) const { return begin() + tiles * (last - first); }

protected:
    Iterator first;
    Iterator last;
    difference_type tiles;
};

}  // namespace thrust