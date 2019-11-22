#pragma once
#include <type_traits>
#include <thrust/swap.h>

namespace cupoc {
namespace utility {

struct sp_counter_base
{
    std::size_t owner_count;
	__host__ __device__
    sp_counter_base() : owner_count(1) {}
	__host__
    virtual ~sp_counter_base() {}
	__host__
    virtual void dispose(void*) = 0;
};

template<typename T>
struct sp_counter_impl_p : sp_counter_base
{
    __host__ __device__
    sp_counter_impl_p() : sp_counter_base() {}
	__host__
    virtual void dispose(void* p) {if(p) cudaFree(reinterpret_cast<void*>(p));}
};


template<typename T>
class shared_ptr
{
public:
    typedef T element_type; //!< type of the managed object

private:
    void* current_ptr; //!< the stored pointer to the object itself is anonymous (sp_counter_base relies on a void* pointer for dispose)
    sp_counter_base* counter; //!< pointer to shared counting/disposal structure (is NULL iff current_ptr is NULL)

    template<typename U> friend class shared_ptr;

public:
    __host__ __device__
    shared_ptr() noexcept : current_ptr(NULL), counter(NULL) {}

    template<typename U>
    __host__ __device__
    explicit shared_ptr(U* ptr) : current_ptr(reinterpret_cast<void*>(ptr)), counter(NULL) {
        if(ptr) counter = new sp_counter_impl_p<T>();
    }

    template<typename U>
    __host__ __device__
    shared_ptr(const shared_ptr<U>& src, T* ptr) noexcept : current_ptr(ptr), counter(src.counter) {
        ++(counter->owner_count);
    }

    __host__ __device__
    shared_ptr( const shared_ptr& src ) noexcept : current_ptr(src.current_ptr), counter(src.counter) {
        if(counter) ++(counter->owner_count);
    }

	template<typename U>
	__host__ __device__
	shared_ptr(const shared_ptr<U>& src) noexcept : current_ptr(src.current_ptr), counter(src.counter) {
        if(counter) ++(counter->owner_count);
    }

    __host__
    ~shared_ptr() {
        if(counter) {
            --(counter->owner_count);
            if(counter->owner_count == 0) {
                counter->dispose(current_ptr);
                delete counter;
                counter = NULL;
            }
        }
    }

    __host__
    inline shared_ptr& operator=(const shared_ptr& src) noexcept {
        shared_ptr(src).swap(*this);
        return *this;
    }

    template<typename U>
    __host__ __device__
    inline shared_ptr& operator=( const shared_ptr<U>& src ) noexcept {
        shared_ptr(src).swap(*this);
        return *this;
    }

    __host__ __device__
    inline T* get() const noexcept {return reinterpret_cast<T*>(current_ptr);}

    __host__ __device__
    inline void swap(shared_ptr& other) noexcept {
        thrust::swap(current_ptr, other.current_ptr);
        thrust::swap(counter, other.counter);
    }
    __host__ __device__
    inline typename std::add_lvalue_reference<T>::type operator*() const noexcept {return *reinterpret_cast<T*>(current_ptr);}

    __host__ __device__
    inline T* operator->() const noexcept {return reinterpret_cast<T*>(current_ptr);}

    __host__ __device__
    operator bool() const noexcept {return get() != NULL;}

    template<typename T2> __host__ __device__ bool operator==(const shared_ptr<T2>& other) const noexcept {return get() == other.get();}
    template<typename T2> __host__ __device__ bool operator!=(const shared_ptr<T2>& other) const noexcept {return get() != other.get();}
    template<typename T2> __host__ __device__ bool operator< (const shared_ptr<T2>& other) const noexcept {return get() <  other.get();}
    template<typename T2> __host__ __device__ bool operator> (const shared_ptr<T2>& other) const noexcept {return get() >  other.get();}
    template<typename T2> __host__ __device__ bool operator<=(const shared_ptr<T2>& other) const noexcept {return get() <= other.get();}
    template<typename T2> __host__ __device__ bool operator>=(const shared_ptr<T2>& other) const noexcept {return get() >= other.get();}

};

}
}