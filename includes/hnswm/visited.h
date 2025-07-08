#pragma once
#include <stdint.h>
#include <stdexcept>

#define counter_t uint32_t

class VisitedCheck
{
private:
    counter_t curCounterVal_;
    counter_t *data_;
    uint64_t size_;

public:
    VisitedCheck(uint64_t size = 80000)
    {
        this->size_ = size;
        this->data_ = static_cast<counter_t*>(calloc(size, sizeof(counter_t)));
        this->curCounterVal_ = 0;
    }

    // reset the visited set
    void clear()
    {
        this->curCounterVal_++;
    }

    // mark visited
    void insert(uint32_t x)
    {
        if (x >= this->size_)
            throw std::runtime_error("Visited set was given a value larger than what was allocated");
        this->data_[x] = this->curCounterVal_;
    }

    // check if visited
    bool contains(uint32_t x)
    {
        return (this->data_[x] == this->curCounterVal_);
    }

    void reserve(uint64_t size)
    {
        if (size <= this->size_)
            return;
        if (this->data_)
            free(this->data_);
        this->size_ = size;
        this->data_ = static_cast<counter_t*>(calloc(size, sizeof(counter_t)));
    }
};
