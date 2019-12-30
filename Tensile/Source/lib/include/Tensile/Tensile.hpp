/*******************************************************************************
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *******************************************************************************/

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include <Tensile/Macros.hpp>
#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/SolutionLibrary_fwd.hpp>

#include <Tensile/KernelArguments.hpp>
#include <Tensile/geom.hpp>

/**
 * \mainpage 
 * 
 * Tensile is a tool for creating a library of tensor contractions, including
 * matrix multiplications in a benchmark-driven manner.
 * 
 * The host library contains classes and functions for selecting one or more
 * kernels to launch to solve a particular problem, and then launching those
 * kernels.  Kernels are selected based on the performance on a particular GPU
 * (from the benchmark runs) as well as the limitations and applicability of
 * the kernels to particular problem sizes.
 *
 * Design goals for the host library include:
 *  - A library that requires no generated code on the host
 *  - Kernel selection configured via a file
 *      - No recompilation needed to enable a different set of kernels.
 *  - Kernel selection & determination of kernel launches decoupled from host
 *  runtime language
 *      - Eliminate as much of the Hip and OpenCL-specific code as possible
 *  - Support future work including implicit GEMMs
 *  - Use modern C++
 *  - Eliminate the need for client code to call intricately named functions
 *    such as `tensile_Cijk_Ailk_Bljk_SB()`.
 */

/**
 * \addtogroup Tensile
 * @{
 */

/**
 * @brief Primary namespace for Tensile host code.
 */
namespace Tensile
{
    template<typename T>
    class TENSILE_API ProblemKey
    {
    public:
        template<typename... Ts>
        ProblemKey(Ts... args) {
            init(args...);
        }

        virtual ~ProblemKey() = default;

        void addKeyAttribute(T value)
        {
            _keys.push_back(value);
        }

        bool operator< (const ProblemKey<T> & p) const
        {
            for (unsigned long i=0; i<this->_keys.size(); i++) {
                if (this->_keys[i] < p._keys[i])
                    return true;
                else if (this->_keys[i] > p._keys[i])
                    return false;
            }
            if (p._keys.size() == this->_keys.size())
                return false; // get here if all indices are equal
            else if (this->_keys.size() < p._keys.size())
                return true;
            else
                return false;
        };

        bool operator== (const ProblemKey<T> & p) const
        {
            if (p._keys.size() != this->_keys.size())
                return false;
            for (unsigned long i=0; i<this->_keys.size(); i++) {
                if (p._keys[i] != this->_keys[i])
                    return false;
            }
            return true;
        };

        size_t hash() const {
            size_t h=0;
            for (int i=0; i<_keys.size(); i++) {
                h ^= _keys[i] + 0x9b9773e99e3779b9 + (h<<6) + (h>>2);
            }
            return h;
        }

    private:
        void init (T value) {
            _keys.push_back(value);
        }

        void init (std::vector<T> values) {
            _keys.insert(_keys.end(), values.begin(), values.end());
        }

        template<typename... Ts>
        void init (std::vector<T> values, Ts... args) {
            _keys.insert(_keys.end(), values.begin(), values.end());
            init (args...);
        }

        template<typename... Ts>
        void init(T value, Ts... args) {
            _keys.push_back(value);
            init(args ...);
        }

    private:
	std::vector<T> _keys;
    };

    template <typename T>
    struct ProblemKeyHash {
        size_t operator()(const ProblemKey<T> &problemKey ) const
        {
            return problemKey.hash();
        }
    };

    template <typename MyProblemKey, typename MyProblemKeyHash, typename MySolution>
    class TENSILE_API CachedProblemMap
    {
    public:
        std::shared_ptr<MySolution> find(MyProblemKey key)
        {
            decltype(problemMap.end()) theSolution;
            {
                // Acquire a shared lock for reading map
                std::shared_lock<std::shared_timed_mutex> lock(mutex);

                // Look up the tuple in the map
                theSolution = problemMap.find(key);

                // If tuple already exists, atomically increment count and return
                if(theSolution != problemMap.end())
                {
                    return theSolution->second;
                }
            } // Release shared lock

            return nullptr;
        }

        void add (MyProblemKey key, std::shared_ptr<MySolution> entry)
        {
            // Acquire an exclusive lock for modifying map
            std::lock_guard<std::shared_timed_mutex> lock(mutex);

            // If doesn't already exist, insert tuple by moving
            problemMap.emplace(key, entry);
        }


    private:
        std::unordered_map<MyProblemKey, std::shared_ptr<MySolution>, MyProblemKeyHash> problemMap;
        std::shared_timed_mutex mutex;
    };

    /**
     * \ingroup Tensile
     * \defgroup  Problem Problem Definition
     * 
     * @brief Classes for defining problems
     */

    /**
     * \ingroup Problem
     * Base Problem class. A Problem object generically describes a problem to
     * be solved including the type of problem, all sizes and strides, but not
     * including actual pointers to data.
     */
    class TENSILE_API Problem
    {
    public:
        virtual ~Problem();
        virtual ProblemKey<size_t> getKey() const = 0;
        virtual std::string description() const = 0;
    };

    /**
     * \ingroup Problem
     * Base class for problem inputs. This stores the actual pointers to the
     * data.
     */
    class TENSILE_API ProblemInputs
    {
    public:
        virtual ~ProblemInputs();
    };

    /**
     * \ingroup Tensile
     * \defgroup Launching Kernel Launching
     */

    /**
     * \ingroup Launching
     * Describes a single kernel invocation including kernel name, launch
     * bounds, and arguments.
     */
    struct TENSILE_API KernelInvocation
    {
    public:
        std::string kernelName;

        dim3 workGroupSize;
        dim3 numWorkGroups;
        dim3 numWorkItems;
        size_t sharedMemBytes = 0;

        KernelArguments args;
    };

    /**
     * \ingroup Tensile
     * \defgroup Hardware Hardware Description Classes
     */

    /**
     * \ingroup Hardware
     * Abstract base class for describing hardware capabilities and properties.
     */
    class TENSILE_API Hardware
    {
    public:
        Hardware();
        virtual ~Hardware();

        virtual size_t id() const
        {
            return 0;
        }
        virtual std::string description() const = 0;
    };

    /**
     * \ingroup Solution
     * Generally encapsulates a single kernel or set of kernels that can be
     * used to solve a particular problem.
     */
    class TENSILE_API Solution
    {
    public:
        virtual ~Solution();

        virtual std::string name() const = 0;
        virtual std::string description() const = 0;

    };

    /**
     * \ingroup Launching
     * Base class for objects capable of launching kernels based on
     * KernelArguments objects.
     */
    class TENSILE_API SolutionAdapter
    {
    public:
        virtual ~SolutionAdapter();

        virtual std::string name() const = 0;
    };

#ifdef TENSILE_DEFAULT_SERIALIZATION
    /**
     * Interface for deserializing a library file.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    TENSILE_API std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> LoadLibraryFile(std::string const& filename);
#endif
}

/** @} */
