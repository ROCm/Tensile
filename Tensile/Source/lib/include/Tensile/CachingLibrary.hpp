
/*******************************************************************************
 * MIT License
 *
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
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

#include <atomic>
#include <shared_mutex>
#include <unordered_map>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

namespace Tensile
{
    template <typename Key, typename Value>
    class CacheMap
    {
    public:
        CacheMap(Value const& nullValue)
            : m_nullValue(nullValue),
              m_lookupEfficiency(Debug::Instance().printLookupEfficiency()),
              m_lookups(0),
              m_hits(0)

        {}

        ~CacheMap()
        {
            if(m_lookupEfficiency)
                std::cout << "CacheMap: " << m_hits << "/" << m_lookups << " cache hits" << std::endl;
        }

        Value find(Key const& key)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_map.find(key);

            if(m_lookupEfficiency)
            {
                m_lookups++;
                if(iter != m_map.end())
                    m_hits++;
            }

            if(iter != m_map.end())
                return iter->second;

            return m_nullValue;
        }

        void add(Key const& key, Value const& value)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            m_map.emplace(key, value);
        }

    private:
        std::unordered_map<Key, Value> m_map;
        std::shared_timed_mutex m_mutex;
        Value m_nullValue;

        bool m_lookupEfficiency;
        std::atomic<int64_t> m_lookups;
        std::atomic<int64_t> m_hits;
    };

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class CachingLibrary: public SolutionLibrary<MyProblem, MySolution>
    {
    public:
        using Library = SolutionLibrary<MyProblem, MySolution>;
        using Key = std::tuple<MyProblem, AMDGPU>;
        using Cache = CacheMap<Key, std::shared_ptr<MySolution>>;

        CachingLibrary(std::shared_ptr<Library> subLibrary)
            : m_subLibrary(subLibrary),
              m_cache(nullptr)
        {}

        virtual std::shared_ptr<MySolution>
            findBestSolution(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            try
            {
                auto amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                auto key = std::make_tuple(problem, amdgpu);

                auto rv = m_cache.find(key);
                if(rv) return rv;

                rv = m_subLibrary->findBestSolution(problem, hardware);
                if(rv)
                    m_cache.add(key, rv);

                return rv;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findBestSolution(problem, hardware);
            }
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            return m_subLibrary->findAllSolutions(problem, hardware);
        }

        std::shared_ptr<MySolution>
            findSolutionInCache(MyProblem const& problem,
                                Hardware  const& hardware) const
        {
            auto amdgpu = dynamic_cast<AMDGPU const&>(hardware);
            auto key = std::make_tuple(problem, amdgpu);

            return m_cache.find(key);
        }

        virtual std::string type() const override
        {
            return "Caching Library";
        }
        virtual std::string description() const override
        {
            return "Caching Library";
        }

        std::shared_ptr<Library> library() const
        {
            return m_subLibrary;
        }

    private:
        std::shared_ptr<Library> m_subLibrary;
        mutable Cache m_cache;
    };

#if 0
    struct ContractionCachingLibrary: public CachingLibrary<ContractionProblem>
    {
        using Super = CachingLibrary<ContractionProblem>;
        using Library = typename Super::Library;
        using Key = typename Super::Key;

        ContractionCachingLibrary(std::shared_ptr<Library> subLibrary)
            : CachingLibrary<ContractionProblem>(subLibrary)
        {}

    };
#endif

}
