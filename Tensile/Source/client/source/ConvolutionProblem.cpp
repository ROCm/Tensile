/**
 * MIT License
 *
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <ConvolutionProblem.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>


namespace Tensile
{
    const size_t ConvolutionProblem::InvalidPos=-1;
    ConvolutionProblem::ActivationFormat::ActivationFormat() :
        m_filterPositions(MaxNumSpatialDims,0)
    {
    }
    void ConvolutionProblem::ActivationFormat::FromIdentifier(std::string identifier,
            size_t formatNumSpatialDims, size_t numSpatialDims, std::vector<size_t> *filters)
    {
        // summation dimensions immediately follow the spatial dim(s)
        m_formatIdentifier = identifier;
        if (identifier == "NCHW")
        {
            assert(formatNumSpatialDims == 2);
            m_format = TensorFormat::NCHW;

            size_t position = 0;
            if (filters)
                for (int fi=0; fi<filters->size(); fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = position++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }
            for (auto si=0; si<numSpatialDims; si++)
                m_spatialPositions.push_back(position++);

            m_channelPosition = position++;
            m_batchPosition = position++;
        }
        else if (identifier == "NHWC")
        {
            assert(formatNumSpatialDims == 2);
            m_format = TensorFormat::NHWC;

            m_channelPosition = 0;

            size_t position = m_channelPosition+1;
            if (filters)
                for (int fi=0; fi<filters->size(); fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = position++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }

            // assume spatial dimensions are collapsed here:
            std::cout << "FIXME\n";
            for (auto si=0; si<numSpatialDims; si++)
                m_spatialPositions.push_back(position++);
            m_batchPosition = position++;
        }
        else if (identifier == "CNHW")
        {
            assert(formatNumSpatialDims == 2);
            m_format = TensorFormat::CNHW;

            size_t position = 0;
            if (filters)
                for (int fi=0; fi<filters->size(); fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = position++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }
            // assume spatial dimensions are collapsed here:
            std::cout << "FIXME\n";
            for (auto si=0; si<numSpatialDims; si++)
                m_spatialPositions.push_back(position++);
            m_batchPosition   = position++;
            m_channelPosition = position++;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid tensor format in convolution identifier:") +
                identifier);
        }
    }

    std::string ConvolutionProblem::ActivationFormat::description() const
    {
        std::ostringstream rv;
        rv << m_formatIdentifier << "_"
           << " batchPosition=" << m_batchPosition
           << " channelPosition=" << m_channelPosition;
        rv << " spatialPositions[0,1,2]=";
        for (auto i=0;i<m_spatialPositions.size();i++)
        {
            if (i!=0)
                rv << ",";
            rv << static_cast<int>(m_spatialPositions[i]);
        }

        rv << " filterPosition[0,1,2]="
           << static_cast<int64_t>(m_filterPositions[0]) << ","
           << static_cast<int64_t>(m_filterPositions[1]) << ","
           << static_cast<int64_t>(m_filterPositions[2]);
        return rv.str();
    }

    ConvolutionProblem::WeightFormat::WeightFormat() :
        m_filterPositions(MaxNumSpatialDims,0)
    {
    }

    void ConvolutionProblem::WeightFormat::FromIdentifier(std::string identifier, bool transposeCK,
            size_t formatNumSpatialDims, std::vector<size_t> *filters)
    {
        m_formatIdentifier = identifier;
        if (identifier == "KCYX")
            m_format = TensorFormat::KCYX;
        else if (identifier == "CKYX")
            m_format = TensorFormat::CKYX;
        else
          throw std::runtime_error(std::string("Invalid weight format in convolution identifier:") +
              identifier);

        if ((identifier == "KCYX" and !transposeCK) ||
            (identifier == "CKYX" and  transposeCK)) {

            assert(formatNumSpatialDims == 2);

            size_t position = 0;
            if (filters)
                // Weight dims are assigned in reverse order for optimal Tensile summation processing
                for (int fi=0; fi<filters->size(); fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = position++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }
            m_cinPosition = position++;
            m_coutPosition = position;

        } else if ((identifier == "CKYX" and !transposeCK) ||
                   (identifier == "KCYX" and  transposeCK)) {
            assert(formatNumSpatialDims == 2);
            size_t filterPosition = 0; // TODO -> change to position
            if (filters)
                for (int fi=0; fi<filters->size(); fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = filterPosition++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }
            m_coutPosition = filterPosition;
            m_cinPosition  = m_coutPosition+1;
        } else {
          throw std::runtime_error(std::string("Invalid weight format in convolution identifier:") +
              identifier);
        }
    }
    std::string ConvolutionProblem::WeightFormat::description() const
    {
        std::ostringstream rv;
        rv << m_formatIdentifier << "_"
           << " coutPosition=" << m_coutPosition
           << " cinPosition=" << m_cinPosition
           << " filterPositions="
                << static_cast<int64_t>(m_filterPositions[0]) << ","
                << static_cast<int64_t>(m_filterPositions[1]) << ","
                << static_cast<int64_t>(m_filterPositions[2]);
        return rv.str();
    }

    // Setup for forward or backward data
    void ConvolutionProblem::LoopCounts::setupForData(
        ConvolutionProblem const& convProblem, ContractionProblem const& problem)
    {
        batchCount = problem.a().sizes()[convProblem.formatA().batchPosition()];
        cinCount = problem.a().sizes()[convProblem.formatA().channelPosition()];
        coutCount = problem.b().sizes()[convProblem.formatB().weights().coutPosition()];
        for (int si=0; si<convProblem.formatA().spatialPositions().size(); si++)
        {
            auto spatialPositionA = convProblem.formatA().spatialPositions()[si];
            auto const problemSpatialSize = problem.a().sizes()[spatialPositionA];
            scount[si] = problemSpatialSize;
        }

        // Setup filter counts, translate -1 to the filter dim from problem size
        // fcount[0] is X
        for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
        {
            auto const filterPositionA = convProblem.formatA().filterPositions()[fi];
            if (filterPositionA != ConvolutionProblem::InvalidPos)
            {
                auto const convFilterSize = convProblem.filter()[fi]; // filter from convolution-identifier
                auto const problemFilterSize = problem.a().sizes()[filterPositionA];
                if (convFilterSize != -1)
                  assert(convFilterSize == problemFilterSize);
                fcount[fi] = problemFilterSize;
            }
        }

    }


    void ConvolutionProblem::FromIdentifier(std::string identifier)
    {
      // example identifier: ConvolutionForward_NCHW_KCHW_NCHW_filter:3x3x1_stride:1x1x1_dilation:1x1x1_groups:1
      std::vector<std::string> parts;
      boost::split(parts, identifier, boost::algorithm::is_any_of("_"));

      if (parts.size() < 4)
        // id, formatA, formatB, outputTensor
        throw std::runtime_error(std::string("Invalid convolution identifier- must have at least 3 sections:") + identifier);

      m_operationIdentifier = parts[0];
      size_t formatNumSpatialDims = parts[1].size()-2;

      for (auto part = parts.begin() + 4; part != parts.end(); part++)
      {
          std::vector<std::string> flags;
          std::vector<std::string> xvals;
          boost::split(flags, *part, boost::algorithm::is_any_of(":"));
          assert(flags.size() == 2); // must be key:value pair

          m_spatials.resize(MaxNumSpatialDims, -1);
          m_filters.resize(MaxNumSpatialDims, 1);
          m_strides.resize(MaxNumSpatialDims, 1);
          m_dilations.resize(MaxNumSpatialDims, 1);
          m_padStart.resize(MaxNumSpatialDims, 0);
          m_padEnd.resize(MaxNumSpatialDims, 0);

          if (flags[0] == "spatialDims")
            m_numSpatialDims = boost::lexical_cast<size_t>(flags[1]);
          else if (flags[0] == "indices")
          {
          }
          else if (flags[0] == "spatial")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = formatNumSpatialDims;
            for (auto x : xvals) {
              m_spatials.at(--i) = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "filter")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = formatNumSpatialDims;
            for (auto x : xvals) {
              m_filters.at(--i) = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "stride")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = formatNumSpatialDims;
            for (auto x : xvals) {
              m_strides.at(--i) = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "dilation")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = formatNumSpatialDims;
            for (auto x : xvals) {
              m_dilations.at(--i) = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "padStart")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = formatNumSpatialDims;
            for (auto x : xvals) {
              m_padStart.at(--i) = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "padEnd")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = formatNumSpatialDims;
            for (auto x : xvals) {
              m_padEnd.at(--i) = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "groups")
          {
            m_groups = boost::lexical_cast<int>(flags[1]);
            assert (m_groups == 1); // not supported yet
          }
          else
            throw std::runtime_error(std::string("Invalid flag in convolution identifier:") + flags[0]);

          std::cout << flags[0] << ":::" << flags[1] << "\n";
      };

      // Compute number of expected filter summation dims (for non-unit filters)
      m_numFilterDims = 0; // TODO-.backward-weights
      for (auto f : m_filters)
      {
        if (f != 1)
          m_numFilterDims++;
      }

      if (m_operationIdentifier == "ConvolutionForward" || m_operationIdentifier=="ConvolutionBackwardData")
      {
          m_formatA.FromIdentifier(parts[1], formatNumSpatialDims, m_numSpatialDims, &m_filters);
          m_formatB.weightsW().FromIdentifier(parts[2], m_operationIdentifier=="ConvolutionBackwardData",
               formatNumSpatialDims, &m_filters);
          m_formatD.activationW().FromIdentifier(parts[3], formatNumSpatialDims, m_numSpatialDims, nullptr);
      }
      else
      {
          throw std::runtime_error(std::string("Invalid operation identifier:") +
              m_operationIdentifier);
      }
    }

    void ConvolutionProblem::validate(const ContractionProblem &problem) const
    {
        if (1)
        {
            std::cout << "validate::\n";

            std::cout << "  freeAIndices: ";
            for (auto i : problem.freeIndicesA())
                std::cout << i << ",";

            std::cout << "\n  freeBIndices";
            for (auto i : problem.freeIndicesB())
                std::cout << i << ",";

            std::cout << "\n  batchIndices: ";
            for (auto i : problem.batchIndices())
                std::cout << i << ",";

            std::cout << "\n  summationIndicies: ";
            for (auto i : problem.boundIndices())
                std::cout << i << ",";

            std::cout << "\n";

            if (m_operationIdentifier == "ConvolutionForward")
            {
                std::cout << "  convProblem.formatA        :" << formatA().description() << "\n";
                std::cout << "  convProblem.formatB.weights:" << formatB().weights().description() << "\n";
            }
            else
            {
                throw std::runtime_error(std::string("Unsupported operation identifier for check") +
                    m_operationIdentifier);
            }
        }

        // Ensure positions are where we expect them to be in the convolution tensor description:
        assert(problem.batchIndices().end() !=
            std::find_if(problem.batchIndices().begin(), problem.batchIndices().end(),
            [this](const ContractionProblem::BatchIndex &bi)
            {return bi.a == this->formatA().batchPosition();}));

        assert(problem.boundIndices().end() !=
            std::find_if(problem.boundIndices().begin(), problem.boundIndices().end(),
            [this](const ContractionProblem::BoundIndex &bi)
            {return bi.a == this->formatA().channelPosition();}));
        if (m_operationIdentifier == "ConvolutionForward")
        {
            for (int i=0; i<ConvolutionProblem::MaxNumSpatialDims; i++)
            {
                auto const filterPositionA = formatA().filterPositions()[i];
                if (filterPositionA != ConvolutionProblem::InvalidPos)
                    assert(problem.boundIndices().end() !=
                        std::find_if(problem.boundIndices().begin(), problem.boundIndices().end(),
                        [filterPositionA](const ContractionProblem::BoundIndex &bi)
                        {return bi.a == filterPositionA;}));

                auto const filterPositionB = formatB().weights().filterPositions()[i];
                if (filterPositionB != ConvolutionProblem::InvalidPos)
                    assert(problem.boundIndices().end() !=
                        std::find_if(problem.boundIndices().begin(), problem.boundIndices().end(),
                        [filterPositionB](const ContractionProblem::BoundIndex &bi)
                        {return bi.b == filterPositionB;}));
            }
            for (auto s : formatA().spatialPositions())
                assert(problem.freeIndicesA().end() !=
                    std::find_if(problem.freeIndicesA().begin(), problem.freeIndicesA().end(),
                    [s](const ContractionProblem::FreeIndex &bi)
                    {return bi.isA && bi.i == s;}));
        }
        else
          throw std::runtime_error(std::string("Unsupported operation identifier for check") +
              m_operationIdentifier);

    }

    template <typename T>
    static std::string delimitedVector(const std::vector<T> &v, std::string delimiter)
    {
      std::ostringstream rv;

      std::string delim;
      for (auto e : v) {
        rv << delim << e;
        delim = delimiter;
      }
      return rv.str();
    }

    std::string ConvolutionProblem::description() const
    {
        std::ostringstream rv;

        rv << operationIdentifier() ;

        rv << "_filter:" << delimitedVector(m_filters, "x");
        rv << "_stride:" << delimitedVector(m_strides, "x");
        rv << "_dilation:" << delimitedVector(m_dilations, "x");
        rv << "_padStart:" << delimitedVector(m_padStart, "x");
        rv << "_padEnd:" << delimitedVector(m_padEnd, "x");

        return rv.str();
    }

    TENSILE_API std::ostream & operator<<(std::ostream & stream, ConvolutionProblem const& convolution)
    {
        return stream << convolution.description();
    }

}

