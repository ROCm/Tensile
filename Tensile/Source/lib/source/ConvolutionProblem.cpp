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

#include <Tensile/ConvolutionProblem.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>


namespace Tensile
{
    ConvolutionProblem::ActivationFormat::ActivationFormat() :
        m_filterPositions(MaxNumSpatialDims,0),
        m_spatialPositions(MaxNumSpatialDims,0)
    {
    }
    void ConvolutionProblem::ActivationFormat::FromIdentifier(std::string identifier,
            size_t numSpatialDims, std::vector<size_t> *filters)
    {
        // summation dimensions immediately follow the spatial dim(s)
        m_formatIdentifier = identifier;
        if (identifier == "NCHW")
        {
            assert(numSpatialDims == 2);
            m_format = TensorFormat::NCHW;

            size_t filterPosition = 0;
            if (filters)
                for (auto fi=0;fi<filters->size();fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = filterPosition++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }

            // assume spatial dimensions are collapsed here:
            m_spatialPositions[0] = filterPosition;
            m_channelPosition = m_spatialPositions[0] + 1;
            m_batchPosition = m_channelPosition + 1;

        }
        else if (identifier == "NHWC")
        {
            assert(numSpatialDims == 2);
            m_format = TensorFormat::NHWC;

            m_channelPosition = 0;

            size_t filterPosition = m_channelPosition+1;
            if (filters)
                for (auto fi=0;fi<filters->size();fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = filterPosition++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }

            // assume spatial dimensions are collapsed here:
            m_spatialPositions[0] = filterPosition;
            m_batchPosition = m_spatialPositions[0] + 1;
        }
        else if (identifier == "CNHW")
        {
            assert(numSpatialDims == 2);
            m_format = TensorFormat::CNHW;

            size_t filterPosition = 0;
            if (filters)
                for (auto fi=0;fi<filters->size();fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = filterPosition++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }
            // assume spatial dimensions are collapsed here:
            m_spatialPositions[0] = filterPosition;
            m_batchPosition   = m_spatialPositions[0] + 1;
            m_channelPosition = m_batchPosition + 1;
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
           << " channelPosition=" << m_channelPosition
           << " spatialPosition=" << m_spatialPositions[0]
           << " filterPosition=" << m_filterPositions[0] << "," << m_filterPositions[1] << "," << m_filterPositions[2];
        return rv.str();
    }
    ConvolutionProblem::WeightFormat::WeightFormat() :
        m_filterPositions(MaxNumSpatialDims,0)
    {
    }
    void ConvolutionProblem::WeightFormat::FromIdentifier(std::string identifier, bool transposeCK,
            size_t numSpatialDims, std::vector<size_t> *filters)
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

            size_t filterPosition = 0;
            if (filters)
                for (auto fi=0;fi<filters->size();fi++)
                {
                    if ((*filters)[fi] != 1)
                        m_filterPositions[fi] = filterPosition++;
                    else
                        m_filterPositions[fi] = InvalidPos;
                }
            m_cinPosition = filterPosition;
            m_coutPosition = m_cinPosition+1;

        } else if ((identifier == "CKYX" and !transposeCK) ||
                   (identifier == "KCYX" and  transposeCK)) {
            size_t filterPosition = 0;
            if (filters)
                for (auto fi=0;fi<filters->size();fi++)
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
           << " coutIndex=" << m_coutPosition
           << " cinIndex=" << m_cinPosition
           << " filterPositions=" << m_filterPositions[0] << "," << m_filterPositions[1] << "," << m_filterPositions[2];
        return rv.str();
    }

    void ConvolutionProblem::FromIdentifier(std::string identifier)
    {
      // example identifier: ConvolutionForward_NCHW_KCHW_NCHW_filter:3x3x1_stride:1x1x1_dilation:1x1x1_groups:1
      std::vector<std::string> parts;
      boost::split(parts, identifier, boost::algorithm::is_any_of("_"));

      if (parts.size() < 4)
        // id, tensora, tensorb, outputTensor
        throw std::runtime_error(std::string("Invalid convolution identifier- must have at least 3 sections:") + identifier);

      m_operationIdentifier = parts[0];
      m_numSpatialDims = parts[1].size()-2;

      for (auto part = parts.begin() + 4; part != parts.end(); part++)
      {
          std::vector<std::string> flags;
          std::vector<std::string> xvals;
          boost::split(flags, *part, boost::algorithm::is_any_of(":"));
          assert(flags.size() == 2); // must be key:value pair

          m_filter.resize(MaxNumSpatialDims, 1);
          m_stride.resize(MaxNumSpatialDims, 1);
          m_dilation.resize(MaxNumSpatialDims, 1);
          m_padStart.resize(MaxNumSpatialDims, 0);
          m_padEnd.resize(MaxNumSpatialDims, 0);

          if (flags[0] == "ps")
            m_packSpatial = boost::lexical_cast<int>(flags[1]);
          else if (flags[0] == "filter")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = MaxNumSpatialDims-m_numSpatialDims;
            for (auto x : xvals) {
              m_filter[i++] = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "stride")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = MaxNumSpatialDims-m_numSpatialDims;
            for (auto x : xvals) {
              m_stride[i++] = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "dilation")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = MaxNumSpatialDims-m_numSpatialDims;
            for (auto x : xvals) {
              m_dilation[i++] = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "padStart")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = MaxNumSpatialDims-m_numSpatialDims;
            for (auto x : xvals) {
              m_padStart[i++] = boost::lexical_cast<size_t>(x);
            }
          }
          else if (flags[0] == "padEnd")
          {
            boost::split(xvals, flags[1], boost::algorithm::is_any_of("x"));
            int i = MaxNumSpatialDims-m_numSpatialDims;
            for (auto x : xvals) {
              m_padEnd[i++] = boost::lexical_cast<size_t>(x);
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
      for (auto f : m_filter)
      {
        if (f != 1)
          m_numFilterDims++;
      }

      if (m_operationIdentifier == "ConvolutionForward" || m_operationIdentifier=="ConvolutionBackwardData")
      {
          m_tensorA.FromIdentifier(parts[1], m_numSpatialDims, &m_filter);
          m_tensorB.weightsW().FromIdentifier(parts[2], m_operationIdentifier=="ConvolutionBackwardData",
              m_numSpatialDims, &m_filter);
          m_tensorD.activationW().FromIdentifier(parts[3], m_numSpatialDims, nullptr);
      }
      else
      {
          throw std::runtime_error(std::string("Invalid operation identifier:") +
              m_operationIdentifier);
      }
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

        rv << "_filter:" << delimitedVector(m_filter, "x");
        rv << "_stride:" << delimitedVector(m_stride, "x");
        rv << "_dilation:" << delimitedVector(m_dilation, "x");
        rv << "_padStart:" << delimitedVector(m_padStart, "x");
        rv << "_padEnd:" << delimitedVector(m_padEnd, "x");

        return rv.str();
    }

    TENSILE_API std::ostream & operator<<(std::ostream & stream, ConvolutionProblem const& convolution)
    {
        return stream << convolution.description();
    }

}

