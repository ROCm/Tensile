/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <CSVStackFile.hpp>
#include <iomanip>

#include <Tensile/Utils.hpp>

namespace Tensile
{
    namespace Client
    {
        CSVStackFile::CSVStackFile(std::string const& filename)
            : m_stream(new std::ofstream(filename.c_str()))
        {
        }

        void null_deleter(void* ptr) {}

        CSVStackFile::CSVStackFile(std::ostream& stream)
            : m_stream(&stream, null_deleter)
        {
        }

        CSVStackFile::CSVStackFile(std::shared_ptr<std::ostream> stream)
            : m_stream(stream)
        {
        }

        CSVStackFile::~CSVStackFile() {}

        void CSVStackFile::setHeaderForKey(std::string const& key, std::string const& header)
        {
            if(m_headers.find(key) == m_headers.end())
                m_keyOrder.push_back(key);

            m_headers[key] = header;
        }

        void CSVStackFile::push()
        {
            m_stack.push_back(m_currentRow);
        }

        void CSVStackFile::pop()
        {
            m_currentRow = m_stack.back();
            m_stack.pop_back();
        }

        void CSVStackFile::writeCurrentRow()
        {
            if(m_firstRow && !m_headers.empty())
                writeRow(m_headers);

            m_firstRow = false;

            writeRow(m_currentRow);

            if(m_stack.empty())
                m_currentRow.clear();
            else
                m_currentRow = m_stack.back();
        }

        void CSVStackFile::writeRow(std::unordered_map<std::string, std::string> const& row)
        {
            bool firstCol = true;
            for(auto const& key : m_keyOrder)
            {
                if(!firstCol)
                    (*m_stream) << ", ";

                std::string value = "";

                auto it = row.find(key);
                if(it != row.end())
                    value = escape(it->second);

                (*m_stream) << value;

                firstCol = false;
            }

            (*m_stream) << std::endl;
        }

        std::string CSVStackFile::escape(std::string const& value)
        {
            // An actual quote needs more attention.
            if(value.find('"') != std::string::npos)
                return escapeQuote(value);

            bool needQuote = false;

            std::string badValues = ",\n\r";
            for(char c : badValues)
            {
                if(value.find(c) != std::string::npos)
                {
                    needQuote = true;
                    break;
                }
            }

            if(needQuote)
                return concatenate("\"", value, "\"");

            return value;
        }

        std::string CSVStackFile::escapeQuote(std::string const& value)
        {
            std::ostringstream rv;
            rv << '"';

            for(char c : value)
            {
                if(c == '"')
                    rv << "\"\"";
                else
                    rv << c;
            }

            rv << '"';

            return rv.str();
        }

        void CSVStackFile::setValueForKey(std::string const& key, std::string const& value)
        {
            m_currentRow[key] = value;
        }

        void CSVStackFile::setValueForKey(std::string const& key, double const& value)
        {
            if(value > 0.01)
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << value;
                setValueForKey(key, ss.str());
            }
            else
                setValueForKey(key, boost::lexical_cast<std::string>(value));
        }
    } // namespace Client
} // namespace Tensile
