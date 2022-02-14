/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2020 Advanced Micro Devices, Inc.
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

#include <Tensile/msgpack/MessagePack.hpp>

#include <Tensile/msgpack/Loading.hpp>

#include <stdio.h>

namespace Tensile
{
    namespace Serialization
    {
        void objectToMap(msgpack::object&                                  object,
                         std::unordered_map<std::string, msgpack::object>& result)
        {
            if(object.type != msgpack::type::object_type::MAP)
                throw std::runtime_error(concatenate("Expected MAP, found ", object.type));

            for(uint32_t i = 0; i < object.via.map.size; i++)
            {
                auto& element = object.via.map.ptr[i];

                std::string key;
                switch(element.key.type)
                {
                case msgpack::type::object_type::STR:
                {
                    element.key.convert(key);
                    break;
                }
                case msgpack::type::object_type::POSITIVE_INTEGER:
                {
                    auto iKey = element.key.as<uint32_t>();
                    key       = std::to_string(iKey);
                    break;
                }
                default:
                    throw std::runtime_error("Unexpected map key type");
                }

                result[key] = std::move(element.val);
            }
        }
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        MessagePackLoadLibraryFile(std::string const& filename)
    {
        // parse file into a msgpack::object_handle
        FILE* f = fopen(filename.c_str(), "rb");
        if (!f)
        {
            if(Debug::Instance().printDataInit())
                perror("Error loading msgpack data");

            return nullptr;
        }

        msgpack::object_handle result;
        try
        {
            constexpr size_t buffer_size = 1 << 19;

            msgpack::unpacker unp;
            bool finished_parsing;
            do
            {
                unp.reserve_buffer(buffer_size);
                size_t bytes_read = fread(unp.buffer(), 1, buffer_size, f);
                unp.buffer_consumed(bytes_read);
                finished_parsing = unp.next(result); // may throw msgpack::parse_error
            } while(!finished_parsing && !ferror(f));

            if(!finished_parsing)
            {
                if(feof(f))
                    std::cerr << "Unexpected end of file: " << filename << std::endl;
                else
                    perror("Error reading Tensile library file");

                fclose(f);
                return nullptr;
            }
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cerr << "Error loading msgpack data:" << std::endl << exc.what() << std::endl;

            fclose(f);
            return nullptr;
        }
        fclose(f);

        // copy data from msgpack::object_handle into MasterSolutionLibrary
        try
        {
            std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

            Serialization::MessagePackInput min(result.get());

            Serialization::PointerMappingTraits<Tensile::MasterContractionLibrary,
                                                Serialization::MessagePackInput>::mapping(min, rv);

            if(!min.error.empty())
            {
                std::ostringstream msg;
                msg << "Error loading msgpack data:" << std::endl;
                for(auto const& err : min.error)
                    msg << err << std::endl;

                throw std::runtime_error(msg.str());
            }

            return rv;
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cerr << "Error loading msgpack data:" << std::endl << exc.what() << std::endl;

            return nullptr;
        }
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        MessagePackLoadLibraryData(std::vector<uint8_t> const& data)
    {
        try
        {
            std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

            auto result = msgpack::unpack((const char*)data.data(), data.size());
            Serialization::MessagePackInput min(result.get());

            Serialization::PointerMappingTraits<Tensile::MasterContractionLibrary,
                                                Serialization::MessagePackInput>::mapping(min, rv);

            if(!min.error.empty())
            {
                std::ostringstream msg;
                msg << "Error loading msgpack data:" << std::endl;
                for(auto const& err : min.error)
                    msg << err << std::endl;

                throw std::runtime_error(msg.str());
            }

            return rv;
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading msgpack data:" << std::endl << exc.what() << std::endl;

            return nullptr;
        }
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        MessagePackLoadLibraryFile<ContractionProblem, ContractionSolution>(
            std::string const& filename);

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        MessagePackLoadLibraryData<ContractionProblem, ContractionSolution>(
            std::vector<uint8_t> const& data);
}
