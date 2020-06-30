#include <Tensile/Serialization/MessagePack.hpp>
#include <fstream>

namespace Tensile
{
    namespace Serialization
    {
        std::map<std::string, msgpack::object> objectToMap(msgpack::object& object)
        {
            assert(object.type == msgpack::type::object_type::MAP);

            std::map<std::string, msgpack::object> result;
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
            return result;
        }
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        MessagePackLoadLibraryFile(std::string const& filename)
    {
        std::ifstream        in(filename, std::ios::in | std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());

        return MessagePackLoadLibraryData<MyProblem, MySolution>(data);
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        MessagePackLoadLibraryData(std::vector<uint8_t> const& data)
    {
        std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

        auto result = msgpack::unpack((const char*)data.data(), data.size());
        Serialization::MessagePackInput min(result.get());

        Serialization::PointerMappingTraits<Tensile::MasterContractionLibrary,
                                            Serialization::MessagePackInput>::mapping(min, rv);

        return rv;
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        MessagePackLoadLibraryFile<ContractionProblem, ContractionSolution>(
            std::string const& filename);

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        MessagePackLoadLibraryData<ContractionProblem, ContractionSolution>(
            std::vector<uint8_t> const& data);
}
