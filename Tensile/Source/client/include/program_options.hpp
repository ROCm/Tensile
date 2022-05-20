/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
// This emulates removed dependency of boost library.

#ifndef PROGRAM_OPTIONS_H
#define PROGRAM_OPTIONS_H

#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/TensorOps.hpp>
#include "ResultReporter.hpp"

#include <cinttypes>
#include <cstdio>
#include <iomanip>
#include <map>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

#define DEBUG_ENABLE 0
#define DEBUG_LOG_PO(__VA_ARGS__) if (DEBUG_ENABLE) printf(__VA_ARGS__)

using namespace Tensile;

namespace roc
{
    template <typename Target>
    inline Target lexical_cast(const std::string &arg)
    {
        Target result = Target(std::stoull(arg, 0, 10));
        return result;
    }
    
    inline std::string lexical_cast_to_string(const std::string &arg)
    {
        std::string result;
        result = arg;
        return result;
    }
    
    template <typename Source>
    inline std::string lexical_cast_to_string(const Source &arg)
    {
        std::string result;
        result = std::to_string(arg);
        return result;
    }
    
    inline std::vector<std::string>& split(
    std::vector<std::string>& Result,
    const std::string& Input,
    std::string Pred)
    {
        std::size_t found = std::string::npos;
        std::size_t prev = 0;
        while (1)
        {
            found = std::string::npos;
            for (auto p : Pred)
            {
                found = std::min(Input.find(p, prev), found);
            }                
            if (found != std::string::npos)
            {
                std::string s(Input.begin() + prev, Input.begin() + found);
                Result.push_back(s);
                prev = found + 1;
            }
            else
                break;
        }
        DEBUG_LOG_PO("[split] loop done\n");
        std::string f(Input.begin() + prev, Input.end());
        Result.push_back(f);
        DEBUG_LOG_PO("[split] done\n");
        return Result;        
    }
    
    // Regular expression for token delimiters (whitespace and commas)
    static const std::regex program_options_regex{"[, \\f\\n\\r\\t\\v]+",
                                                  std::regex_constants::optimize};
    class any
    {
    public: // structors
        any()
          : content(0)
        {
        }

        any(const any & other)
          : content(other.content ? other.content->clone() : 0)
        {
        }

        ~any()
        {
            delete content;
        }

    public: // modifiers

        any & swap(any & rhs)
        {
            placeholder* tmp = content;
            content = rhs.content;
            rhs.content = tmp;
            return *this;
        }


#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
        template<typename ValueType>
        any & operator=(const ValueType & rhs)
        {
            any(rhs).swap(*this);
            return *this;
        }

        any & operator=(any rhs)
        {
            rhs.swap(*this);
            return *this;
        }

#else 
        any & operator=(const any& rhs)
        {
            any(rhs).swap(*this);
            return *this;
        }

        // move assignment
        any & operator=(any&& rhs)
        {
            rhs.swap(*this);
            any().swap(rhs);
            return *this;
        }

        // Perfect forwarding of ValueType
        template <class ValueType>
        any & operator=(ValueType&& rhs)
        {
            any(static_cast<ValueType&&>(rhs)).swap(*this);
            return *this;
        }
#endif

    public: // queries

        bool empty() const
        {
            return !content;
        }

        void clear()
        {
            any().swap(*this);
        }

    public: // types (public so any_cast can be non-friend)

        class placeholder
        {
        public: // structors

            virtual ~placeholder()
            {
            }

        public: // queries
        
            virtual placeholder * clone() const = 0;

        };

        template<typename ValueType>
        class holder
          : public placeholder
        {
        public: // structors

            holder(const ValueType & value)
              : held(value)
            {
            }
        public: // queries

            placeholder * clone()
            {
                return new holder(held);
            }

        public: // representation

            ValueType held;

        private: // intentionally left unimplemented
            holder & operator=(const holder &);
        };
    public: // representation (public so any_cast can be non-friend)
        placeholder * content;

    }; //class any
    
    class value_base
    {
    protected:
        bool m_has_actual  = false;
        bool m_has_default = false;

    public:
        virtual ~value_base() = default;

        bool has_actual() const
        {
            return m_has_actual;
        }

        bool has_default() const
        {
            return m_has_default;
        }
    };

    // Value parameters
    template <typename T>
    class value : public value_base
    {
        T  m_var; // Variable to be modified if no pointer provided
        T* m_var_ptr; // Pointer to variable to be modified

    public:
        // Constructor
        explicit value()
            : m_var_ptr(nullptr)
        {
        }

        explicit value(T var, bool defaulted)
            : m_var(var)
            , m_var_ptr(nullptr)
        {
            m_has_actual  = !defaulted;
            m_has_default = defaulted;
        }

        explicit value(T* var_ptr)
            : m_var_ptr(var_ptr)
        {
        }

        // Allows actual_value() and default_value()
        value* operator->()
        {
            return this;
        }

        // Get the value
        const T& get_value() const
        {
            if(m_var_ptr)
                return *m_var_ptr;
            else
                return m_var;
        }

        // Set actual value
        value& actual_value(T val)
        {
            if(m_var_ptr){
                *m_var_ptr = std::move(val);
            }else{
                m_var = std::move(val);
            }
            m_has_actual = true;
            return *this;
        }

        // Set default value
        value* default_value(T val, std::string const& desc)
        {
            if(!m_has_actual)
            {
                if(m_var_ptr)
                    *m_var_ptr = std::move(val);
                else
                    m_var = std::move(val);
                m_has_default = true;
            }
            return this;
        }
        
        value& default_value(T val)
        {
            if(!m_has_actual)
            {
                if(m_var_ptr)
                    *m_var_ptr = std::move(val);
                else
                    m_var = std::move(val);
                m_has_default = true;
            }
            return *this;
        }
        
    };

    // bool_switch is a value<bool>, which is handled specially
    using bool_switch = value<bool>;

    class variable_value
    {
        std::shared_ptr<value_base> m_val;

    public:
        // Constructor
        explicit variable_value() = default;

        template <typename T>
        explicit variable_value(const T& xv, bool xdefaulted)
            : m_val(std::make_shared<value<T>>(xv, xdefaulted))
        {
        }

        explicit variable_value(std::shared_ptr<value_base> val)
            : m_val(val)
        {
        }

        // Member functions
        bool empty() const
        {
            return !m_val.get() || (!m_val->has_actual() && !m_val->has_default());
        }

        bool defaulted() const
        {
            return m_val.get() && !m_val->has_actual() && m_val->has_default();
        }

        template <typename T>
        const T& as() const
        {
            if(value<T>* val = dynamic_cast<value<T>*>(m_val.get()))
                return val->get_value();
            else
                throw std::logic_error("Internal error: Invalid cast");
        }
        
        template <typename T>
        const void set(T& in) const
        {
            if(value<T>* val = dynamic_cast<value<T>*>(m_val.get()))
            {
                val->actual_value(in);
                return;
            }
            else
                throw std::logic_error("Internal error: Invalid cast");
        }
        
    };

    using variables_map = std::map<std::string, variable_value>;

    class options_description
    {
        // desc_option describes a particular option
        class desc_option
        {
            std::string                                  m_opts;
            std::shared_ptr<value_base>                  m_val;
            std::string                                  m_desc;
        public:
            // Constructor with options, value and description
            template <typename T>
            desc_option(std::string opts, value<T>* val, std::string desc)
                : m_opts(std::move(opts))
                , m_val(new auto(std::move(*val)))
                , m_desc(std::move(desc))
            {
            }
            
            template <typename T>
            desc_option(std::string opts, value<T> val, std::string desc)
                : m_opts(std::move(opts))
                , m_val(new auto(std::move(val)))
                , m_desc(std::move(desc))
            {
            }

            // Constructor with options and description
            desc_option(std::string opts, std::string desc)
                : m_opts(std::move(opts))
                , m_val(nullptr)
                , m_desc(std::move(desc))
            {
            }

            // Copy constructor is deleted
            desc_option(const desc_option&) = delete;

            // Move constructor
            desc_option(desc_option&& other) = default;

            // Accessors
            const std::string& get_opts() const
            {
                return m_opts;
            }

            const std::shared_ptr<value_base> get_val() const
            {
                return m_val;
            }

            const std::string& get_desc() const
            {
                return m_desc;
            }

            // Set a value
            void set_val(int& argc, char**& argv, std::string inopt, std::unordered_map<std::string,std::string>* p_uncfg) const
            {
                // We test all supported types with dynamic_cast and parse accordingly
                DEBUG_LOG_PO("[set_val] start\n");
                bool match = false;
                if(inopt.compare(0, 7, "--init-") == 0)
                {
                    DEBUG_LOG_PO("[set_val] skip DEBUG_LOG_PO\n");
                    if(p_uncfg != nullptr)
                        (*p_uncfg)[std::string(inopt.begin() + 2, inopt.end())] = std::string(*argv);
                    match = true;
                }
                else if(inopt.compare(0, 14, "--bounds-check") == 0)
                {
                    DEBUG_LOG_PO("[set_val] skip BoundsCheckMode\n");
                    if(p_uncfg != nullptr)
                        (*p_uncfg)[std::string(inopt.begin() + 2, inopt.end())] = std::string(*argv);
                    match = true;
                }
                else if(inopt == "--problem-size" || 
                        inopt == "--a-strides" ||
                        inopt == "--b-strides" ||
                        inopt == "--c-strides" ||
                        inopt == "--d-strides" ||
                        inopt == "--convolution-problem" ||
                        inopt == "--a-zero-pads" ||
                        inopt == "--b-zero-pads")
                {
                    DEBUG_LOG_PO("[set_val] problem size/a,b,c,d stride\n");
                    if(auto* ptr = dynamic_cast<value<std::vector<std::vector<size_t>>>*>(m_val.get()))
                    {
                        std::string in(*argv);
                        std::vector<std::string> values;
                        std::vector<size_t> values_int;
                        std::string spliter(",");
                        if(inopt == "--a-zero-pads" || inopt == "--b-zero-pads")
                            spliter = ",;";
                        roc::split(values, in, spliter);
                        size_t tuples = values.size();
                        auto vals = ptr->get_value();
                        for(auto v : values)
                        {
                            values_int.push_back(lexical_cast<size_t>(v));
                            if(values_int.size() == tuples)
                            {
                                vals.push_back(values_int);
                                values_int.clear();
                            }
                        }
                        ptr->actual_value(vals);
                        match = true;
                        if(!values_int.empty())
                            match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<TensorOp>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] TensorOp\n");
                    std::string in(*argv);
                    match = true;
                    if(in == "None")
                    {
                        ptr->actual_value(TensorOp(TensorOp::Type::None));
                    }
                    else if(in == "ComplexConjugate")
                    {
                        ptr->actual_value(TensorOp(TensorOp::Type::ComplexConjugate));
                    }
                    else
                    {
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<Client::LogLevel>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] DataType\n");
                    std::string in(*argv);
                    match = true;
                    if(in == "Error")
                    {
                        ptr->actual_value(Client::LogLevel::Error);
                    }
                    else if(in == "Terse")
                    {
                        ptr->actual_value(Client::LogLevel::Terse);
                    }
                    else if(in == "Normal")
                    {
                        ptr->actual_value(Client::LogLevel::Normal);
                    }
                    else if(in == "Verbose")
                    {
                        ptr->actual_value(Client::LogLevel::Verbose);
                    }
                    else if(in == "Debug")
                    {
                        ptr->actual_value(Client::LogLevel::Debug);
                    }
                    else
                    {
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<DataType>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] DataType\n");
                    std::string in(*argv);
                    match = true;
                    if(in == "Float")
                    {
                        ptr->actual_value(DataType::Float);
                    }
                    else if(in == "Double")
                    {
                        ptr->actual_value(DataType::Double);
                    }
                    else if(in == "ComplexFloat")
                    {
                        ptr->actual_value(DataType::ComplexFloat);
                    }
                    else if(in == "ComplexDouble")
                    {
                        ptr->actual_value(DataType::ComplexDouble);
                    }
                    else if(in == "Half")
                    {
                        ptr->actual_value(DataType::Half);
                    }
                    else if(in == "Int8x4")
                    {
                        ptr->actual_value(DataType::Int8x4);
                    }
                    else if(in == "Int32")
                    {
                        ptr->actual_value(DataType::Int32);
                    }
                    else if(in == "BFloat16")
                    {
                        ptr->actual_value(DataType::BFloat16);
                    }
                    else if(in == "Int8")
                    {
                        ptr->actual_value(DataType::Int8);
                    }
                    else
                    {
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<PerformanceMetric>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] PerformanceMetric\n");
                    std::string in(*argv);
                    match = true;
                    if(in == "DeviceEfficiency")
                    {
                        ptr->actual_value(PerformanceMetric::DeviceEfficiency);
                    }
                    else if(in == "CUEfficiency")
                    {
                        ptr->actual_value(PerformanceMetric::CUEfficiency);
                    }
                    else if(in == "Auto")
                    {
                        ptr->actual_value(PerformanceMetric::Auto);
                    }
                    else
                    {
                        DEBUG_LOG_PO("[set_val] Unsupported PerformanceMetric type\n");
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<int32_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] int32_t\n");
                    int32_t val;
                    match = argc && sscanf(*argv, "%" SCNd32, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<uint32_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] uint32_t\n");
                    uint32_t val;
                    match = argc && sscanf(*argv, "%" SCNu32, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<int64_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] int64_t\n");
                    int64_t val;
                    match = argc && sscanf(*argv, "%" SCNd64, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<uint64_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] uint64_t\n");
                    uint64_t val;
                    match = argc && sscanf(*argv, "%" SCNu64, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<float>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] float\n");
                    float val;
                    match = argc && sscanf(*argv, "%f", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<double>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] double\n");
                    double val;
                    match = argc && sscanf(*argv, "%lf", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<char>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] char\n");
                    char val;
                    match = argc && sscanf(*argv, " %c", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<int8_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] int8_t\n");
                    int8_t val;
                    match = argc && sscanf(*argv, " %c", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<bool>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] bool\n");
                    std::string in(*argv);
                    match = true;
                    if(in == "True" || in == "true" || in == "1")
                        ptr->actual_value(true);
                    else if(in == "False" || in == "false" || in == "0")
                        ptr->actual_value(false);
                    else
                        match = false;
                }
                else if(auto* ptr = dynamic_cast<value<std::string>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] string\n");
                    if(argc)
                    {
                        ptr->actual_value(*argv);
                        match = true;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<std::vector<std::string>>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] vector<string>\n");
                    if(argc)
                    {
                        std::string in(*argv);
                        std::vector<std::string> values;
                        std::string spliter(",;");
                        roc::split(values, in, spliter);
                        ptr->actual_value(values);
                        match = true;
                    }
                }
                else
                {
                    DEBUG_LOG_PO("[set_val] Internal error: Unsupported data type (setting value)\n");
                    throw std::logic_error("Internal error: Unsupported data type (setting value)");
                }
                if(!match)
                    throw std::invalid_argument(argc ? "Invalid value for " + inopt
                                                     : "Missing required value for " + inopt);

                // Skip past the argument's value
                ++argv;
                --argc;
            }
            // Set a value
            void set_val(int& argc, const char* argv, std::string inopt, std::unordered_map<std::string,std::string>* p_uncfg) const
            {
                // We test all supported types with dynamic_cast and parse accordingly
                DEBUG_LOG_PO("[set_val] start\n");
                bool match = false;
                if(inopt.compare(0, 7, "--init-") == 0)
                {
                    DEBUG_LOG_PO("[set_val] skip DEBUG_LOG_PO\n");
                    if(p_uncfg != nullptr)
                        (*p_uncfg)[std::string(inopt.begin() + 2, inopt.end())] = std::string(argv);
                    match = true;
                }
                else if(inopt.compare(0, 14, "--bounds-check") == 0)
                {
                    DEBUG_LOG_PO("[set_val] skip BoundsCheckMode\n");
                    if(p_uncfg != nullptr)
                        (*p_uncfg)[std::string(inopt.begin() + 2, inopt.end())] = std::string(argv);
                    match = true;
                }
                else if(inopt == "--problem-size" || 
                        inopt == "--a-strides" ||
                        inopt == "--b-strides" ||
                        inopt == "--c-strides" ||
                        inopt == "--d-strides" ||
                        inopt == "--convolution-problem" ||
                        inopt == "--a-zero-pads" ||
                        inopt == "--b-zero-pads")
                {
                    DEBUG_LOG_PO("[set_val] problem size/a,b,c,d stride\n");
                    if(auto* ptr = dynamic_cast<value<std::vector<std::vector<size_t>>>*>(m_val.get()))
                    {
                        std::string in(argv);
                        std::vector<std::string> values;
                        std::vector<size_t> values_int;
                        std::string spliter(",");
                        if(inopt == "--a-zero-pads" || inopt == "--b-zero-pads")
                            spliter = ",;";
                        roc::split(values, in, spliter);
                        size_t tuples = values.size();
                        auto vals = ptr->get_value();
                        for(auto v : values)
                        {
                            values_int.push_back(lexical_cast<size_t>(v));
                            if(values_int.size() == tuples)
                            {
                                vals.push_back(values_int);
                                values_int.clear();
                            }
                        }
                        ptr->actual_value(vals);
                        match = true;
                        if(!values_int.empty())
                            match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<TensorOp>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] TensorOp\n");
                    std::string in(argv);
                    match = true;
                    if(in == "None")
                    {
                        ptr->actual_value(TensorOp(TensorOp::Type::None));
                    }
                    else if(in == "ComplexConjugate")
                    {
                        ptr->actual_value(TensorOp(TensorOp::Type::ComplexConjugate));
                    }
                    else
                    {
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<Client::LogLevel>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] DataType\n");
                    std::string in(argv);
                    match = true;
                    if(in == "Error")
                    {
                        ptr->actual_value(Client::LogLevel::Error);
                    }
                    else if(in == "Terse")
                    {
                        ptr->actual_value(Client::LogLevel::Terse);
                    }
                    else if(in == "Normal")
                    {
                        ptr->actual_value(Client::LogLevel::Normal);
                    }
                    else if(in == "Verbose")
                    {
                        ptr->actual_value(Client::LogLevel::Verbose);
                    }
                    else if(in == "Debug")
                    {
                        ptr->actual_value(Client::LogLevel::Debug);
                    }
                    else
                    {
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<DataType>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] DataType\n");
                    std::string in(argv);
                    match = true;
                    if(in == "Float")
                    {
                        ptr->actual_value(DataType::Float);
                    }
                    else if(in == "Double")
                    {
                        ptr->actual_value(DataType::Double);
                    }
                    else if(in == "ComplexFloat")
                    {
                        ptr->actual_value(DataType::ComplexFloat);
                    }
                    else if(in == "ComplexDouble")
                    {
                        ptr->actual_value(DataType::ComplexDouble);
                    }
                    else if(in == "Half")
                    {
                        ptr->actual_value(DataType::Half);
                    }
                    else if(in == "Int8x4")
                    {
                        ptr->actual_value(DataType::Int8x4);
                    }
                    else if(in == "Int32")
                    {
                        ptr->actual_value(DataType::Int32);
                    }
                    else if(in == "BFloat16")
                    {
                        ptr->actual_value(DataType::BFloat16);
                    }
                    else if(in == "Int8")
                    {
                        ptr->actual_value(DataType::Int8);
                    }
                    else
                    {
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<PerformanceMetric>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] PerformanceMetric\n");
                    std::string in(argv);
                    match = true;
                    if(in == "DeviceEfficiency")
                    {
                        ptr->actual_value(PerformanceMetric::DeviceEfficiency);
                    }
                    else if(in == "CUEfficiency")
                    {
                        ptr->actual_value(PerformanceMetric::CUEfficiency);
                    }
                    else if(in == "Auto")
                    {
                        ptr->actual_value(PerformanceMetric::Auto);
                    }
                    else
                    {
                        DEBUG_LOG_PO("[set_val] Unsupported PerformanceMetric type\n");
                        match = false;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<int32_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] int32_t\n");
                    int32_t val;
                    match = argc && sscanf(argv, "%" SCNd32, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<uint32_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] uint32_t\n");
                    uint32_t val;
                    match = argc && sscanf(argv, "%" SCNu32, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<int64_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] int64_t\n");
                    int64_t val;
                    match = argc && sscanf(argv, "%" SCNd64, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<uint64_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] uint64_t\n");
                    uint64_t val;
                    match = argc && sscanf(argv, "%" SCNu64, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<float>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] float\n");
                    float val;
                    match = argc && sscanf(argv, "%f", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<double>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] double\n");
                    double val;
                    match = argc && sscanf(argv, "%lf", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<char>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] char\n");
                    char val;
                    match = argc && sscanf(argv, " %c", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<int8_t>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] int8_t\n");
                    int8_t val;
                    match = argc && sscanf(argv, " %c", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<bool>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] bool\n");
                    std::string in(argv);
                    match = true;
                    if(in == "True" || in == "true" || in == "1")
                        ptr->actual_value(true);
                    else if(in == "False" || in == "false" || in == "0")
                        ptr->actual_value(false);
                    else
                        match = false;
                }
                else if(auto* ptr = dynamic_cast<value<std::string>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] string\n");
                    if(argc)
                    {
                        ptr->actual_value(argv);
                        match = true;
                    }
                }
                else if(auto* ptr = dynamic_cast<value<std::vector<std::string>>*>(m_val.get()))
                {
                    DEBUG_LOG_PO("[set_val] vector<string>\n");
                    if(argc)
                    {
                        std::string in(argv);
                        std::vector<std::string> values;
                        std::string spliter(",;");
                        roc::split(values, in, spliter);
                        ptr->actual_value(values);
                        match = true;
                    }
                }
                else
                {
                    DEBUG_LOG_PO("[set_val] Internal error: Unsupported data type (setting value)\n");
                    throw std::logic_error("Internal error: Unsupported data type (setting value)");
                }
                if(!match)
                    throw std::invalid_argument(argc ? "Invalid value for " + inopt
                                                     : "Missing required value for " + inopt);

                // Skip past the argument's value
                --argc;
            }
            
        };

        // Description and option list
        std::string              m_desc;
        std::vector<desc_option> m_optlist;

        // desc_optionlist allows chains of options to be parenthesized
        class desc_optionlist
        {
            std::vector<desc_option>& m_list;

        public:
            explicit desc_optionlist(std::vector<desc_option>& list)
                : m_list(list)
            {
            }

            template <typename... Ts>
            desc_optionlist operator()(Ts&&... arg)
            {
                m_list.push_back(desc_option(std::forward<Ts>(arg)...));
                return *this;
            }
        };

        // Parse an option at the current (argc, argv) position
        void parse_option(int& argc, std::vector<std::string>& argv, variables_map& vm, bool ignoreUnknown,
                          std::unordered_map<std::string,std::string>* p_uncfg) const
        {
            DEBUG_LOG_PO("[parse_option]\n");
            // Iterate across all options
            for(const auto& opt : m_optlist)
            {
                // Canonical name used for map
                std::string canonical_name;

                // Iterate across tokens in the opts
                DEBUG_LOG_PO("[parse_option] Iterate across tokens in the opts\n");
                for(std::sregex_token_iterator tok{
                        opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};
                    tok != std::sregex_token_iterator();
                    ++tok)
                {
                    // The first option in a list of options is the canonical name
                    if(!canonical_name.length())
                        canonical_name = tok->str();
                       
                    // If the length of the option is 1, it is single-dash; otherwise double-dash
                    const char* prefix = tok->length() == 1 ? "-" : "--";
                    
                    // If option matches
                    DEBUG_LOG_PO("[parse_option] Check if option matches\n");
                    if(argv[argv.size() - argc] == prefix + tok->str())
                    {
                        
                        //++argv;
                        --argc;

                        // If option has a value, set it
                        auto got = opt.get_val().get();
                        if(got){
                            opt.set_val(argc, argv[argv.size() - argc].c_str(), prefix + tok->str(), p_uncfg);
                        }

                        // Add seen options to map
                        vm[canonical_name] = variable_value(opt.get_val());
                        DEBUG_LOG_PO("[parse_option] Add seen options to map successfully\n");
                        return; // Return successfully
                    }
                    DEBUG_LOG_PO("[parse_option] Check if option matches done\n");
                }
            }

            // No options were matched
            if(ignoreUnknown)
            {
                --argc;
            }
            else
            {
                DEBUG_LOG_PO("[parse_option] ERROR: option not defined\n");
                throw std::invalid_argument("Option " + std::string(argv[0]) + " is not defined.");
            }
        }
        
        // Parse an option at the current (argc, argv) position
        void parse_option(int& argc, char**& argv, variables_map& vm, bool ignoreUnknown,
                          std::unordered_map<std::string,std::string>* p_uncfg) const
        {
            DEBUG_LOG_PO("[parse_option]\n");
            // Iterate across all options
            for(const auto& opt : m_optlist)
            {
                // Canonical name used for map
                std::string canonical_name;

                // Iterate across tokens in the opts
                DEBUG_LOG_PO("[parse_option] Iterate across tokens in the opts\n");
                for(std::sregex_token_iterator tok{
                        opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};
                    tok != std::sregex_token_iterator();
                    ++tok)
                {
                    // The first option in a list of options is the canonical name
                    if(!canonical_name.length())
                        canonical_name = tok->str();
                      
                    // If the length of the option is 1, it is single-dash; otherwise double-dash
                    const char* prefix = tok->length() == 1 ? "-" : "--";
                    
                    // If option matches
                    DEBUG_LOG_PO("[parse_option] Check if option matches\n");
                    if(*argv == prefix + tok->str())
                    {
                        
                        ++argv;
                        --argc;

                        // If option has a value, set it
                        auto got = opt.get_val().get();
                        if(got){
                            opt.set_val(argc, argv, prefix + tok->str(), p_uncfg);
                        }

                        // Add seen options to map
                        vm[canonical_name] = variable_value(opt.get_val());
                        DEBUG_LOG_PO("[parse_option] Add seen options to map successfully\n");
                        return; // Return successfully
                    }
                    DEBUG_LOG_PO("[parse_option] Check if option matches done\n");
                }
            }

            // No options were matched
            if(ignoreUnknown)
            {
                ++argv;
                --argc;
            }
            else
            {
                DEBUG_LOG_PO("[parse_option] ERROR: option not defined\n");
                throw std::invalid_argument("Option " + std::string(argv[0]) + " is not defined.");
            }
        }

    public:
        // Constructor
        explicit options_description(std::string desc)
            : m_desc(std::move(desc))
        {
        }

        // Start a desc_optionlist chain
        desc_optionlist add_options() &
        {
            return desc_optionlist(m_optlist);
        }

        // Parse all options
        void parse_options(int&                                         argc,
                           char**&                                      argv,
                           variables_map&                               vm,
                           bool                                         ignoreUnknown = false,
                           std::unordered_map<std::string,std::string>* p_uncfg = nullptr) const
        {
            // Add options with default values to map
            for(const auto& opt : m_optlist)
            {
                std::sregex_token_iterator tok{
                    opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};

                // Canonical name used for map
                std::string canonical_name = tok->str();

                if(opt.get_val().get() && opt.get_val()->has_default())
                {   
                    //DEBUG_LOG_PO("[parse_options] add an option\n");
                    vm[canonical_name] = variable_value(opt.get_val());
                }
            }
            DEBUG_LOG_PO("[parse_options] add options done\n");
            // Parse options
            while(argc)
                parse_option(argc, argv, vm, ignoreUnknown, p_uncfg);

        }

        // Parse all options
        void parse_options(int&                                         argc,
                           std::vector<std::string>&                    argv,
                           variables_map&                               vm,
                           bool                                         ignoreUnknown = false,
                           std::unordered_map<std::string,std::string>* p_uncfg = nullptr) const
        {
            // Add options with default values to map
            for(const auto& opt : m_optlist)
            {
                std::sregex_token_iterator tok{
                    opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};

                // Canonical name used for map
                std::string canonical_name = tok->str();

                if(opt.get_val().get() && opt.get_val()->has_default())
                {   
                    //DEBUG_LOG_PO("[parse_options] add an option\n");
                    vm[canonical_name] = variable_value(opt.get_val());
                }
            }
            DEBUG_LOG_PO("[parse_options] add options done\n");
            // Parse options
            while(argc)
                parse_option(argc, argv, vm, ignoreUnknown, p_uncfg);

        }

        // Formatted output of command-line arguments description
        friend std::ostream& operator<<(std::ostream& os, const options_description& d)
        {
            // Iterate across all options
            for(const auto& opt : d.m_optlist)
            {
                bool               first = true;
                const char*        delim = "";
                std::ostringstream left;
                // Iterate across tokens in the opts
                for(std::sregex_token_iterator tok{opt.get_opts().begin(),
                                                   opt.get_opts().end(),
                                                   program_options_regex,
                                                   -1};
                    tok != std::sregex_token_iterator();
                    ++tok, first = false, delim = " ")
                {
                    // If the length of the option is 1, it is single-dash; otherwise double-dash
                    const char* prefix = tok->length() == 1 ? "-" : "--";
                    left << delim << (first ? "" : "|") << prefix << tok->str();
                }  

                os << std::setw(30) << std::left << left.str() << " " << opt.get_desc() << " ";
                left.str(std::string());
                
                // Print the default value of the variable type if it exists
                // We do not print the default value for bool
                const value_base* val = opt.get_val().get();
                if(val && !dynamic_cast<const value<bool>*>(val))
                {
                    if(val->has_default())
                    { 
                        // We test all supported types with dynamic_cast and print accordingly
                        left << " (Default value is: ";
                        if(dynamic_cast<const value<int32_t>*>(val))
                            left << dynamic_cast<const value<int32_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<uint32_t>*>(val))
                            left << dynamic_cast<const value<uint32_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<int64_t>*>(val))
                            left << dynamic_cast<const value<int64_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<uint64_t>*>(val))
                            left << dynamic_cast<const value<uint64_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<float>*>(val))
                            left << dynamic_cast<const value<float>*>(val)->get_value();
                        else if(dynamic_cast<const value<double>*>(val))
                            left << dynamic_cast<const value<double>*>(val)->get_value();
                        else if(dynamic_cast<const value<char>*>(val))
                            left << dynamic_cast<const value<char>*>(val)->get_value();
                        else if(dynamic_cast<const value<int8_t>*>(val))
                            left << dynamic_cast<const value<int8_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<std::string>*>(val))
                            left << dynamic_cast<const value<std::string>*>(val)->get_value();
                        else if(dynamic_cast<const value<Tensile::PerformanceMetric>*>(val))
                            left << dynamic_cast<const value<Tensile::PerformanceMetric>*>(val)->get_value();
                        else
                            left << "Internal error: Unsupported data type (printing value)";
                        left << ")";
                    }
                }
                os << left.str() << "\n\n";
            } 
            return os << std::flush;
        }
    };

    // Class representing command line parser
    class parse_command_line
    {
        variables_map m_vm;
        std::unordered_map<std::string,std::string> m_unconfig;
    public:
        parse_command_line(int                        argc,
                           const char**               argv,
                           const options_description& desc,
                           bool                       ignoreUnknown = false)
        {
            char** argv_tmp = (char**)argv;
            ++argv_tmp; // Skip argv[0]
            --argc;
            desc.parse_options(argc, argv_tmp, m_vm, ignoreUnknown);
        }
        
        parse_command_line(int                        argc,
                           char**                     argv,
                           const options_description& desc,
                           bool                       ignoreUnknown = false)
        {
            ++argv; // Skip argv[0]
            --argc;
            desc.parse_options(argc, argv, m_vm, ignoreUnknown);
        }

        // Copy the variables_map
        friend inline void store(const parse_command_line& p, variables_map& vm);
        friend inline void store(const parse_command_line&& p, variables_map& vm);

    };
    
    inline void store(const parse_command_line& p, variables_map& vm)
    {
        vm = p.m_vm;
    }
    
    inline void store(const parse_command_line&& p, variables_map& vm)
    {
        vm = std::move(p.m_vm);
    }
    
    class parse_config_file
    {
        variables_map m_vm;
        std::unordered_map<std::string,std::string> m_unconfig;
    public:
        parse_config_file(std::ifstream&             file,
                          const options_description& desc,
                          bool                       ignoreUnknown = false)
        {
            DEBUG_LOG_PO("[parse_config_file] start\n");
            //file to argc and argv
            m_unconfig.clear();
            std::string line;
            std::vector<std::string> vecstr;
            char **argv;
            int argc = 0;
            char str[256];

            while(std::getline(file, line))
            {
                std::size_t found = line.find('=');
                if (found != std::string::npos)
                {
                    std::string front(line.begin(), line.begin() + found);
                    std::string back(line.begin() + found + 1, line.end());
                    if (back.length())
                    {
                        vecstr.push_back("--" + front);
                        argc++;
                        vecstr.push_back(back);
                        argc++;
                    }
                }
                else
                {
                    // Skip this line
                }
            }
            argv = new char*[argc + 1]; // Skip [0]
            for (int i = 0; i < argc; i++)
            {
                size_t len = vecstr[i].length();
                argv[i + 1] = new char[len + 1];
                for(int j = 0; j < len ; j++)
                {
                    argv[i + 1][j] = vecstr[i][j];
                }
                argv[i + 1][len]='\0';
            }
            
            DEBUG_LOG_PO("[parse_config_file] parse options\n");
            desc.parse_options(argc, vecstr, m_vm, ignoreUnknown, &m_unconfig);
            DEBUG_LOG_PO("[parse_config_file] parse options done\n");
            
            for (int i = 0; i < argc; i++)
            {
                delete[] argv[i + 1];
            }
            delete[] argv;
            
            DEBUG_LOG_PO("[parse_config_file] done\n");
        }
        
       // Copy the variables_map
        friend inline void store(const parse_config_file& p, variables_map& vm, std::unordered_map<std::string,std::string>* p_uncfg);
        friend inline void store(const parse_config_file&& p, variables_map& vm, std::unordered_map<std::string,std::string>* p_uncfg);
    };
    
    inline void store(const parse_config_file& p, variables_map& vm, std::unordered_map<std::string,std::string>* p_uncfg)
    {
        vm = p.m_vm;
        if(p_uncfg != nullptr)
            *p_uncfg = p.m_unconfig;
    }
    
    inline void store(const parse_config_file&& p, variables_map& vm, std::unordered_map<std::string,std::string>* p_uncfg)
    {
        vm = std::move(p.m_vm);
        if(p_uncfg != nullptr)
            *p_uncfg = p.m_unconfig;
    }

    // We can define the notify() function as a no-op for our purposes
    inline void notify(const variables_map&) {}
    namespace algorithm {
        
        inline std::string 
        is_any_of( const std::string& Set )
        {
            std::string result = Set;
            return result; 
        }

    }; // class algorithm

}

#endif