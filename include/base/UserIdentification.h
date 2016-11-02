#ifndef BASE__useridentification
#define BASE__useridentification

#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

// ---------------------------------------------------------------
//		definitions
// ---------------------------------------------------------------


#define NR_USERS 6


// ---------------------------------------------------------------
// Reference: http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
/*
Usage:
std::cout << type_name<decltype(foo_value())>() << '\n';
*/
// ---------------------------------------------------------------
template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

// ---------------------------------------------------------------
//		debugging
// ---------------------------------------------------------------



// ---------------------------------------------------------------
//		pointer handling
// ---------------------------------------------------------------

template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != nullptr) {
		pInterfaceToRelease->Release();
		pInterfaceToRelease = nullptr;
	}
}

#endif