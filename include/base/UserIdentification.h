#ifndef BASE_USER_IDENTIFICATION_H_
#define BASE_USER_IDENTIFICATION_H_

#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

// ---------------------------------------------------------------
//		constants
// ---------------------------------------------------------------

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------
//		definitions - fixed
// ---------------------------------------------------------------

#define NR_USERS 6

// joint indices
namespace base
{
	enum JointType {
		JointType_SpineBase = 1 << 0,
		JointType_SpineMid = 1 << 1,
		JointType_Neck = 1 << 2,
		JointType_Head = 1 << 3,
		JointType_ShoulderLeft = 1 << 4,
		JointType_ElbowLeft = 1 << 5,
		JointType_WristLeft = 1 << 6,
		JointType_HandLeft = 1 << 7,
		JointType_ShoulderRight = 1 << 8,
		JointType_ElbowRight = 1 << 9,
		JointType_WristRight = 1 << 10,
		JointType_HandRight = 1 << 11,
		JointType_HipLeft = 1 << 12,
		JointType_KneeLeft = 1 << 13,
		JointType_AnkleLeft = 1 << 14,
		JointType_FootLeft = 1 << 15,
		JointType_HipRight = 1 << 16,
		JointType_KneeRight = 1 << 17,
		JointType_AnkleRight = 1 << 18,
		JointType_FootRight = 1 << 19,
		JointType_SpineShoulder = 1 << 20,
		JointType_HandTipLeft = 1 << 21,
		JointType_ThumbLeft = 1 << 22,
		JointType_HandTipRight = 1 << 23,
		JointType_ThumbRight = 1 << 24,
		JointType_Count = 25
	};

	enum StreamType
	{
		StreamType_Depth = 1 << 0,
		StreamType_Color = 1 << 1,
		StreamType_Infrared = 1 << 2,
		StreamType_Body = 1 << 3,
		StreamType_BodyIndex = 1 << 4
	};

	enum ImageSpace
	{
		ImageSpace_Depth = 1 << 0,
		ImageSpace_Color = 1 << 1,
		ImageSpace_Infrared = 1 << 2,
		ImageSpace_BodyIndex = 1 << 3
	};

	enum StreamSize
	{
		StreamSize_WidthColor = 1920,
		StreamSize_HeightColor = 1080,
		StreamSize_WidthDepth = 512,
		StreamSize_HeightDepth = 424,
		StreamSize_WidthBodyIndex = StreamSize_WidthDepth,
		StreamSize_HeightBodyIndex = StreamSize_HeightDepth
	};
}


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