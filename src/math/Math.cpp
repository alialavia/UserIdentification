
#include <math/Math.h>
#include <opencv2/core/mat.hpp>
#include <opencv2\imgproc.hpp>

using namespace math;


template<typename T>
void Array3D<T>::CopyTo(size_t x, size_t y, size_t z, T in)
{
	size_t pos = GetPos(x, y, z);
	if (mData.at(pos) != nullptr)
	{
		// free old memory
		delete(mData.at(pos));
		mData.at(pos) = nullptr;
	}
	// allocate new object
	T * obj = new T(in);

	// assign
	mData.at(pos) = obj;
}


// template specification
template<>
void Array3D<cv::Mat>::CopyTo(size_t x, size_t y, size_t z, cv::Mat in)
{

	// assign
	size_t pos = GetPos(x, y, z);
	if (mData.at(pos) != nullptr)
	{
		// free old memory
		delete(mData.at(pos));
		mData.at(pos) = nullptr;
	}

	// allocate new element
	cv::Mat * m = new cv::Mat();

	// make deep copy of input
	*m = in.clone();

	// assign
	mData.at(pos) = m;
}

template<>
void Array3D<cv::Mat>::operator=(Array3D<cv::Mat> &other)
{
	// delete release data
	Reset();
	// copy properties
	mWidth = other.mWidth;
	mHeight = other.mHeight;
	mLength = other.mLength;
	// reallocate data
	mData = std::vector<cv::Mat*>(mWidth*mHeight*mLength, nullptr);
	// make deep copy of array data
	for (size_t x = 0; x < other.Size(0); x++)
	{
		for (size_t y = 0; y < other.Size(1); y++)
		{
			for (size_t z = 0; z < other.Size(2); z++)
			{
				size_t index = GetPos(x, y, z);
				if (mData[index] != nullptr)
				{
					// matrix header
					cv::Mat * new_obj = new cv::Mat();

					// clone matrix
					*new_obj = (other(x, y, z)).clone();

					other(x, y, z);

					// save in container vector
					mData.at(index) = new_obj;
				}
			}
		}
	}
}
