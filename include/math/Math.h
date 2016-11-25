#ifndef MATH_MATH_H_
#define MATH_MATH_H_

#include <opencv2/core/mat.hpp>
#include <iostream>

namespace math {

	// ---------------------------------------------------------------
	//		math containers
	// ---------------------------------------------------------------

	template <typename T>
	class Array3D {
		size_t mWidth, mHeight, mLength;
		std::vector<T*> mData;
	public:

		size_t Size(size_t dim) const{
			size_t s = 0;
			if (dim == 0) {
				s = mWidth;
			}
			else if (dim == 1) {
				s = mHeight;
			}
			else if (dim == 2) {
				s = mLength;
			}
			else
			{
				throw std::invalid_argument("Invalid array dimension.");
			}
			return s;
		}
		Array3D(size_t x, size_t y, size_t z) :
			mWidth(x), mHeight(y), mLength(z), mData(x*y*z, nullptr)
		{}
		~Array3D()
		{
			// release memory
			for (size_t i = 0; i < mData.size(); i++)
			{
				if (mData[i] != nullptr)
				{
					delete(mData[i]);
				}
			}
		}

		void Reset()
		{
			for (size_t i = 0; i < mData.size(); i++)
			{
				if (mData[i] != nullptr)
				{
					delete(mData[i]);
					mData[i] = nullptr;
				}
			}
		}
		bool IsFree(size_t x, size_t y, size_t z) const
		{
			return (mData.at(GetPos(x, y, z)) == nullptr);
		}

		// assign pointer
		void Assign(size_t x, size_t y, size_t z, T* in)
		{
			size_t pos = GetPos(x, y, z);
			if (mData.at(pos) != nullptr)
			{
				// free old memory
				delete(mData.at(pos));
			}
			// assign
			mData.at(pos) = in;
		}

		// TODO: is this necessary? already got assignment operator
		// copy
		void CopyTo(size_t x, size_t y, size_t z, T in);

		// return by reference
		T& operator()(size_t x, size_t y, size_t z) {
			return *(mData.at(GetPos(x, y, z)));
		}
		// return by value
		T operator()(size_t x, size_t y, size_t z) const {
			return *(mData.at(GetPos(x, y, z)));
		}

		// assignment operator - make copy
		void operator = (Array3D<T> &other) {
			// delete release data
			Reset();
			// copy properties
			mWidth = other.mWidth;
			mHeight = other.mHeight;
			mLength = other.mLength;
			// reallocate data
			mData = std::vector<T*>(mWidth*mHeight*mLength, nullptr);
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
							T* new_obj = new T();
							// copy data at memory location
							memcpy(new_obj, *other(x, y, z), sizeof(T));
							// save in container vector
							mData.at(index) = new_obj;
						}
					}
				}
			}
		}

	private:
		size_t GetPos(size_t x, size_t y, size_t z) const
		{
			size_t index = x + y * mWidth + z * mWidth * mHeight;
			if (index >= mData.size())
			{
				std::cout << "Array size overflow: index "<< index << " | array size: " << mData.size() << std::endl;
				throw std::invalid_argument("Array size overflow.");
			}

			return index;
		}
	};

} // namespace

#endif
