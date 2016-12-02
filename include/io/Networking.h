#ifndef IO_NETWORKING_H_
#define IO_NETWORKING_H_

#include <cstdint>
#include <vector>
#include <iostream>
#include <winsock.h>

#define _DEBUG_NETWORKING

// define macro ourselfes - dont use winsock2!
#define _WS2_32_WINSOCK_SWAP_LONGLONG(l)            \
            ( ( ((l) >> 56) & 0x00000000000000FFLL ) |       \
              ( ((l) >> 40) & 0x000000000000FF00LL ) |       \
              ( ((l) >> 24) & 0x0000000000FF0000LL ) |       \
              ( ((l) >>  8) & 0x00000000FF000000LL ) |       \
              ( ((l) <<  8) & 0x000000FF00000000LL ) |       \
              ( ((l) << 24) & 0x0000FF0000000000LL ) |       \
              ( ((l) << 40) & 0x00FF000000000000LL ) |       \
              ( ((l) << 56) & 0xFF00000000000000LL ) )


namespace cv {
	class Mat;
}

namespace io
{
	class TCPClient
	{
	public:
		TCPClient();
		~TCPClient();
		void Config(char* host_name = "127.0.0.1", int host_port = 80);
		bool Connect(char* host_name, int host_port);
		bool Connect();
		void Close();

		/// <summary>
		/// Sends a RGB image. The RGB color system uses 3 channels (Red, Green, Blue) 8 bits = 1 byte each, which have (unsigned) integer values from 0 to 255.
		/// </summary>
		/// <param name="img">The img.</param>
		/// <returns>Number of bytes sent</returns>
		int SendRGBImage(const cv::Mat &img);
		void SendRGBTestImage(int size = 100);

		// ----- send

		bool SendKeyboard();
		int SendString(std::string val);
		// range: -127 .. 127
		int SendChar(char id);
		// range:  .. 255
		int SendUChar(unsigned char val);
		// range: –32,768 .. 32,767
		int SendShort(short val);
		// range: 0 .. 65535
		int SendUShort(unsigned short ushort);
		// range: –2,147,483,648 .. 2,147,483,647
		int SendInt(int val);
		// range: 0 .. 4,294,967,295
		int SendUInt(uint32_t size);
		// range: false .. true
		int SendBool(bool val);
		// range: 3.4E +/- 38 (7 digits)
		int SendFloat(float val);
		// range: +/- 1.7e +/- 308
		// TODO: fix
		int SendDouble(double val);

		// ------ receive

		std::string ReceiveStringWithVarLength();
		int ReceiveMessage(SOCKET socket_id, char *buf, int *len);
		int ReceiveRGBImage(cv::Mat &output, int img_width);

		template<typename T>
		T TCPClient::Receive8bit()
		{
			char data_received;	// 8bit
			recv(mSocketID, (char*)&data_received, 1, 0);
			return (T)data_received;
		}

		template<typename T>
		T TCPClient::Receive16bit()
		{
			uint16_t val;	// 16bit
			recv(mSocketID, (char*)&val, 2, 0);
			T x;
			val = ntohs(val);
			memcpy(&x, &val, sizeof(T));
			return x;
		}

		template<typename T>
		T TCPClient::Receive32bit()
		{
			uint32_t val;	// 32bit
			recv(mSocketID, (char*)&val, 4, 0);
			T x;
			val = ntohl(val);
			memcpy(&x, &val, sizeof(T));
			return x;
		}

		// TODO: fix
		template<typename T>
		T TCPClient::Receive64bit()
		{
			uint64_t val;	// 64bit
			int msg_length = 8;
			ReceiveMessage(mSocketID, (char*)&val, &msg_length);
			T x;
			val = _WS2_32_WINSOCK_SWAP_LONGLONG(val);
			memcpy(&x, &val, sizeof(T));
			return x;
		}

		// ------ Floating point Endianness conversion

		uint32_t htonf(float f)
		{
			uint32_t x;
			memcpy(&x, &f, sizeof(float));
			return htonl(x);
		}

		float ntohf(uint32_t nf)
		{
			float x;
			nf = ntohl(nf);
			memcpy(&x, &nf, sizeof(float));
			return x;
		}

		uint64_t htond(double d)
		{
			uint64_t x;
			memcpy(&x, &d, sizeof(double));
			return _WS2_32_WINSOCK_SWAP_LONGLONG(x);
		}

		// ------ deprecated
		int SendImageWithLength(const cv::Mat &img);
		int SendImageBatchWithLength(const std::vector<cv::Mat> &images);

	private:
		bool OpenSocket();
		SOCKET mSocketID;
		char* mHostName;
		int mHostPort;
	};
}

#endif