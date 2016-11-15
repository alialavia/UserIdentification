#ifndef IO_NETWORKING_H_
#define IO_NETWORKING_H_

#include <cstdint>
#include <windows.h>
#include <vector>

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
		bool Connect(char* host_name = "127.0.0.1", int host_port = 80);
		bool Reconnect();
		void Close();

		/// <summary>
		/// Sends a RGB image. The RGB color system uses 3 channels (Red, Green, Blue) 8 bits = 1 byte each, which have (unsigned) integer values from 0 to 255.
		/// </summary>
		/// <param name="img">The img.</param>
		/// <returns>Number of bytes sent</returns>
		int SendRGBImage(const cv::Mat &img);
		void SendRGBTestImage(int size = 100);

		// ----- send

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

		// ------ receive

		template<typename T>
		T TCPClient::Receive8bit()
		{
			char data_received;	// 8bit
			long rc;
			rc = recv(mSocketID, (char*)&data_received, 1, 0);
			std::cout << "--- " << rc << "bytes\n";
			return (T)data_received;
		}

		template<typename T>
		T TCPClient::Receive16bit()
		{
			u_short val;	// 16bit
			long rc;
			rc = recv(mSocketID, (char*)&val, 2, 0);
			std::cout << "--- " << rc << "bytes\n";
			return (T)ntohs(val);
		}

		template<typename T>
		T TCPClient::Receive32bit()
		{
			u_long val;	// 32bit
			long rc;
			rc = recv(mSocketID, (char*)&val, 4, 0);
			std::cout << "--- " << rc << "bytes\n";
			return (T)ntohl(val);
		}

		// TODO: fix - not working atm
		template<typename T>
		T TCPClient::Receive64bit()
		{
			unsigned long long val;	// 64bit
			long rc;
			rc = recv(mSocketID, (char*)&val, 8, 0);
			std::cout << "--- " << rc << "bytes\n";
			return (T)ntohd(val);
		}



		bool SendKeyboard();

		// receive
		int ReceiveMessage(int socket_id, char *buf, int *len);
		int ReceiveRGBImage(cv::Mat &output, int img_width);




		// ------ deprecated
		int SendImageWithLength(const cv::Mat &img);
		int SendImageBatchWithLength(const std::vector<cv::Mat> &images);

	private:
		bool OpenSocket();
		int mSocketID;
		char* mHostName;
		int mHostPort;
	};
}

#endif