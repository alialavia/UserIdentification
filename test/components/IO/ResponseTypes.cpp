#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/cvdef.h>


#include <typeindex>
#include <typeinfo>
#include <map>


class Response;
class Request;
class Response1;
class Response;


// ------------------------ RESPONSES

class Response {
	// unique response type identifier
	int const cResponseType;
public:
	int GetResponseType()
	{
		return cResponseType;
	}
	virtual void Load() = 0;
protected:
	Response(int typeID) : cResponseType(typeID) { }
};


// response types
class Response1 : public Response
{
public:
	Response1() :Response(1) {}
	int mMyData1 = 0;
	void Load()
	{
		// receive response specific data from server
		mMyData1 = 1337;
	}
};

class Response2 : public Response
{
public:
	Response2() :Response(2) {}
	int mMyData2 = 0;
	void Load()
	{

	}
};

// response lookup
class ResponseFactory
{
public:
	void* Allocate(int type_id)
	{
		if(type_id == 1)
		{
			return new Response1();
		}
	}

};


// ------------------------ REQUESTS

class Request
{
public: 
	Request():pR(nullptr)
	{
		
	}

	~Request()
	{
		if(pR)
		{
			delete(pR);
			pR = nullptr;
		}
	}

	// return response pointer
	template<class ResponseType>
	void CopyResponse(ResponseType* out)
	{
		// link outer pointer to response
		//*out = *pR;

		// copy data
		memcpy(out, pR, sizeof(ResponseType));
	}

	void AcquireResponse(int simulated_response_id)
	{
		// get response id (from server)

		// generate response container depending on response id received from server
		if (simulated_response_id == 1)
		{
			// arrived response
			Response1* custom_response = new Response1();

			// Load response from server
			custom_response->Load();

			// store pointer
			pR = custom_response;

		}
		else if (simulated_response_id == 2)
		{
			// generate second response

		}
	}

	// -----------------------------------------
	// link to derived response
	void* pR;
};


class Request1: public Request
{

public:
	// implement custom request submission

};



void test_typeindex_map()
{

	std::map<std::type_index, std::vector<int>> mymap;


	mymap[typeid(Response1)].push_back(8);
	mymap[typeid(Response1)].push_back(9);

	mymap[typeid(Response2)].push_back(9);
	mymap[typeid(Response2)].push_back(9);


	Response1 output;

	int nr_responses = mymap[typeid(output)].size();
	std::cout << "Nr. entries in map: " << nr_responses << std::endl;

	if(nr_responses > 0)
	{
		for(int i=0;i<nr_responses;i++)
		{
			
			std::cout << "Val " << i << " : " << mymap[typeid(output)][i] << std::endl;
		}
	}
}


template<class T>
void get_type_id_from_ptr(T *response_container)
{
	std::cout << "ID: " << typeid(T).name() << std::endl;
}


int main(int argc, char** argv)
{
	Request1* myRequest = new Request1();

	// acquire response
	myRequest->AcquireResponse(1);

	// copy response from container
	Response1 outResponse;
	myRequest->CopyResponse(&outResponse);

	// delete original response container
	delete(myRequest);

	// display
	std::cout << outResponse.mMyData1;


	get_type_id_from_ptr(&outResponse);


	return 0;
} 
