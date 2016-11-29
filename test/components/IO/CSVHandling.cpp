#include <iostream>
#include <io/CSVHandling.h>


int main(int argc, char** argv)
{
	io::CSVWriter writer("somename2.csv");
	std::string some_Str = "Mystring";
	writer.addEntry<std::string>(some_Str);
	return 0;
} 
