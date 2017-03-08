#include <io\CSVHandling.h>
#include <fstream>

#include <iostream>
#include <windows.h>
#include <fileapi.h>

using namespace io;

CSVWriter::CSVWriter(std::string filename) :filename_(filename), col_nr(0), line_nr(0) {
	filehandle_ = new std::ofstream();
	// create directory
	std::size_t botDirPos = filename.find_last_of("/");
	// get directory
	std::string dir = filename.substr(0, botDirPos);
	// get file
	std::string file = filename.substr(botDirPos, filename.length());

	if(!dirExists(dir))
	{
		CreateDirectory(dir.c_str(), NULL);
	}
	
	// check if file exists
	struct stat buffer;
	if((stat(filename.c_str(), &buffer) == 0))
	{
		// append
		filehandle_->open(filename, std::ofstream::app);
	}
	else
	{
		filehandle_->open(filename);
	}
}
void CSVWriter::EndRow() {
	line_nr++;
	col_nr = 0;
	*filehandle_ << "\n";
}
CSVWriter::~CSVWriter() {
	filehandle_->close();
	delete(filehandle_);
}
void CSVWriter::changeFile(std::string filename) {
	filehandle_->close();
	delete(filehandle_);
	filehandle_ = new std::ofstream();
	struct stat buffer;
	if ((stat(filename.c_str(), &buffer) == 0))
	{
		// append
		filehandle_->open(filename, std::ofstream::app);
	}
	else
	{
		filehandle_->open(filename);
	}
	filename_ = filename;
}

bool CSVWriter::dirExists(const std::string& dirName_in)
{
	DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}
