#include <io\CSVHandling.h>
#include <fstream>

using namespace io;

CSVWriter::CSVWriter(std::string filename) :filename_(filename), col_nr(0), line_nr(0) {
	filehandle_ = new std::ofstream();
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
void CSVWriter::startNewRow() {
	line_nr++;
	col_nr = 0;
	*filehandle_ << "\n";
}
CSVWriter::~CSVWriter() {
	filehandle_->close();
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
