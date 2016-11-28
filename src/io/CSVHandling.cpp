#include <io\CSVHandling.h>
#include <fstream>

using namespace io;

CSVWriter::CSVWriter(std::string filename) :filename_(filename), col_nr(0), line_nr(0) {
	filehandle_ = new std::ofstream();
	filehandle_->open(filename);
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
	filehandle_->open(filename);
	filename_ = filename;
}

// -------- CSV parser

template <class T>
CSVParser<T>::CSVParser(std::string filename):filename_(filename), line_number_(0) {
		filehandle_ = new std::ifstream(filename);
}

template <class T>
CSVParser<T>::~CSVParser() {
	filehandle_->close();
	delete filehandle_;
}

template <class T>
bool CSVParser<T>::OpenFile() {
	if (filehandle_->is_open()) {
		return true;
	}
	else {
		return false;
	}
}

template <class T>
bool CSVParser<T>::IterateRows()
{

	line_number_++;

	std::string item_string;

	std::getline(*filehandle_, current_line_);

	if (!filehandle_->eof())
	{
		//std::cout << current_line_ << std::endl;

		current_items_.clear();
		std::istringstream inStream(current_line_);

		T value;
		while (std::getline(inStream, item_string, ',') && std::istringstream(item_string) >> value)
		{
			//std::cout << csvItem << std::endl;
			current_items_.push_back(value);
		}
		return true;
	}
	return false;
}


template <class T>
T CSVParser<T>::GetVal(int col) {
	return current_items_[col];
}
