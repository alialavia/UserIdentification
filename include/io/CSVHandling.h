#ifndef IO_CSVHANDLING_H_
#define IO_CSVHANDLING_H_

#include <string>
#include <vector>
#include <fstream>

namespace io 
{

class CSVWriter {
public:
	CSVWriter(std::string filename);
	~CSVWriter();
	void EndRow();
	template<class T> void addEntry(T val) {
		if (col_nr > 0) {
			// add new column
			*filehandle_ << ",";
		}
		*filehandle_ << val;
		col_nr++;
	};
	template<class T>
	void addList(std::vector<T> values) {
		for (size_t i = 0; i < values.size();i++) {
			addEntry(values[i]);
			EndRow();
		}
	}
	template<class T>
	void addRow(std::vector<T> values) {
		for (size_t i = 0; i < values.size(); i++) {
			addEntry(values[i]);
		}
		EndRow();
	}
	void changeFile(std::string filename);
	static bool dirExists(const std::string& dirName_in);
private:
	std::string filename_;
	std::ofstream* filehandle_;
	int col_nr;
	int line_nr;
};


// TODO: refactoring to separated definition ending up in linker error
template <class T>
class CSVParser {
public:
	CSVParser(std::string filename) :filename_(filename), line_number_(0) {
		filehandle_ = new std::ifstream(filename);
	}

	bool OpenFile() {
		if (filehandle_->is_open()) {
			return true;
		}
		else {
			return false;
		}
	}
	bool IterateRows()
	{
		line_number_++;
		std::string item_string;
		std::getline(*filehandle_, current_line_);
		if (!filehandle_->eof())
		{
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
	T GetVal(int col) {
		return current_items_[col];
	}

	~CSVParser() {
		filehandle_->close();
		delete filehandle_;
	}
private:
	std::string filename_;
	int line_number_;
	std::string current_line_;
	std::vector<T> current_items_;
	T current_item_;

	std::ifstream* filehandle_;

};

}	// /namespace

#endif
