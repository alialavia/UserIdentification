#include <string>
#include <fstream>
#include <vector>

class CSVWriter {
public:
	CSVWriter(std::string filename) :filename_(filename), col_nr(0), line_nr(0) {
		filehandle_ = new std::ofstream();
		filehandle_->open(filename);
	}
	void startNewRow() {
		line_nr++;
		col_nr = 0;
		*filehandle_ << "\n";
	}
	template<class T> void addEntry(T val) {
		if (col_nr > 0) {
			// add new column
			*filehandle_ << ",";
		}
		*filehandle_ << val;
		col_nr++;
	}
	~CSVWriter() {
		filehandle_->close();
	}
	void changeFile(std::string filename) {
		filehandle_->close();
		delete(filehandle_);
		filehandle_ = new std::ofstream();
		filehandle_->open(filename);
		filename_ = filename;
	}

private:
	std::string filename_;
	std::ofstream* filehandle_;
	int col_nr;
	int line_nr;
};


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
