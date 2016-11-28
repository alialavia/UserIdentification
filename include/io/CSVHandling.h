#ifndef TRACKING_FACETRACKER_H_
#define TRACKING_FACETRACKER_H_

#include <string>
#include <vector>

namespace io 
{

class CSVWriter {
public:
	CSVWriter(std::string filename);
	~CSVWriter();
	void startNewRow();
	template<class T> void addEntry(T val) {
		if (col_nr > 0) {
			// add new column
			*filehandle_ << ",";
		}
		*filehandle_ << val;
		col_nr++;
	};
	void changeFile(std::string filename);
private:
	std::string filename_;
	std::ofstream* filehandle_;
	int col_nr;
	int line_nr;
};

template <class T>
class CSVParser {
public:
	CSVParser(std::string filename);
	bool OpenFile();
	bool IterateRows();
	T GetVal(int col);

	~CSVParser();
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
