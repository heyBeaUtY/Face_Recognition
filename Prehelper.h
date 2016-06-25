#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <cv.h>
#include <vector>
#include <utility>
using namespace cv;
using namespace std;

void CutImg(IplImage* src, CvRect rect,IplImage* res);

vector<Rect> detectAndDraw( Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip);

pair<IplImage*,Rect>  DetectandExtract(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip);

int read_img(const string& dir, vector<Mat> &images);

vector<pair<char*,Mat>>  read_img(const string& dir);

void preprocess_trainingdata(char* dir,int k);

void resizeandtogray(char* dir,int k, 	vector<Mat> &images, vector<int> &labels,
		vector<Mat> &testimages, vector<int> &testlabels);

Ptr<FaceRecognizer> Train(vector<Mat> images, vector<int> labels,
	vector<Mat> testimages, vector<int> testlabels);

void CaptureandRecognize(Ptr<FaceRecognizer>model);
