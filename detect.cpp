//Detect.cpp
//Preprocessing - Detect, Cut and Save
//@Author : Rachel-Zhang

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include "Prehelper.h"

using namespace std;
using namespace cv;
#define CAM 2
#define PHO 1
#define K 15


int main( )
{
	string inputName;	
	int mode;

	char dir[256] = "D:\\Courses\\CV\\Face_recognition\\pic\\"; 
	/************************************************************************/
	/*                                Model                                      */
	/************************************************************************/
	//preprocess_trainingdata(dir,K); //face_detection and extract to file
	vector<Mat> images,testimages;
	vector<int> labels,testlabels;
	//togray, normalize and resize; load to images,labels,testimages,testlabels
	resizeandtogray(dir,K,images,labels,testimages,testlabels); 
	//recognition
	Ptr<FaceRecognizer> model = Train(images,labels,testimages,testlabels);
	/************************************************************************/
	/*                                   Detection and Recognition                                   */
	/************************************************************************/
	CaptureandRecognize(model);

	system("pause");
	return 0;
}

