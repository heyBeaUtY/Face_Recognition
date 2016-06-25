#include "Prehelper.h"
#include "BrowseDir.h"
#include "StatDir.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <contrib/contrib.hpp>
#include <cv.h>
#include <imgproc/imgproc.hpp>
#include <objdetect/objdetect.hpp>

#include <string>
#include <string.h>
#include <utility>
using namespace cv;
using namespace std;
CascadeClassifier cascade, nestedCascade; 
double scale = 1.0;
bool tryflip = false;

void CutImg(IplImage* src, CvRect rect,IplImage* res)
{
	CvSize imgsize;
	imgsize.height = rect.height;
	imgsize.width = rect.width;
	cvSetImageROI(src,rect);
	cvCopy(src,res);
	cvResetImageROI(res);
}

void resizeandtogray(char* dir,int K, vector<Mat> &images, vector<int> &labels,
	vector<Mat> &testimages, vector<int> &testlabels)
{
	IplImage* standard = cvLoadImage("D:\\privacy\\picture\\photo\\2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	string cur_dir;
	char id[5];
	int i,j;
	for(int i=1; i<=K; i++)
	{
		cur_dir = dir;
		cur_dir.append("gray\\");	
		_itoa(i,id,10);
		cur_dir.append(id);
		const char* dd = cur_dir.c_str();
		CStatDir statdir;
		if (!statdir.SetInitDir(dd))
		{
			puts("Dir not exist");
			return;
		}
		cout<<"Processing samples in Class "<<i<<endl;
		vector<char*>file_vec = statdir.BeginBrowseFilenames("*.*");
		for (j=0;j<file_vec.size();j++)
		{
			IplImage* cur_img = cvLoadImage(file_vec[j],CV_LOAD_IMAGE_GRAYSCALE);
			cvResize(cur_img,standard,CV_INTER_AREA);
			Mat cur_mat = cvarrToMat(standard,true),des_mat;
			cv::normalize(cur_mat,des_mat,0, 255, NORM_MINMAX, CV_8UC1);
			cvSaveImage(file_vec[j],cvCloneImage(&(IplImage) des_mat));
			if(j!=file_vec.size()-1)
			{
				images.push_back(des_mat);
				labels.push_back(i);
			}
			else
			{
				testimages.push_back(des_mat);
				testlabels.push_back(i);
			}
		}
		cout<<file_vec.size()<<" images."<<endl;
	}
}

vector<pair<char*,Mat>>  read_img(const string& dir)
{
	CStatDir statdir;
	pair<char*,Mat> pfi;
	vector<pair<char*,Mat>> Vp;
	if (!statdir.SetInitDir(dir.c_str()))
	{
		cout<<"Direct "<<dir<<"  not exist!"<<endl;
		return Vp;
	}
	int cls_id = dir[dir.length()-1]-'0';
	vector<char*>file_vec = statdir.BeginBrowseFilenames("*.*");
	int i,s = file_vec.size();
	for (i=0;i<s;i++)
	{
		pfi.first = file_vec[i];
		pfi.second = imread(file_vec[i]);
		Vp.push_back(pfi);
	}
	return Vp;
}

vector<Rect> detectAndDraw( Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip )
{
	int i = 0;
	double t = 0;
	vector<Rect> faces, faces2,res;
	const static Scalar colors[] =  { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255)} ;
	Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

	cvtColor( img, gray, CV_BGR2GRAY );
	resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
	equalizeHist( smallImg, smallImg );

	t = (double)cvGetTickCount();
	cascade.detectMultiScale( smallImg, faces,
		1.1, 2, 0
		|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		//|CV_HAAR_SCALE_IMAGE
		,
		Size(30, 30) );
	if( tryflip )
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale( smallImg, faces2,
			1.1, 2, 0
			|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			//|CV_HAAR_SCALE_IMAGE
			,
			Size(30, 30) );
		for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)cvGetTickCount() - t;
//	printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
// 	Mat imgcopy = img.clone();
// 	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
// 	{
// 		Mat smallImgROI;
// 		vector<Rect> nestedObjects;
// 		Point center;
// 		Scalar color = colors[i%8];
// 		int radius;
// 
// 		double aspect_ratio = (double)r->width/r->height;
//  		rectangle( imgcopy, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
//  			cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
//  			color, 3, 8, 0);
// 		if( nestedCascade.empty() )
// 			continue;
// 		smallImgROI = smallImg(*r);
// 		nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
// 			1.1, 2, 0
// 			|CV_HAAR_FIND_BIGGEST_OBJECT
// 			//|CV_HAAR_DO_ROUGH_SEARCH
// 			//|CV_HAAR_DO_CANNY_PRUNING
// 			//|CV_HAAR_SCALE_IMAGE
// 			,
// 			Size(30, 30) );
// 	}
//	cv::imshow( "result", imgcopy );
	return faces;
}

pair<IplImage*,Rect> DetectandExtract(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{
	vector<Rect> Rvec = detectAndDraw(img,cascade,nestedCascade,scale,tryflip);
	int i,maxxsize=0,id=-1,area;
	for (i=0;i<Rvec.size();i++)
	{
		area = Rvec[i].width*Rvec[i].height;
		if(maxxsize<area)
		{
			maxxsize = area;
			id = i;
		}
	}
	IplImage* transimg = cvCloneImage(&(IplImage)img);
	pair<IplImage*,Rect> p;
	if(id!=-1)
	{
		CvSize imgsize;
		imgsize.height = Rvec[id].height;
		imgsize.width = Rvec[id].width;
		IplImage* res = cvCreateImage(imgsize,transimg->depth,transimg->nChannels);
		CutImg(transimg,Rvec[id],res);
		p.first = res; p.second = Rvec[id];
	}
	return p;
}

void Init()
{
	string cascadeName = "E:/software/opencv2.4.6.0/data/haarcascades/haarcascade_frontalface_alt.xml";
	string nestedCascadeName = "E:/software/opencv2.4.6.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
	double scale = 1.0;
	bool tryflip = false;
	if( !cascade.load( cascadeName ) ||!nestedCascade.load( nestedCascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade or nestedCascade" << endl;//若出现该问题请去检查cascadeName，可能是opencv版本路径问题
		return;
	}
}

void preprocess_trainingdata(char* dir,int K)
{
	Init();
	/************************************************************************/
	/*                                  detect face and save                                    */
	/************************************************************************/
	int i,j;
	cout<<"detect and save..."<<endl;
	string cur_dir;
	char id[5];
	for(i=1; i<=K; i++)
	{
		cur_dir = dir;
		_itoa(i,id,10);
		cur_dir.append("color\\");
		cur_dir.append(id);
		vector<pair<char*,Mat>> imgs=read_img(cur_dir);
		IplImage* res; Rect r;
		pair<IplImage*, Rect> p;
		for(j=0;j<imgs.size();j++)
		{
			p = DetectandExtract(imgs[j].second,cascade,nestedCascade,scale,tryflip);
			res = p.first;
			if(res)
				cvSaveImage(imgs[j].first,res);
		}
	}
}

static Mat toGrayscale(InputArray _src) {
	Mat src = _src.getMat();
	// only allow one channel
	if(src.channels() != 1) {
		CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
	}
	// create and return normalized image
	Mat dst;
	cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

Ptr<FaceRecognizer> Train(vector<Mat> images, vector<int> labels,
	vector<Mat> testimages, vector<int> testlabels)
{
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer(10);//10 Principal components
	cout<<"train"<<endl;
	model->train(images,labels);

	Mat eigenvalues = model->getMat("eigenvalues");
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getMat("eigenvectors");
	// From this we will display the (at most) first 10 Eigenfaces:
	for (int i = 0; i < min(15, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = toGrayscale(ev.reshape(1, images[0].rows));
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		imshow(format("%d", i), cgrayscale);
	}
	waitKey(0);

	int i,acc=0,predict_l;
	for (i=0;i<testimages.size();i++)
	{
		predict_l = model->predict(testimages[i]);
		if(predict_l != testlabels[i])
		{
			cout<<"An error in recognition: sample "<<i+1<<", predict "<<
				predict_l<<", groundtruth "<<testlabels[i]<<endl;
			imshow("error 1",testimages[i]);
			waitKey();
		}
		else
			acc++;
	}
	cout<<"Recognition Rate: "<<acc*1.0/testimages.size()<<endl;
	return model;
}

void CaptureandRecognize(Ptr<FaceRecognizer>model)
{
	CvCapture* capture = 0;
	Mat frame, frameCopy, image;
	Init();
	capture = cvCaptureFromCAM(0);
	if(!capture){
		cout << "Capture from camera didn't work" << endl;
		return;
	}

	//write the size of features on picture
	CvFont font;    
	double hScale=1;   
	double vScale=1;    
	int lineWidth=2;
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
	char strlabel[5]; 

	while(true)
	{
		IplImage* iplImg = cvQueryFrame( capture );
		frame = iplImg;
		if( frame.empty() )
			break;
		if( iplImg->origin == IPL_ORIGIN_TL )
			frame.copyTo( frameCopy );
		else
			flip( frame, frameCopy, 0 );

		pair<IplImage*,Rect> p = DetectandExtract(frameCopy, cascade,nestedCascade,scale,tryflip);
		IplImage* ori_detected = p.first;
		Rect r = p.second;
		if(ori_detected &&strlen(ori_detected->imageData)>10)
		{
			IplImage* gray_detected = cvCreateImage(cvGetSize(ori_detected),ori_detected->depth,1);
			cvCvtColor(ori_detected,gray_detected,CV_RGB2GRAY);
			IplImage* detected = cvLoadImage("D:\\privacy\\picture\\photo\\2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
			cvResize(gray_detected,detected);
			int pred_label = model->predict(cvarrToMat(detected,true));
			//cout<<pred_label<<endl;
			_itoa(pred_label,strlabel,10);;
			cvRectangle(iplImg,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),cvScalar(50,255,50),3);
			cvPutText(iplImg,strlabel,cvPoint(r.x+r.width,r.y-20),&font,CV_RGB(255,0,0));//在图片中输出字符
			cvShowImage("Show",iplImg);
		}

		if( waitKey( 10 ) >= 0 )
			goto _cleanup_;
	}

	waitKey(0);

_cleanup_:
	cvReleaseCapture( &capture );

}
