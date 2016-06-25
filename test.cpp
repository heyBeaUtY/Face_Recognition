#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "detection.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

//  the primary trained classifier such as frontal face
string cascadeName = "E:/software/opencv2.4.6.0/data/haarcascades/haarcascade_frontalface_alt.xml";
//an optional secondary classifier such as eyes
string nestedCascadeName = "E:/software/opencv2.4.6.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main( )
{
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    string inputName;
    bool tryflip = false;
	
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;
	 int n_pic;
	 char** pic_paths = (char**) malloc(sizeof*pic_paths);
	 scanf("%d",&n_pic);

    for( int i = 1; i <= n_pic; i++ )
    {
		pic_paths[i] = (char*) malloc(sizeof(char)*100);
		cin>>pic_paths[i];
        cout << "Processing " << i << " " <<  pic_paths[i] << endl;
		inputName.assign( pic_paths[i] );
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;//若出现该问题请去检查cascadeName，可能是opencv版本路径问题
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
        }
    }

    cvNamedWindow( "result", 1 );

    if( capture )
    {
        cout << "In capture ..." << endl;
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

            detectAndDraw( frameCopy, cascade, nestedCascade, scale, tryflip );

            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    else
    {
        cout << "In image read" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf), c;
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
                        c = waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    cvDestroyWindow("result");

    return 0;
}
