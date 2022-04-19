#include <unistd.h>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include<iostream>
using namespace cv;
using namespace std;
using namespace xfeatures2d;
 
#define UNKNOWN_FLOW_THRESH 1e9
Mat src;
int COUNT = 0;
int COUNT2 = 0;
//gamma

void gammaTransform(Mat& rad)
{
	//rad.convertTo(rad, CV_32FC1);
	Mat srcImage = src.clone();
	Mat resultImage = srcImage.clone();
	//Mat src_lab;
		if (srcImage.channels() == 1)
	{
		for (int i= 0; i < src.rows; i++) 
		{
			for (int j = 0; j < src.cols; j++) 
			{
				float f = (srcImage.at<uchar>(i,j)+ 0.5f) / 255;
				f = (float)(pow(f, rad.at<float>(i,j)));
				resultImage.at<uchar>(i,j) = saturate_cast<uchar>(f*255.0f - 0.5f);
			}
		}
	}
	else{
		for (int i= 0; i < src.rows; i++) 
		{
			for (int j = 0; j < src.cols; j++) 
			{
				float f1 = (srcImage.at<Vec3b>(i,j)[0]+ 0.5f) / 255;
				float f2 = (srcImage.at<Vec3b>(i,j)[1]+ 0.5f) / 255;
				float f3 = (srcImage.at<Vec3b>(i,j)[2]+ 0.5f) / 255;
				f1 = (float)(pow(f1, (rad.at<float>(i,j))));
				f2 = (float)(pow(f2, (rad.at<float>(i,j))));
				f3 = (float)(pow(f3, (rad.at<float>(i,j))));
				resultImage.at<Vec3b>(i,j)[0] = saturate_cast<uchar>(f1*255.0f - 0.5f);
				resultImage.at<Vec3b>(i,j)[1] = saturate_cast<uchar>(f2*255.0f - 0.5f);
				resultImage.at<Vec3b>(i,j)[2] = saturate_cast<uchar>(f3*255.0f - 0.5f);
			}
		}
	}
	GaussianBlur(resultImage, resultImage, Size(3, 3), 1, 1);

	stringstream str;
    str << "./enhance/" << COUNT << ".png";  
	imwrite(str.str(), resultImage);
	COUNT++;


	imshow("enhance",resultImage);
}


//调整Lab分量
void  getEveryPixels(Mat& inputImage,Mat &outputImage,int dev){

   // namedWindow("show",WINDOW_AUTOSIZE);
  //  imshow("show",inputImage);
    
    outputImage = inputImage.clone();
    int rowNumbers = outputImage.rows;
    int cloNumbers = outputImage.cols;

    for (int i = 0; i < rowNumbers; i++)
    {
        for (int j = 0; j < cloNumbers; j++)
        {
          	//L
          //  outputImage.at<Vec3b>(i,j)[0] += dev ;
			//a
           // outputImage.at<Vec3b>(i,j)[1] = ;
  			 //b
           // outputImage.at<Vec3b>(i,j)[2] += dev;
		   if(outputImage.at<Vec3b>(i,j)[0] < dev){
			   outputImage.at<Vec3b>(i,j)[0] = 0;
		   }
        }
    }
    
}
//着色
void makecolorwheel(vector<Scalar> &colorwheel)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
 
    int i;
 
	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,	   255*i/RY,	 0));
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,		 0));
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,		   255,		 255*i/GC));
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,		   255-255*i/CB, 255));
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,	   0,		 255));
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,	   0,		 255-255*i/MR));
}
 
void motionToColor(Mat flow, Mat &color)
{
	Mat result;
	if (color.empty())
		color.create(flow.rows, flow.cols,CV_32F);
    //float rad[flow.rows][flow.cols];
	Mat rad(flow.rows, flow.cols, CV_32FC3);
	static vector<Scalar> colorwheel; //Scalar r,g,b
	if (colorwheel.empty())
		makecolorwheel(colorwheel);
 
	// determine motion range:
    float maxrad = -1;
 
	// Find max flow to normalize fx and fy
	for (int i= 0; i < flow.rows; i++) 
	{
		for (int j = 0; j < flow.cols; j++) 
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			rad.at<float>(i,j) = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad.at<float>(i,j) ? maxrad : rad.at<float>(i,j);
		}
	}
 
	for (int i= 0; i < flow.rows; i++) 
	{
		for (int j = 0; j < flow.cols; j++) 
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
 
			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			rad.at<float>(i,j) = sqrt(fx * fx + fy * fy);
			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;

		}
	}
	//归一化
	normalize(rad, rad, 0.2, 1, NORM_MINMAX);
	
	float max = rad.at<float>(0,0),min= rad.at<float>(0,0);
	for (int i= 0; i < flow.rows; i++) 
		{
			for (int j = 0; j < flow.cols; j++) 
			{
				if(rad.at<float>(i,j)>4){
					rad.at<float>(i,j)=0.76;
				}
				if(rad.at<float>(i,j)<0){
					rad.at<float>(i,j)=0.33;
				} 
			/*	if(rad.at<float>(i,j)= NULL){
					rad.at<float>(i,j)=3.3;
				}
				*/
				
				if(rad.at<float>(i,j)>max){
					max =  rad.at<float>(i,j);	
				}
				if(rad.at<float>(i,j)<min){
					min = rad.at<float>(i,j);
				}

				
				
			//	printf("%f    ",rad.at<float>(i,j));
			}
		//	printf("\n");
		}
	printf("max = %f\t min = %f\n",max,min);
	gammaTransform(rad);



}
 
int main(int, char**)
{
    VideoCapture cap;
	VideoWriter frame_write;

	cap.open(0);
	cap.open("/home/lwq/pictures/2.mp4");
	cap.set(CV_CAP_PROP_POS_FRAMES, 1);
/*
	Size S = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int)cap.get(CAP_PROP_FRAME_HEIGHT));
	int fps = cap.get(CAP_PROP_FPS);

	printf("current fps : %d \n", fps);
	VideoWriter writer("test.mp4", CAP_OPENCV_MJPEG, fps, S, true);
	
*/

 
    if( !cap.isOpened() )
        return -1;
 
    Mat prevgray, gray, flow, cflow, frame, lab,lab_p, lab_gray;
	
 //   namedWindow("flow", 1);
    namedWindow("original", 0);
	namedWindow("enhance", 0);
 
	Mat motion2color;
 
    for(;;)
    {
		double t = (double)cvGetTickCount();
 
        cap >> frame;

		stringstream str;
		str << "./pictures/" << COUNT2 << ".png";  
		imwrite(str.str(), frame);
		COUNT2++;


		src = frame.clone();
        cvtColor(frame, gray, CV_BGR2GRAY);
		imshow("original", frame);

  		if( prevgray.data )
        {
			// 稠密光流,Flow：frame输出的光流矩阵。矩阵大小同输入的图像一样大，是两个值，分别表示这个点在x方向与y方向的运动量（偏移量）
            calcOpticalFlowFarneback(prevgray, gray, frame, 0.5, 3, 15, 3, 5, 1.2, 0);
			motionToColor(frame, motion2color);
        //    imshow("flow", motion2color);
        }
	//	waitKey(30);
		//writer.write(frame);
        std::swap(prevgray, gray);

		if(char(waitKey(10)>=0)){
            break;
		}
 
    }
//	writer.release();
	cap.release();

    return 0;
}