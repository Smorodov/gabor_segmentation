
#define  _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <direct.h>
#include "fstream"
#include "iostream"
#include <vector>
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include "opencv2\opencv.hpp"
#include "cvgabor.h"
using namespace std;
using namespace cv;

const int N_scales=5;
const int N_orientations=8;

//-------------------------
// Запись матрицы в файл
//-------------------------

int save_mat_image(Mat& mat, string name)
{
	double maxVal,minVal;
	minMaxLoc(mat,&minVal,&maxVal);
	Mat tmp,tmp8u;
	tmp=Mat(mat.rows,mat.cols,CV_32FC1);
	if(fabs(maxVal-minVal)>FLT_EPSILON)
	{
		tmp=mat-minVal;
		tmp*=(255.0/(maxVal-minVal));
	}
	tmp8u=Mat(mat.rows,mat.cols,CV_8UC1);
	tmp.convertTo(tmp8u,CV_8UC1);

	imwrite(name,tmp8u);
	return 1;
}

void gabor_extraction(Mat& img,vector<Mat>& Descriptors)
{
	Size img_size = Size(img.cols,img.rows);
	Mat imtmp(img.rows,img.cols,CV_32FC1);
	img.convertTo(imtmp,CV_32FC1);

	CvGabor *Gabor;

	double sigma=2*CV_PI; // По умолчанию 2*PI, но приемлемые результаты дают и PI/2

	// Цикл по масштабам
	for (int v=0;v<N_scales;v++)
	{
		// Цикл по ориентациям
		for (int u=0;u<N_orientations;u++) 
		{
			Gabor = new CvGabor(u,(double)v,sigma);
			Descriptors[v*N_orientations+u]=Mat(imtmp.rows,imtmp.cols,CV_32FC1);
			Gabor->conv_img_a(imtmp,Descriptors[v*N_orientations+u],CV_GABOR_MAG);
			Descriptors[v*N_orientations+u];

			// Часто применяют сглаживание
			cv::GaussianBlur(Descriptors[v*N_orientations+u],Descriptors[v*N_orientations+u],Size(21,21),30);
			delete Gabor;
		}
	}

}

void extract_features(Mat& img,vector<Mat>& Descriptors)
{			
	Size img_size = Size(img.cols,img.rows);
	Mat gray(img_size,CV_8UC1);
	if(img.channels()==3)
	{
		cv::cvtColor(img,gray,CV_BGR2GRAY);
	}
	else
	{
		img.copyTo(gray);
	}

	gabor_extraction(gray,Descriptors);
}


// -----------------------------------------------------------------
// Сегментация (вычисление признаков + кластеризация)
// -----------------------------------------------------------------
void SegmentImage(Mat& Img,int numClasters,Mat &labels)
{
	int N_Descriptors=N_scales*N_orientations;
	// создаем вектор дескрипторов по количеству фильтров в банке
	vector<Mat> Descriptors(N_Descriptors); 
	// извлекаем произнаки (отклики фильтров)
	extract_features(Img,Descriptors);
	// Готовим данные для кластеризации
	Mat samples,centers;
	// Вектор mGabor.size() мерных векторов
	samples=Mat((Descriptors[0].rows *Descriptors[0].cols ),N_Descriptors,CV_32FC1);
	samples=0;
	for(int i=0;i<Descriptors[0].rows*Descriptors[0].cols;i++)
	{
		for(int j=0;j<N_Descriptors;j++)
		{
			double val=(Descriptors[j].reshape(1,1)).at<float>(i);
			samples.at<float>(i,j)=val;
		}
	}
	// Кластеризация. Разбиваем полученые mGabor.size() мерные точки на группы.
	kmeans(samples,numClasters, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, 0.010), 3, KMEANS_PP_CENTERS, centers);
	// Изображаем результат кластеризации
	labels=labels.reshape(1,Descriptors[0].rows);
}


int main(int argc, char * argv[])
{
	setlocale(LC_ALL, "Russian");

	// Чтение параметров командной строки
	if (argc<1)
	{
		cout << "Неправильное количество параметров." << "Использование: your_project <imagefile>" << endl;
		return 0;
	}

	Mat img = imread(argv[1],0);

	if(img.empty())
	{
		cout << "Ошибка загрузки изображения!" << endl;	
		return  0;
	}
	Mat labels;
	SegmentImage(img,2,labels);

	Mat result(img.rows,img.cols,CV_8UC3);
	cv::cvtColor(img,result,cv::COLOR_GRAY2BGR);
	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			int r=result.at<Vec3b>(i,j)[1]+labels.at<int>(i,j)*50;
			if(r>255){r=255;}
			result.at<Vec3b>(i,j)[1]=r;
		}
	}
	imshow("result",result);
	// Вывод результата сегментаци
	//save_mat_image(labels,"Desc0.jpg");		
	imwrite("result.jpg",result);

	waitKey(0);
	return 0;
}