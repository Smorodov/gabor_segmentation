#include "cvgabor.h"
using namespace std;
using namespace cv;
//----------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------
CvGabor::CvGabor()    //Конструктор
{
}
//----------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------
CvGabor::~CvGabor()   //Деструктор
{

}
//----------------------------------------------------------------------------------------------
// Создаем фильтр Габора с ориентацией iMu*PI/8, масштабом iNu и дисперсией sigma равной dSigma
// Пространственная частота F устанавливается по умолчанию равной sqrt(2).
//----------------------------------------------------------------------------------------------
CvGabor::CvGabor(int iMu, int iNu, double dSigma)
{ 
	F = sqrt(2.0);				// F устанавливается по умолчанию равной sqrt(2)
	Init(iMu, iNu, dSigma, F);	// Инициализация параметров, создание ядра
}


/*!
\fn CvGabor::CvGabor(int iMu, int iNu, double dSigma, double dF)
Construct a gabor

Parameters:
iMu		The orientation iMu*PI/8
iNu 		The scale
dSigma 		The sigma value of Gabor
dF		The spatial frequency        //їХјдЖµВК dF

Returns:
None

Create a gabor with a orientation iMu*PI/8, a scale iNu, a sigma value dSigma, and a spatial frequence dF. It calls Init() to generate parameters and kernels.
*/
CvGabor::CvGabor(int iMu, int iNu, double dSigma, double dF)   //їХјдЖµВКєНsigma¶јКЗУГ»§ЧФјєЙиЦГµДЎЈ
{

	Init(iMu, iNu, dSigma, dF);

}


/*!
\fn CvGabor::CvGabor(double dPhi, int iNu)
Construct a gabor

Parameters:
dPhi		The orientation in arc   
iNu 		The scale

Returns:
None

Create a gabor with a orientation dPhi, and with a scale iNu. The sigma (Sigma) and the spatial frequence (F) are set to 2*PI and sqrt(2) defaultly. It calls Init() to generate parameters and kernels.
*/
CvGabor::CvGabor(double dPhi, int iNu)   //
{
	Sigma = 2*PI;
	F = sqrt(2.0);
	Init(dPhi, iNu, Sigma, F);
}


/*!
\fn CvGabor::CvGabor(double dPhi, int iNu, double dSigma)
Construct a gabor

Parameters:
dPhi		The orientation in arc
iNu 		The scale
dSigma		The sigma value of Gabor

Returns:
None

Create a gabor with a orientation dPhi, a scale iNu, and a sigma value dSigma. The spatial frequence (F) is set to sqrt(2) defaultly. It calls Init() to generate parameters and kernels.
*/
CvGabor::CvGabor(double dPhi, int iNu, double dSigma) //·ЅПтЎўіЯґзґуРЎєНsigma¶јКЗУГ»§ЧФјєЙи¶ЁЈ¬їХјдЖµВКУГИ±КЎЦµsqrt(2.0)
{

	F = sqrt(2.0);
	Init(dPhi, iNu, dSigma, F);
}


/*!
\fn CvGabor::CvGabor(double dPhi, int iNu, double dSigma, double dF)
Construct a gabor

Parameters:
dPhi		The orientation in arc
iNu 		The scale
dSigma 		The sigma value of Gabor
dF		The spatial frequency 

Returns:
None

Create a gabor with a orientation dPhi, a scale iNu, a sigma value dSigma, and a spatial frequence dF. It calls Init() to generate parameters and kernels.
*/
CvGabor::CvGabor(double dPhi, int iNu, double dSigma, double dF)  //ЛщУРµД¶јКЗУГ»§ЧФјєЙи¶ЁЎЈ
{

	Init(dPhi, iNu, dSigma,dF);
}

/*!
\fn CvGabor::IsInit()
Determine the gabor is initilised or not

Parameters:
None

Returns:
a boolean value, TRUE is initilised or FALSE is non-initilised.

Determine whether the gabor has been initlized - variables F, K, Kmax, Phi, Sigma are filled.
*/
bool CvGabor::IsInit()
{

	return bInitialised;
}

/*!
\fn CvGabor::mask_width()
Give out the width of the mask

Parameters:
None

Returns:
The long type show the width.

Return the width of mask (should be NxN) by the value of Sigma and iNu.
*/
long CvGabor::mask_width()  //И·¶ЁmaskµДїн¶И ЈЁN*Nґ°їЪЈ©
{

	long lWidth;
	if (IsInit() == false)  
	{
		perror ("Error: The Object has not been initilised in mask_width()!\n");
		return 0;
	}
	else 
	{
		//determine the width of Mask
		double dModSigma = Sigma/K;
		double dWidth = (int)(dModSigma*6 + 1);
		//test whether dWidth is an odd.
		if (fmod(dWidth, 2.0)==0.0) dWidth++;
		lWidth = (long)dWidth;

		return lWidth;
	}
}


/*!
\fn CvGabor::creat_kernel()
Create gabor kernel

Parameters:
None

Returns:
None

Create 2 gabor kernels - REAL and IMAG, with an orientation and a scale 
*/
void CvGabor::creat_kernel()   //ґґЅЁgaborєЛ
{

	if (IsInit() == false) {perror("Error: The Object has not been initilised in creat_kernel()!\n");}
	else 
	{
		Mat mReal, mImag;

		mReal = Mat( Width, Width, CV_32F);  //КµІїґ°їЪїтµДґуРЎ
		mImag = Mat( Width, Width, CV_32FC1);  //РйІїґ°їЪїтµДґуРЎ

		/**************************** Gabor Function ****************************/ 
		int x, y;
		double dReal;
		double dImag;
		double dTemp1, dTemp2, dTemp3;

		for (int i = 0; i < Width; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				x = i-(Width-1)/2;
				y = j-(Width-1)/2;
				dTemp1 = (pow(K,2)/pow(Sigma,2))*exp(-(pow((double)x,2)+pow((double)y,2))*pow(K,2)/(2*pow(Sigma,2)));//
				dTemp2 = cos(K*cos(Phi)*x + K*sin(Phi)*y) - exp(-(pow(Sigma,2)/2));  //
				dTemp3 = sin(K*cos(Phi)*x + K*sin(Phi)*y);  //
				dReal = dTemp1*dTemp2;    //
				dImag = dTemp1*dTemp3;    //
				mReal.at<float>(i,j)=dReal;
				mImag.at<float>(i,j)=dImag;
			} 
		}//FOR
		/**************************** Gabor Function ****************************/
		bKernel = true;		
		mReal.copyTo(Real);
		mImag.copyTo(Imag);
	}
}
/*!
\fn CvGabor::get_image(int Type)
Get the speific type of image of Gabor

Parameters:
Type		The Type of gabor kernel, e.g. REAL, IMAG, MAG, PHASE   

Returns:
Pointer to image structure, or NULL on failure	

Return an Image (gandalf image class) with a specific Type   "REAL"	"IMAG" "MAG" "PHASE"  
*/
Mat CvGabor::get_image(int Type)
{

	if(IsKernelCreate() == false)
	{ 
		perror("Error: the Gabor kernel has not been created in get_image()!\n");
		return Mat();
	}
	else
	{  
		Mat pImage;
		Mat newimage;
		newimage   = Mat(Size(Width,Width), CV_8UC1);
		pImage     = Mat(Size(Width,Width), CV_32FC1);
		Mat kernel = Mat(Width, Width, CV_32FC1);
		double ve;
		Scalar S;
		Size size  = Size(Width, Width);
		int rows = size.height;
		int cols = size.width;
		switch(Type)
		{
		case 1:  //Real
			Real.copyTo(kernel);
			kernel.copyTo(pImage);
			break;
		case 2:  //Imag

			Imag.copyTo(kernel);
			kernel.copyTo(pImage);
			break; 
		case 3:  //Magnitude
			///@todo  
			break;
		case 4:  //Phase
			///@todo
			break;
		}
		cv::normalize(pImage,pImage,0,255,CV_MINMAX);
		pImage.convertTo(newimage,CV_8UC1);
		return newimage;
	}
}


/*!
\fn CvGabor::IsKernelCreate()
Determine the gabor kernel is created or not

Parameters:
None

Returns:
a boolean value, TRUE is created or FALSE is non-created.

Determine whether a gabor kernel is created.
*/
bool CvGabor::IsKernelCreate()
{

	return bKernel;
}


/*!
\fn CvGabor::get_mask_width()
Reads the width of Mask

Parameters:
None

Returns:
Pointer to long type width of mask.
*/
long CvGabor::get_mask_width()
{
	return Width;
}


/*!
\fn CvGabor::Init(int iMu, int iNu, double dSigma, double dF)
Initilize the.gabor

Parameters:
iMu 	The orientations which is iMu*PI.8
iNu 	The scale can be from -5 to infinit
dSigma 	The Sigma value of gabor, Normally set to 2*PI
dF 	The spatial frequence , normally is sqrt(2)

Returns:

Initilize the.gabor with the orientation iMu, the scale iNu, the sigma dSigma, the frequency dF, it will call the function creat_kernel(); So a gabor is created.
*/
void CvGabor::Init(int iMu, int iNu, double dSigma, double dF)
{
	//Initilise the parameters 
	bInitialised = false;
	bKernel = false;

	Sigma = dSigma;
	F = dF;

	Kmax = PI/2;

	// Absolute value of K
	K = Kmax / pow(F, (double)iNu);
	Phi = PI*iMu/8;
	bInitialised = true;
	Width = mask_width();
	Real = Mat( Width, Width, CV_32FC1);
	Imag = Mat( Width, Width, CV_32FC1);
	creat_kernel();  
}

/*!
\fn CvGabor::Init(double dPhi, int iNu, double dSigma, double dF)
Initilize the.gabor

Parameters:
dPhi 	The orientations 
iNu 	The scale can be from -5 to infinit
dSigma 	The Sigma value of gabor, Normally set to 2*PI
dF 	The spatial frequence , normally is sqrt(2)

Returns:
None

Initilize the.gabor with the orientation dPhi, the scale iNu, the sigma dSigma, the frequency dF, it will call the function creat_kernel(); So a gabor is created.filename 	The name of the image file
file_format 	The format of the file, e.g. GAN_PNG_FORMAT
image 	The image structure to be written to the file
octrlstr 	Format-dependent control structure

*/
void CvGabor::Init(double dPhi, int iNu, double dSigma, double dF)
{

	bInitialised = false;
	bKernel = false;
	Sigma = dSigma;
	F = dF;

	Kmax = PI/2;

	// Absolute value of K
	K = Kmax / pow(F, (double)iNu);
	Phi = dPhi;
	bInitialised = true;
	Width = mask_width();
	Real = Mat( Width, Width, CV_32FC1);
	Imag = Mat( Width, Width, CV_32FC1);
	creat_kernel();  
}



/*!
\fn CvGabor::get_matrix(int Type)
Get a matrix by the type of kernel

Parameters:
Type		The type of kernel, e.g. REAL, IMAG, MAG, PHASE

Returns:
Pointer to matrix structure, or NULL on failure.

Return the gabor kernel.
*/
Mat CvGabor::get_matrix(int Type)
{
	if (!IsKernelCreate()) {perror("Error: the gabor kernel has not been created!\n"); return Mat();}
	switch (Type)
	{
	case CV_GABOR_REAL:
		return Real;
		break;
	case CV_GABOR_IMAG:
		return Imag;
		break;
	case CV_GABOR_MAG:
		return Mat();
		break;
	case CV_GABOR_PHASE:
		return Mat();
		break;
	}
}




/*!
\fn CvGabor::output_file(const char *filename, Gan_ImageFileFormat file_format, int Type)
Writes a gabor kernel as an image file.

Parameters:
filename 	The name of the image file
file_format 	The format of the file, e.g. GAN_PNG_FORMAT
Type		The Type of gabor kernel, e.g. REAL, IMAG, MAG, PHASE   
Returns:
None

Writes an image from the provided image structure into the given file and the type of gabor kernel.
*/
void CvGabor::output_file(const char *filename, int Type)
{
	Mat pImage;
	pImage = get_image(Type);
	if(!pImage.empty())
	{
		if( imwrite(filename, pImage )) printf("%s has been written successfully!\n", filename);
		else printf("Error: writting %s has failed!\n", filename);
	}
	else 
		perror("Error: the image is empty in output_file()!\n"); 
}

/*!
\fn CvGabor::conv_img_a(IplImage *src, IplImage *dst, int Type)
*/
void CvGabor::conv_img_a(Mat& src, Mat& dst, int Type)   //НјПсЧцgaborѕн»э  єЇКэГыЈєconv_img_a
{
	double ve, re,im;

	int width = src.cols;
	int height = src.rows;
	Mat mat = Mat(src.rows, src.cols, CV_32FC1);
	src.copyTo(mat);

	Mat rmat = Mat(width, height, CV_32FC1);  //ґжКµІї
	Mat imat = Mat(width, height, CV_32FC1);  //ґжРйІї

	Mat kernel = Mat( Width, Width, CV_32FC1 ); //ґґЅЁєЛєЇКэґ°їЪ

	switch (Type)
	{
	case CV_GABOR_REAL:		//
		Real.copyTo(kernel);
		filter2D(mat,mat,CV_32FC1,kernel);	
		break;
	case CV_GABOR_IMAG:      //
		Imag.copyTo(kernel);
		filter2D(mat,mat,CV_32FC1,kernel);
		break;
	case CV_GABOR_MAG:   //
		/* Real Response */
		Real.copyTo(kernel);
		filter2D(mat,rmat,CV_32FC1,kernel);
		/* Imag Response */
		Imag.copyTo(kernel);
		filter2D(mat,imat,CV_32FC1,kernel);
		/* Magnitude response is the square root of the sum of the square of real response and imaginary response */
		cv::magnitude(imat,rmat,mat);
		break;
	case CV_GABOR_PHASE:
		/* Real Response */
		Real.copyTo(kernel);
		filter2D(mat,rmat,CV_32FC1,kernel);
		/* Imag Response */
		Imag.copyTo(kernel);
		filter2D(mat,imat,CV_32FC1,kernel);

		cv::phase(imat,rmat,mat,false);
		break;
	}

	if (dst.type() == CV_8UC1)  //№йТ»»Ї
	{
		cv::normalize(mat,mat,0,255,CV_MINMAX);
		mat.convertTo(dst,CV_8UC1);
	}

	if (dst.type() == CV_32FC1)
	{
		mat.copyTo(dst);
	} 
}


/*!
\fn CvGabor::CvGabor(int iMu, int iNu)
*/
CvGabor::CvGabor(int iMu, int iNu)
{
	double dSigma = 2*PI; 
	F = sqrt(2.0);
	Init(iMu, iNu, dSigma, F);
}


/*!
\fn CvGabor::normalize( const CvArr* src, CvArr* dst, double a, double b, int norm_type, const CvArr* mask )
*/
void CvGabor::normalize( Mat& src, Mat& dst, double a, double b, int norm_type, Mat& mask )
{
	  Mat tmp;
//	__BEGIN__;

	double scale, shift;

	if( norm_type == CV_MINMAX )
	{
		double smin = 0, smax = 0;
		double dmin = MIN( a, b ), dmax = MAX( a, b );
		minMaxLoc( src, &smin, &smax, 0, 0, mask );
		scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
		shift = dmin - smin*scale;
	}
	else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
	{
		scale = norm( src, 0, norm_type, mask );
		scale = scale > DBL_EPSILON ? 1./scale : 0.;
		shift = 0;
	}
	else {}



	if( mask.empty() )
		src.convertTo( dst, scale, shift);
	else
	{
		src.convertTo( tmp, scale, shift);
		tmp.copyTo(dst, mask);
	}

//	__END__;
}


/*!
\fn CvGabor::conv_img(IplImage *src, IplImage *dst, int Type)
*/

void CvGabor::conv_img(Mat& src, Mat& dst, int Type)   //єЇКэГыЈєconv_img
{
	double ve, re,im;

	Mat mat = Mat(src.rows, src.cols, CV_32FC1);
	src.convertTo(mat,CV_32FC1);

	Mat rmat = Mat(src.rows, src.cols, CV_32FC1);
	Mat imat = Mat(src.rows, src.cols, CV_32FC1);

	switch (Type)
	{
	case CV_GABOR_REAL:
		filter2D( mat, mat,CV_32FC1, Real);
		break;
	case CV_GABOR_IMAG:
		filter2D( mat, mat,CV_32FC1, Imag);
		break;
	case CV_GABOR_MAG:
		filter2D( mat, rmat,CV_32FC1, Real);
		filter2D( mat, imat,CV_32FC1, Imag);
		cv::magnitude(imat,rmat,mat);
		break;
	case CV_GABOR_PHASE:
		filter2D( mat, rmat,CV_32FC1, Real);
		filter2D( mat, imat,CV_32FC1, Imag);
		cv::phase(imat,rmat,mat,false);
		break;
	}
	
	if (dst.type() == CV_8UC1)
	{
		cv::normalize(mat, mat, 0, 255, CV_MINMAX);
		mat.convertTo(dst,CV_8UC1);
	}
	
	if (dst.type() == CV_32FC1)
	{
		mat.copyTo(dst);
	}
}
