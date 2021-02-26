#include "mtcnn_new.h"
#include "NewFaceDetectionLib.h"
//#include <objbase.h>
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include "boost/scoped_ptr.hpp"
#include "caffe/layer.hpp"
#include <caffe/solver.hpp>
#include <caffe/sgd_solvers.hpp>
#include "caffe/layer_factory.hpp"
#include <caffe/layers/tanh_layer.hpp>
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/Softmax_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;

//using namespace FaceInception;
static int MinImageWidth = 50;
static int MinImageHeight = 50;


#define Register_Layer(type, ext)																			\
char gInstantiationGuard##type##Layer;																\
template <typename Dtype>																			\
caffe::shared_ptr<caffe::Layer<float> > Creator_##type##Layer(const caffe::LayerParameter& param)	\
{																									\
return caffe::shared_ptr<caffe::Layer<float> >(new caffe::##type##Layer<float>(param));				\
};																								\
static caffe::LayerRegisterer<float> g_creator_f_##type(string(#type)+string(ext), Creator_##type##Layer<float>);


#define Register_Solver(type, ext)															\
  template <typename Dtype>                                                    \
  caffe::Solver<float>* Creator_##type##Solver(                                       \
      const caffe::SolverParameter& param)                                            \
  {                                                                            \
    return new caffe::type##Solver<float>(param);                                     \
  }                                                                            \
  static caffe::SolverRegisterer<float> g_creator_f_##type(string(#type)+string(ext), Creator_##type##Solver<float>);    \




Register_Layer(Input, "1");
Register_Layer(InnerProduct, "1");
Register_Layer(Dropout, "1");
Register_Layer(Convolution, "1");
Register_Layer(ReLU, "1");
Register_Layer(Pooling, "1");
Register_Layer(LRN, "1");
Register_Layer(Softmax, "1");
Register_Layer(TanH, "1");
Register_Layer(PReLU, "1");
Register_Layer(Deconvolution, "1");
Register_Layer(EuclideanLoss, "1");
Register_Solver(SGD, "1");

FaceDetector* CreateFaceDetector(char *modelfolder, int GpuID)
{


	FaceDetector *fd = new FaceDetector(modelfolder, FaceDetector::MODEL_V1, GpuID);
	return fd;
}

void FreeFaceDetector(FaceDetector *detector)
{
	if (detector != NULL)
	{
		delete detector;
		detector = NULL;

	}
}

int DetectFaces_Mat(FaceDetector *detector, cv::Mat src, double **p_ftr, int *p_Nftr, int *p_ftrArrayTotalLength, UserFaceSettings settings, DetectedFace** faces)
{
	int FtrDim = (5 + 4) * 2; //5 points + 4 corners of rectangle
	int ok = 0;
	double *outputs = NULL;
	int Noutputs = 0;

	if (p_ftr == NULL)
		return 1;

	

	try
	{
		cv::Mat image = src.clone();
		if (image.rows < MinImageWidth || image.cols < MinImageHeight)
			throw 1;

		vector<FaceDetector::BoundingBox> res = detector->Detect(image, FaceDetector::BGR, FaceDetector::ORIENT_UP, settings.sizeThresh, settings.pThresh, settings.rThresh, settings.rThresh);

		Noutputs = res.size();
		for (int i = 0; i < res.size(); i++)
		{
			if (res[i].score < settings.minConf)
			{
				Noutputs--;
			}
		}

		outputs = (double *)malloc(Noutputs * FtrDim * sizeof(double));
		if (outputs == NULL)
			goto finalize;
		int oix = 0;
		for (int i = 0; i < res.size(); i++)
		{
			if (res[i].score < settings.minConf)
			{
				continue;
			}
			for (int p = 0; p < 5; p++)
			{
				//circle(show_image, points[i][p], 2, Scalar(0, 255, 255), -1);
				outputs[oix++] = res[i].points_x[p];
				outputs[oix++] = res[i].points_y[p];
			}
			outputs[oix++] = res[i].x1;
			outputs[oix++] = res[i].y1;

			outputs[oix++] = res[i].x1;
			outputs[oix++] = res[i].y2;

			outputs[oix++] = res[i].x2;
			outputs[oix++] = res[i].y2;

			outputs[oix++] = res[i].x2;
			outputs[oix++] = res[i].y1;
		}
		if (faces != NULL)
		{
			*faces = new DetectedFace[Noutputs];
			oix = 0;
			for (int i = 0; i < res.size(); i++)
			{
				if (res[i].score < settings.minConf)
				{
					continue;
				}
				for (int p = 0; p < 5; p++)
				{
					//circle(show_image, points[i][p], 2, Scalar(0, 255, 255), -1);
					(*faces)[oix].pnts[p].x = res[i].points_x[p];
					(*faces)[oix].pnts[p].y = res[i].points_y[p];
				}
				(*faces)[oix].confidenc = res[i].score;
				(*faces)[oix].x = res[i].x1;
				(*faces)[oix].y = res[i].y1;
				(*faces)[oix].width = res[i].x2 - res[i].x1;
				(*faces)[oix].height = res[i].y2 - res[i].y1;
				oix++;
			}
		}
		ok = 1;
	}
	//catch (int ex)
	//{
	//	cout << "Exception: " << ex << endl;
	//}
	catch (...)
	{
		cout << "Exception" << endl;
	}
finalize:
	if (!ok)
	{
		if (outputs != NULL)
		{
			free(outputs);
			outputs = NULL;
		}
		if (p_ftr != NULL)
			*p_ftr = NULL;
		if (p_Nftr != NULL)
			*p_Nftr = 0;
		if (p_ftrArrayTotalLength != NULL)
			*p_ftrArrayTotalLength = 0;
		return 2;
	}
	else
	{
		if (p_ftr != NULL)
			*p_ftr = outputs;
		if (p_Nftr != NULL)
			*p_Nftr = Noutputs;
		if (p_ftrArrayTotalLength != NULL)
			*p_ftrArrayTotalLength = FtrDim * Noutputs;
	}
	return 0;

}

int DetectFaces(FaceDetector *detector, unsigned char *imagebuffer, int buffersize, double **p_ftr, int *p_Nftr, int *p_ftrArrayTotalLength)
{
	int FtrDim = (5 + 4) * 2; //5 points + 4 corners of rectangle
	int ok = 0;
	double *outputs = NULL;
	int Noutputs = 0;

	if (p_ftr == NULL)
		return 1;
	cv::Mat RawData = cv::Mat(1, buffersize, CV_8UC1, imagebuffer);
	cv::Mat image = imdecode(RawData, cv::IMREAD_COLOR);
	UserFaceSettings face_settings;

	face_settings.pThresh = 0.6f;
	face_settings.rThresh = 0.7f;
	face_settings.nmsThresh = 0.7f;
	face_settings.sizeThresh = 20;
	face_settings.minConf = 0.0;
	face_settings.facialKeisDetection = 1;

	return DetectFaces_Mat(detector, image, p_ftr, p_Nftr, p_ftrArrayTotalLength, face_settings);


}

int DetectFaces_File(FaceDetector *detector, char *imagefilepath, double **p_ftr, int *p_Nftr, int *p_ftrArrayTotalLength)
{
	FILE *fp = fopen(imagefilepath, "rb");
	fseek(fp, 0L, SEEK_END);
	long len = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	unsigned char * buffer = (unsigned char *)malloc(len);
	fread(buffer, 1, len, fp);
	fclose(fp);
	return DetectFaces(detector, buffer, len, p_ftr, p_Nftr, p_ftrArrayTotalLength);
}

void FreeResults(double *ftr)
{
	if (ftr != NULL)
	{
		free(ftr);
		ftr = NULL;
	}
}