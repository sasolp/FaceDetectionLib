#ifndef NEWFACEDETECTION_WRAPPER_H
#define NEWFACEDETECTION_WRAPPER_H
typedef struct _Point
{
	int x;
	int y;
}_Point;
typedef struct _DetectedFace
{
	int x;
	int y;
	int width;
	int height;
	float confidenc;
	_Point pnts[5];
}DetectedFace;
typedef struct _FaceCase
{
	cv::Scalar color;
	cv::Rect bounding_rect;
	_Point pnts_1;
	_Point pnts_2;
	_Point pnts_3;
	_Point pnts_4;
	_Point pnts_5;
	string id;
	string face_img_path;
	string frame_img_path;
	int index;
	int is_lost;
}FaceCase;
typedef struct _UserFaceSettings
{
	float pThresh;
	float rThresh;
	float nmsThresh;
	int sizeThresh;
	float minConf;
	int facialKeisDetection;
}UserFaceSettings;
class FaceDetector;
FaceDetector* CreateFaceDetector(char *modelfolder, int GpuID);
void FreeFaceDetector(FaceDetector *detector);
int DetectFaces_Mat(FaceDetector *detector, cv::Mat src, double **p_ftr, int *p_Nftr, int *p_ftrArrayTotalLength, UserFaceSettings settings, DetectedFace** faces = 0);
int DetectFaces(FaceDetector *detector, unsigned char *imagebuffer, int buffersize, double **p_ftr, int *p_Nftr, int *p_ftrArrayTotalLength);
int DetectFaces_File(FaceDetector *detector, char *imagefile, double **p_ftr, int *p_Nftr, int *p_ftrDim);
void FreeResults(double *ftr);
#endif