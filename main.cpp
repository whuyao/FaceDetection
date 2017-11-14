
#include <QtCore>
#include <QTextCodec>
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h> 
#include <dlib/opencv.h>

using namespace dlib;
using namespace std;

//face_shapes[0].parts(i) i=0~67
/*
0-16 脸颊
17-21 左眉
22-26 右眉
27-30 鼻梁
31-35 鼻翼
36-41 左眼 37-38 上眼睑 40-41 下眼睑
42-47 右眼 43-44 上眼睑 46-47 下眼睑
48-59 外唇 48-54 外唇上 55-59 外唇下
60-67 内唇 60-64 内唇上 65-67 内唇下
*/
int GetFaceShapesFromImg(cv::Mat img, dlib::shape_predictor pose_model, std::vector<dlib::rectangle>& faces, std::vector<dlib::full_object_detection>& face_shapes, bool bIsShowImg = false)
{
	frontal_face_detector detector = get_frontal_face_detector();  

	//识别人脸矩形
	dlib::cv_image<bgr_pixel> cimg(img);
	faces = detector(cimg);	
	cout<<"face count = "<<faces.size()<<endl;

	//识别人脸特征点
	face_shapes.clear();
	for (int i = 0; i < faces.size(); ++i)
		face_shapes.push_back(pose_model(cimg, faces[i]));  

	//
	if (bIsShowImg)
	{
		cv::Mat show_img = img.clone();	
		int i, j;
		for (j=0; j<faces.size(); j++)	//脸的数目
		{
			cv::rectangle(show_img, cv::Rect(faces[j].left(), faces[j].top(), faces[j].width(), faces[j].height()), cv::Scalar(int(255.0/(j+1)), 0, 255), 5);
			cv::putText(show_img, QString::number(j).toStdString(), cvPoint(faces[j].left()+faces[j].width()/2, faces[j].top()-10), cv::FONT_HERSHEY_PLAIN, 5, cv::Scalar(int(255.0/(j+1)), 0, 255), 5);		
			for (int i = 0; i < 68; i++) //68个特征点			
				cv::circle(show_img, cvPoint(face_shapes[j].part(i).x(), face_shapes[j].part(i).y()), 5, cv::Scalar(int(255.0/(j+1)), 0, 255), 5); 			
		}

		int n0 = std::rand();
		cv::namedWindow(QString("face detection-%1").arg(n0).toStdString(), 0);
		imshow(QString("face detection-%1").arg(n0).toStdString(), show_img);
	}

	return faces.size();
}

/*
Features(计算两两之比例):
人脸下巴宽度d0
左眼的宽度d1
右眼的宽度d2
鼻尖与左眼连线的垂直距离d3
鼻尖与右眼连线的垂直距离d4
左右边外廓距离d5
左右眼内廓距离d6
人脸左右边界的距离d7
人脸下巴宽度d8
嘴巴外的横向宽度d9
嘴巴内的横向宽度d10
嘴巴外的横向高度d11
嘴巴内的横向高度d12
两眼中心与左嘴角水平距离d13
两眼中心与右嘴角水平距离d14
鼻翼的宽度d15
右眼的外侧眼角与鼻项的水平距离d16
左眼的内侧眼角与鼻顶的水平距离d17
嘴巴中点与鼻尖的垂直距离d18
鼻尖与嘴角的距离d19
左人脸长度d20
右人脸长度d21
下巴长度 d22
*/
double ComputeDist(double x1, double y1, double x2, double y2)
{
	double _s = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);
	return sqrt(_s);
}

double ComputeDist(dlib::full_object_detection face_shape, int id0, int id1)
{
	double _s = ComputeDist(face_shape.part(id0).x(), face_shape.part(id0).y(), face_shape.part(id1).x(), face_shape.part(id1).y());
	return _s;
}

double ComputeDist(dlib::full_object_detection face_shape, std::vector<int> ids)
{
	double _s = 0;
	for (int i=0; i<ids.size()-1; i++)
	{
		for (int j=i+1; j<ids.size(); j++)
		{
			_s += ComputeDist(face_shape, ids[0], ids[1]);
		}
	}

	return _s;
}



std::vector<double> ComputeFeaturesFromFaceShape(dlib::full_object_detection face_shape, bool bIsOriginFeatures = false)
{
	double f[23];
	memset(f, 0, 23*sizeof(double));
	std::vector<int> tmp_vals;

	// 人脸下巴宽度d0
	f[0] = ComputeDist(face_shape, 6, 10);
	// 左眼的宽度d1
	f[1] = ComputeDist(face_shape, 36, 39);
	// 右眼的宽度d2
	f[2] = ComputeDist(face_shape, 42, 45);
	// 鼻尖与左眼连线的垂直距离d3
	f[3] = ComputeDist((face_shape.part(36).x()+face_shape.part(39).x())/2, (face_shape.part(36).y()+face_shape.part(39).y())/2, face_shape.part(33).x(), face_shape.part(33).y());
	// 鼻尖与右眼连线的垂直距离d4
	f[4] = ComputeDist((face_shape.part(42).x()+face_shape.part(45).x())/2, (face_shape.part(42).y()+face_shape.part(45).y())/2, face_shape.part(33).x(), face_shape.part(33).y());
	// 左右边外廓距离d5
	f[5] = ComputeDist(face_shape, 36, 45);
	// 左右眼内廓距离d6
	f[6] = ComputeDist(face_shape, 39, 42);
	// 人脸左右边界的距离d7
	f[7] = ComputeDist(face_shape, 0, 16);
	// 人脸下巴宽度d8
	f[8] = ComputeDist(face_shape, 6, 10);
	// 嘴巴外的横向宽度d9	
	tmp_vals.clear();
	int b1[] = {48,49,50,51,52,53,54};
	tmp_vals.push_back(48);
	tmp_vals.push_back(49);
	tmp_vals.push_back(50);
	tmp_vals.push_back(51);
	tmp_vals.push_back(52);
	tmp_vals.push_back(53);
	tmp_vals.push_back(54);
	f[9] = ComputeDist(face_shape, tmp_vals);
	// 嘴巴内的横向宽度d10
	tmp_vals.clear();
	tmp_vals.push_back(55); 
	tmp_vals.push_back(56); 
	tmp_vals.push_back(57); 
	tmp_vals.push_back(58); 
	tmp_vals.push_back(59); 
	f[10] = ComputeDist(face_shape, tmp_vals);
	// 嘴巴外的横向高度d11
	f[11] = ComputeDist(face_shape, 51, 57);
	// 嘴巴内的横向高度d12
	f[12] = ComputeDist(face_shape, 62, 66);
	// 两眼中心与左嘴角水平距离d13
	f[13] = ComputeDist(face_shape, 27, 48);
	// 两眼中心与左嘴角水平距离d14
	f[14] = ComputeDist(face_shape, 27, 54);
	// 鼻翼的宽度d15
	tmp_vals.clear();
	tmp_vals.push_back(31); 
	tmp_vals.push_back(32); 
	tmp_vals.push_back(33); 
	tmp_vals.push_back(34); 
	tmp_vals.push_back(35); 
	f[15] = ComputeDist(face_shape, tmp_vals);
	// 右眼的外侧眼角与鼻项的水平距离d16
	f[16] = ComputeDist(face_shape, 45, 30);
	// 左眼的内侧眼角与鼻顶的水平距离d17
	f[17] = ComputeDist(face_shape, 39, 30);
	// 嘴巴中点与鼻尖的垂直距离d18
	f[18] = ComputeDist((face_shape.part(62).x()+face_shape.part(66).x())/2, (face_shape.part(62).y()+face_shape.part(66).y())/2, face_shape.part(30).x(), face_shape.part(30).y());
	// 鼻尖与嘴角的距离d19
	f[19] = (ComputeDist(face_shape, 30, 48)+ComputeDist(face_shape, 30, 54))/2.0f;
	// 左人脸长度
	tmp_vals.clear();
	tmp_vals.push_back(0); 
	tmp_vals.push_back(1); 
	tmp_vals.push_back(2); 
	tmp_vals.push_back(3); 
	tmp_vals.push_back(4); 
	tmp_vals.push_back(5); 
	tmp_vals.push_back(6); 
	tmp_vals.push_back(7); 
	tmp_vals.push_back(8); 
	f[20] = ComputeDist(face_shape, tmp_vals);
	// 右人脸长度d21
	tmp_vals.clear();
	tmp_vals.push_back(8); 
	tmp_vals.push_back(9); 
	tmp_vals.push_back(10); 
	tmp_vals.push_back(11); 
	tmp_vals.push_back(12); 
	tmp_vals.push_back(13); 
	tmp_vals.push_back(14); 
	tmp_vals.push_back(15); 
	tmp_vals.push_back(16); 
	f[22] = ComputeDist(face_shape, tmp_vals);
	// 下巴长度 d22
	tmp_vals.clear();
	tmp_vals.push_back(6); 
	tmp_vals.push_back(7); 
	tmp_vals.push_back(8);
	tmp_vals.push_back(9);
	tmp_vals.push_back(10);
	f[23] = ComputeDist(face_shape, tmp_vals);

	//生成特征
	int i, j;
	std::vector<double> face_features;
	face_features.clear();

	if (bIsOriginFeatures)
	{
		cout<<"<orignal features>"<<endl;
		for (i=0; i<23; i++)
			face_features.push_back(f[i]);

		//归一化
// 		double max_fea=face_features[0], min_fea=face_features[0];
// 		for (i=0; i<23; i++){
// 			max_fea = max_fea<face_features[i] ? max_fea:face_features[i];
// 			min_fea = min_fea>face_features[i] ? min_fea:face_features[i];
// 		}
// 
// 		for (i=0; i<23; i++)
// 			face_features[i] = (face_features[i]-min_fea)/(max_fea-min_fea);
	}
	else
	{
		cout<<"<relative features>"<<endl;
		for (i=0; i<22; i++)
		{
			for (j=i+1; j<23; j++)
			{
				double val = 0;
				if (f[j] > 0)
					val = f[i]/f[j];
				face_features.push_back(val);
			}
		}	
	}
	
	
	return face_features;
}

//获取图像中所有脸部特征
std::vector<std::vector<double>> GetAllFaceFeatures(std::vector<dlib::full_object_detection> face_shapes, bool bIsOriginFeatures = false)
{
	std::vector<std::vector<double>> all_features;
	all_features.clear();
	for (int i=0; i<face_shapes.size(); i++)
	{
		std::vector<double> feas = ComputeFeaturesFromFaceShape(face_shapes[i], bIsOriginFeatures);
		all_features.push_back(feas);
	}
	
	return all_features;
}

//获取图像脸部距离
double ComputeFaceFeaturesDist(std::vector<double> f1, std::vector<double> f2, bool bIsEulDist = false)
{
	if (f1.size() != f2.size())
		return -999;
	
	int i=0;

	if (bIsEulDist)
	{
		double dist = 0;
		for (i=0; i<f1.size(); i++)
		{
			dist += (f1[i]-f2[i])*(f1[i]-f2[i]);		
		}

		return sqrt(dist);
	}
	else
	{
		double sum = 0;
		double sum1_2 = 0;
		double sum2_2 = 0;
		for(i=1; i<f1.size(); ++i)  
		{  
			sum += f1[i]*f2[i];
			sum1_2 += f1[i]*f1[i];
			sum2_2 += f2[i]*f2[i];
		}  

		return 1-sum/(sqrt(sum1_2)*sqrt(sum2_2));
	}	
}


//输出特征测试
class FaceFeature
{
public:
	FaceFeature(){id0=0; id1=0; dist=0;};
	~FaceFeature(){};

	int id0,id1;
	double dist;

	bool operator<(FaceFeature& ff){
		if (this->dist < ff.dist)
			return true;
		else
			return false;
	}


};

//filename1, filename2 输入文件名
//bIsShowImg 是否显示标记图像
//bIsOriginFeatures 是否使用原始绝对特征/相对特征
//bIsEulDist 是否使用欧式距离/余弦距离
void test2pic(const char* filename1, const char* filename2, bool bIsShowImg = true, bool bIsOriginFeatures = false, bool bIsEulDist=false)
{
	cv::Mat mat1 = cv::imread(filename1);
	cv::Mat mat2 = cv::imread(filename2);
	
	dlib::shape_predictor pose_model;  
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	cout<<"load face model success."<<endl;

	std::vector<dlib::rectangle> faces1;
	std::vector<dlib::full_object_detection> shapes1;
	int face_size1 = GetFaceShapesFromImg(mat1, pose_model, faces1, shapes1, bIsShowImg);
	cout<<"Face count of img1 = "<<face_size1<<endl;

	std::vector<dlib::rectangle> faces2;
	std::vector<dlib::full_object_detection> shapes2;
	int face_size2 = GetFaceShapesFromImg(mat2, pose_model, faces2, shapes2, bIsShowImg);
	cout<<"Face count of img2 = "<<face_size2<<endl;

	std::vector<std::vector<double>> vf1, vf2;
	cout<<"computing image 1 face features..."<<endl;
	vf1 = GetAllFaceFeatures(shapes1, bIsOriginFeatures);
	cout<<"computing image 2 face features..."<<endl;
	vf2 = GetAllFaceFeatures(shapes2, bIsOriginFeatures);

	//快排
	std::vector<FaceFeature> ffs;
	for (int i=0; i<face_size1; i++)
	{
		for (int j=0; j<face_size2; j++)
		{
			double dist = ComputeFaceFeaturesDist(vf1[i], vf2[j], bIsEulDist);
			FaceFeature ff;
			ff.id0 = i; ff.id1 = j;
			ff.dist = dist;
			ffs.push_back(ff);
 		}
			
		
	}

	std::sort(ffs.begin(), ffs.end());

	//输出
	cout<<"**************Last time****************"<<endl;
	for (int i=0; i<ffs.size(); i++)
	{
		cout<<"Face_Dist [Img1."<<ffs[i].id0<<"][Img2."<<ffs[i].id1<<"] = "<<ffs[i].dist<<endl;
	}

}

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	srand(time(NULL));
	//set code for locale, support chinese
	QTextCodec* codec =QTextCodec::codecForLocale();
	QTextCodec::setCodecForCStrings(codec);
	QTextCodec::setCodecForTr(codec);

	//filename1, filename2 输入文件名
	//bIsShowImg 是否显示标记图像
	//bIsOriginFeatures 是否使用原始绝对特征/相对特征
	//bIsEulDist 是否使用欧式距离/余弦距离
	test2pic("./lxp0.jpg", "./lxp5.jpg", true, false, true);
	cv::waitKey(0);
	return 1;

	/*
	
	//载入模型 
	shape_predictor pose_model;  
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	cout<<"load face model success."<<endl;

	//载入图片
	cv::Mat img = cv::imread("./lxp2.jpg");
	if (img.empty())
	{
		cout<<"load file error."<<endl;
		return -1;
	}

	//封装函数
	std::vector<full_object_detection> shapes;  
	std::vector<dlib::rectangle> faces;
	int face_size = GetFaceShapesFromImg(img, pose_model, faces, shapes);

	//生成特征数据
	int i, j;
	for (i=0; i<faces.size(); i++)
	{
		cout<<"Face ["<<i<<"] = "<<endl;
		std::vector<double> feas = ComputeFeaturesFromFaceShape(shapes[i]);
		for (j=0; j<feas.size(); j++)
			cout<<feas[j]<<", ";
		cout<<endl;
	}

	*/
	//绘制人脸特征点
	/*
	cv::Mat show_img = img.clone();	
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	for (j=0; j<faces.size(); j++)	//脸的数目
	{
		cv::rectangle(show_img, cv::Rect(faces[j].left(), faces[j].top(), faces[j].width(), faces[j].height()), cv::Scalar(int(255.0/(j+1)), 0, 255), 1);
		for (int i = 0; i < 68; i++) //68个特征点
		{ 			
			cv::circle(show_img, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 1, cv::Scalar(int(255.0/(j+1)), 0, 255), 1); 			
			//cv::addText(show_img, QString::number(i).toStdString(), cvPoint(shapes[j].part(i).x()-3, shapes[j].part(i).y()+3), font);
			cv::putText(show_img, QString::number(i).toStdString(), cvPoint(shapes[j].part(i).x()-3, shapes[j].part(i).y()+3), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(int(255.0/(j+1)), 0, 255));
		}  
	}
		
	cv::namedWindow("face detection", 1);
	imshow("face detection", show_img);
	imwrite("save_img.jpg", show_img);

	//等待
	cv::waitKey(0);
	*/

	

	

	//return a.exec();
}
