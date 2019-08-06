#include <stdio.h>
//#include <tchar.h>
//#include <windows.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <cctype>
#include <memory>

#include <opencv2/opencv.hpp>

#include "../../../s/timer_hd.h"

#include "PuRe.h"
#include "ExCuSe.h"
#include "pupil-tracking/PuReST.h"

/*
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"

*/

#ifdef _DEBUG
#pragma comment(lib, "/opencv41/build/x64/vc15/lib/opencv_world411d.lib")
#else
#pragma comment(lib, "/opencv41/build/x64/vc15/lib/opencv_world411.lib")
#endif

using namespace std;
using namespace cv;


// evaluate the best parameter set over ALL images of the LPW dataset
void lpw_test_all()
{
	using namespace std;

	//PuRe pupil_detect;
	ExCuSe pupil_detect;
	//PupilTrackingMethod* pupil_tracking = new PuReST;
	Pupil pupil, pupil_previous;


	// labeled pupils in the wild - load all file names
	vector<string> fnames;

	//fstream f("c:/datasets/LPW/subselection.txt", ios::in); 
	fstream f("c:/datasets/LPW/all.txt", ios::in);
	while (f.is_open() && !f.eof()) { string fname;  f >> fname; fnames.push_back(fname); }
	cv::Mat frame, frame_gray;

	int frame_increment = 100;
	cout << "enter frame increment (1..n): "; cin >> frame_increment; cout << "\n";

	fstream f2; f2.open("images_all_errors_best_paramset.txt", ios::out);
	Timer timer;
	vector<double> cpu_time;
	cv::Rect roi(0, 0, 640, 480);
	//cv::Rect roi(0, 0, 320, 200);
	
	for (auto fname : fnames)
	{
		// load ground truth positions
		vector<cv::Point2f> ground_truth_pos;
		string fname2 = fname;
		fname2.replace(fname2.size() - 3, 3, "txt");
		fstream f(fname2, ios::in); while (f.is_open() && !f.eof()) { cv::Point2f p; f >> p.x >> p.y; ground_truth_pos.push_back(p); }

		auto capture = new cv::VideoCapture(fname);
		int k = 0;
		float mean_error = 0.0f;
		cv::Mat frame_gray;
		while (capture->read(frame))
		{
			cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

			timer.tick();
			//pupil_detect.run(frame_gray, pupil);
			pupil = pupil_detect.run(frame_gray);
			
			// PuRest
			//pupil_previous = pupil;
			//pupil_tracking->run(k, frame_gray, roi, pupil, pupil_detect);
			

			double dt = timer.tock();
			cpu_time.push_back(dt);

			auto p = pupil.center;
			auto error = cv::norm(ground_truth_pos[k] - p);
			mean_error += error;
			f2 << error << " ";

			k += frame_increment; // skip frames
			capture->set(cv::CAP_PROP_POS_FRAMES, k);

		}
		mean_error = mean_error / (float(k)/float(frame_increment));
		cout << "avi processed: " << fname << " mean error: " << mean_error << endl;
	}
	f2.close();

	f2.open("cpu_time.txt", ios::out);
	for (auto x : cpu_time) { f2 << x << " "; }
	f2.close();
}


shared_ptr<cv::VideoCapture> select_camera(string message)
{
	cout << "\n=== Menu Camera Selection ===\n";

	shared_ptr<cv::VideoCapture> capture = nullptr;
	while (true)
	{
		cout << message;
		int cam_nr = 0;
		cin >> cam_nr;
		capture = make_shared< cv::VideoCapture >(cv::VideoCapture(cam_nr));
		if (capture->isOpened()) { break; }
		cerr << "\ncould not open and initialize camera nr. " << cam_nr << ". please try again!\n";
	}

	// TODO: dialog for webcam resolution entry
	//capture->set(cv::CAP_PROP_FRAME_WIDTH, width);
	//capture->set(cv::CAP_PROP_FRAME_HEIGHT, height);

	return capture;
}


void run_webcam()
{
	// generate a random image
	using namespace cv;
	auto img_rand = Mat(480, 640, CV_8UC3);
	randu(img_rand, Scalar::all(0), Scalar::all(255));

	PuRe pupil_detect;
	Pupil pupil;


	cv::Mat frame;
	cv::Mat frame_gray;

	auto capture = select_camera("select eye camera:");

	Timer timer(100);
	while (true)
	{


		// read a frame from the camera
		capture->read(frame);

		// Apply the classifier to the frame
		if (!frame.empty())
		{
			timer.tick();
			cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
			pupil_detect.run(frame_gray, pupil);
			auto p = pupil.center;
			timer.tock();

			cv::circle(frame, p, 4, Scalar(255, 0, 255), 2);
			imshow("pupil", frame);
			cv::waitKey(1);
		}
	}
}


int main(int argc, const char** argv)
{

	setUseOptimized(true);

	//lpw_test_all();
	run_webcam();
	return 1;
}

