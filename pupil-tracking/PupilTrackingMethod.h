#ifndef PUPILTRACKINGMETHOD_H
#define PUPILTRACKINGMETHOD_H

#include <string>
#include <deque>
//#include <QFuture>

#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"

#include "../PupilDetectionMethod.h"

//#include "utils.h"

class TrackedPupil : public Pupil
{
public:
	TrackedPupil(const int ts_, const Pupil &pupil) :
		Pupil(pupil),
		ts(ts)
	{}

	TrackedPupil() :
		Pupil(),
		ts(0)
	{}

	int ts = 0;
};

class PupilTrackingMethod
{
public:
	PupilTrackingMethod() {
		pupilDiameterKf.init(1, 1);
		pupilDiameterKf.transitionMatrix = ( cv::Mat_<float>(1, 1) << 1 );
		cv::setIdentity( pupilDiameterKf.measurementMatrix );
		cv::setIdentity( pupilDiameterKf.processNoiseCov, cv::Scalar::all(1e-4) );
		cv::setIdentity( pupilDiameterKf.measurementNoiseCov, cv::Scalar::all(1e-2) );
		cv::setIdentity( pupilDiameterKf.errorCovPost, cv::Scalar::all(1e-1) );
	}
	~PupilTrackingMethod() {}

	// Tracking and detection logic
	void run(const int ts_, const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, PupilDetectionMethod &pupilDetectionMethod);

	// Tracking implementation
	virtual void run(const cv::Mat &frame, const cv::Rect &roi, const Pupil &previousPupil, Pupil &pupil, const float &minPupilDiameterPx=-1, const float &maxPupilDiameterPx=-1) = 0;

	std::string description() { return mDesc; }

private:

protected:
	std::string mDesc;

	cv::Size expectedFrameSize = {0, 0};
	TrackedPupil previousPupil;
	std::deque<TrackedPupil> previousPupils;

	int maxAge = 1500;
	int maxTrackingWithoutDetectionTime = 5000;
	int lastDetection;
	bool parallelDetection = false;
	float minDetectionConfidence = 0.7f;
	float minTrackConfidence = 0.9f;

	cv::KalmanFilter pupilDiameterKf;
	float predictedMaxPupilDiameter = -1;

	void predictMaxPupilDiameter();
	void registerPupil(const int ts, Pupil &pupil);

	void reset();
};

#endif // PUPILTRACKINGMETHOD_H
