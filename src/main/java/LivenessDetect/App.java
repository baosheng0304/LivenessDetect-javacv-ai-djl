package LivenessDetect;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class App {
	private static final Logger logger = LoggerFactory.getLogger(LivenessDetector.class);
	private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
	private static final OpenCVFrameConverter.ToMat toJavaCvMat = new OpenCVFrameConverter.ToMat();
	private static LivenessDetector livenessDetector;
	private static FaceDetector faceDetector;
	private static String modelDir;
	private static String option;
	private static String inputPath;
    private static volatile Frame[] tempVideoFrame = new Frame[1];
    private static volatile Mat[] mRgbFrames = new Mat[1];
    private static volatile boolean mStop = false;
    
	private static Mat mRgbFrame = new Mat();
	private static Mat mMaskImg = new Mat();
	private static Mat mCroppedRgb = new Mat();
	private static Mat mBluredFrame = new Mat();
	private static Mat mBluredGray = new Mat();
	private static Mat mCroppedMat = new Mat();
	private static int mPreviewSize = 0;
	
	static int mEllipseCenterX, mEllipseCenterY, mEllipseSizeX, mEllipseSizeY;
	

	public static void main(String[] args) throws IOException, MalformedModelException {

		if (args.length < 3) {
			System.out.println("Insufficient number of argument!");
			DislayUsage();
			return;
		}
		modelDir = args[0];
		option = args[1];
		inputPath = args[2];
		if (!option.equals("image") && !option.equals("video") && !option.equals("ffmpeg")) {
			System.out.println("Invaild option parameter!");
			DislayUsage();
			return;
		}
    	System.out.println("model dir = " + modelDir);
    	logger.info("model dir = " + modelDir);
    	
		livenessDetector = new LivenessDetector(modelDir);
		if (!livenessDetector.init()) {
			System.out.println("Failed to initialize liveness detection engine!");
			return;
		}else {
			System.out.println("Success to initialize liveness detection engine!");
		}
		
		faceDetector = new FaceDetector(modelDir);
		if (!faceDetector.init()) {
			System.out.println("Failed to initialize face detection engine!");
			return;
		}else {
			System.out.println("Success to initialize face detection engine!");
		}
		
		if (option.equals("image"))
		{
			DetectLivenessFromOneImage(inputPath);
		}else if (option.equals("video"))
		{
			if (isInteger(inputPath)) {
				int deviceNum = Integer.parseInt(inputPath);
				DetectLivenessFromCamera(deviceNum);
			}
			else {
				File f = new File(inputPath);
				if (!f.exists()) {
					System.out.println("Invalid video file, check if it exists.. " + inputPath);
					logger.debug("Invalid image file, check if it exists.. " + inputPath);
					return;
				}
				if(f.isDirectory()) { 
					System.out.println("Please specify full path of video file, not directory, " + inputPath);
					logger.debug("Please specify full path of video file, not directory, " + inputPath);
					return;
				}
				DetectLivenessFromVideo(inputPath);
			}
		}else if (option.equals("ffmpeg"))
		{
			if (isInteger(inputPath)) {
				int deviceNum = Integer.parseInt(inputPath);
				DetectLivenessFromCameraFfmpeg(deviceNum);
			}else {
				DetectLivenessFromVideoFfmpeg(inputPath);
			}
		}
	}

	private static boolean isInteger(String strNum) {
	    if (strNum == null) {
	        return false;
	    }
	    try {
	        Integer.parseInt(strNum);
	    } catch (NumberFormatException nfe) {
	        return false;
	    }
	    return true;
	}
	
	private static void DislayUsage()
	{
		System.out.println("Usage: LivenessDetector.jar modelDir option inputpath");
		System.out.println("	Parameters:		");
		System.out.println("		modelDir: absolute path of folder that contains ML models and resource file");
		System.out.println("			Model folder must contains 4 files");
		System.out.println("				deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel, liveness_model.pt, face_mask.png");
		System.out.println("		option: input option, there are 3 possible values - image, video, ffmpeg");
		System.out.println("			image: read image");
		System.out.println("			video: read video frame  from video file or web camera");
		System.out.println("			ffmpeg: read video frame  from video file or web camera by using internal ffmpeg decoder");
		System.out.println("		inputpath: absolute path of input image or video. ");
		System.out.println("			if this parameter is number, open web camera of specified number");
		System.out.println("Example: ");
		System.out.println("	java –jar LivenessDetect-all.jar /home/maxim/model image /home/maxim/model/2dmask_2.png");
		System.out.println("	java –jar LivenessDetect-all.jar /home/maxim/model video /home/maxim/model/Mask_Half_44.mp4");
		System.out.println("	java –jar LivenessDetect-all.jar /home/maxim/model video 0");
		System.out.println("		this open web camera and capture frame");
		System.out.println("	java –jar LivenessDetect-all.jar /home/maxim/model ffmpeg /home/maxim/model/Mask_Half_44.mp4");
		System.out.println("		this open mask_attack.mp4 video file and decode frame with own ffmpeg decoder");
		System.out.println("	java –jar LivenessDetect-all.jar /home/maxim/model ffmpeg 0");
		System.out.println("		this open web camera and capture frame with own ffmpeg decoder");
		return;
	}
	
	private static void DetectLivenessFromOneImage(String imagePath) throws IOException
	{
		File f = new File(imagePath);
		if (!f.exists()) {
			System.out.println("Invalid image file, check if it exists.. " + imagePath);
			logger.debug("Invalid image file, check if it exists.. " + imagePath);
			return;
		}
		if(f.isDirectory()) { 
			System.out.println("Please specify full path of image file, not directory, " + imagePath);
			logger.debug("Please specify full path of image file, not directory, " + imagePath);
			return;
		}
		System.out.println("Reading image... " + imagePath);
		Mat frame = imread(imagePath);
		boolean result = DetectLivenessFromFrame(frame);
		System.out.println(result? "Live!" : "Fake!");
	}

	private static boolean DetectLivenessFromFrame(Mat frame) throws IOException
	{
		BufferedImage bi = Java2DFrameUtils.toBufferedImage(frame);
		Image img = BufferedImageFactory.getInstance().fromImage(bi);
		boolean liveness  = livenessDetector.detectLiveness(img);
		return liveness;
	}

	// This function does not work well on Ubuntu18.04
	// due to capture.open(video_path) fails
	// Use DetectLivenessFromCameraFfmpeg() function on Ubuntu
	private static void DetectLivenessFromCamera(int deviceNum) throws IOException
	{
		String maskPath = modelDir + "/" + "face_mask.png";
		File ff = new File(maskPath);
		if (!ff.exists()) {
			System.out.println("Invalid mask image path, " + maskPath);
			logger.debug("Invalid mask image path, " + maskPath);
			return;
		}
		
		VideoCapture capture = new VideoCapture(deviceNum);
		if (!capture.isOpened()) 
		{
			System.out.println("Can not open the camera !");
			capture.close();
			return;
		}

		CanvasFrame mainframe = new CanvasFrame("Face Detection", CanvasFrame.getDefaultGamma() / 2.2);
		mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
		mainframe.setLocationRelativeTo(null);
		mainframe.setVisible(true);
		Mat sample_frame = new Mat();
		capture.read(sample_frame);
		int previewWidth = sample_frame.cols(); 
		int previewHeight = sample_frame.rows();
//		Size img_size = new Size(sample_frame.cols(), sample_frame.rows());
		if (previewHeight > previewWidth){
			mPreviewSize = previewWidth;
		}else{
			mPreviewSize = previewHeight;
		}
		mainframe.setCanvasSize(mPreviewSize, mPreviewSize);

		Mat face_mask = imread(maskPath);
		resize(face_mask, face_mask, new Size(mPreviewSize, mPreviewSize));
		mMaskImg = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		cvtColor(face_mask, mMaskImg, COLOR_BGR2GRAY);
		mBluredFrame = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mBluredGray = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		mCroppedMat = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mEllipseCenterX = (int) (mPreviewSize * 0.5);
		mEllipseCenterY = (int) (mEllipseCenterX * 0.9);
		mEllipseSizeX = (int) (mPreviewSize * 0.225);
		mEllipseSizeY = (int) (mEllipseSizeX * 1.285);
		
		while (capture.read(mRgbFrame) && mainframe.isVisible()) {
			// crop the frame as square one
			if (previewHeight > previewWidth){
				int delta = previewHeight - previewWidth;
				Rect rt0 = new Rect(0, delta/2, previewWidth, previewWidth);
				mCroppedRgb = new Mat(mRgbFrame, rt0);
			}else{
				int delta = previewWidth - previewHeight;
				Rect rt0 = new Rect(delta/2, 0, previewHeight, previewHeight);
				mCroppedRgb = new Mat(mRgbFrame, rt0);
			}
			// copy mCroppedRgb to mCroppedMat for liveness detection
			mCroppedRgb.copyTo(mCroppedMat);
			cvtColor(mCroppedRgb, mBluredGray, COLOR_BGR2GRAY);
			cvtColor(mBluredGray, mBluredFrame, COLOR_GRAY2BGR );
//			Imgproc.GaussianBlur(mRgbFrame, mBluredFrame, new Size(15, 15), 11.0);
			mBluredFrame.copyTo(mCroppedRgb, mMaskImg);
            putText(mCroppedRgb, "Please fit face in ellipse area... ",
            		new Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.7,
    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
					new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
					new Scalar(255, 0, 255, 0) , 2, LINE_8, 0);
			//faceDetector.getFaceBoxes(mCroppedRgb, 0.9f, true);
			faceBox facebox = faceDetector.extract_facebox(mCroppedRgb, 0.8f, true);
			if (facebox ==null) continue;
			boolean isFitInEllipse = checkfit(facebox);
			String resultString = "";
			if (isFitInEllipse) {
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(255, 0, 0, 0) , 2, LINE_8, 0);
				//int width = facebox.x_right - facebox.x_left;
				//int height = facebox.y_bottom - facebox.y_top;
				//Rect rectCrop = new Rect(facebox.x_left, facebox.y_top, width, height);
				//Mat face_img = new Mat(mCroppedMat, rectCrop);
				boolean liveness = DetectLivenessFromFrame(mCroppedMat);
				if (liveness) {
					resultString = "Real";
				}else {
					resultString = "Fake";
				}
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(0, 255, 0, 0) , 2, LINE_8, 0);
			}else {
				resultString = "Fit Face!";
			}
			
			putText(mCroppedRgb, resultString,
	            		new Point(mPreviewSize / 2 - 50, mPreviewSize - 30), CV_FONT_HERSHEY_SIMPLEX, 1,
	    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			mainframe.showImage(converter.convert(mCroppedRgb));
			//mainframe.validate();
			try {
				Thread.sleep(30);
			} catch (InterruptedException ex) {
				System.out.println(ex.getMessage());
			}
		}//while (capture.read(colorimg) && mainframe.isVisible()) 
		capture.close();
	}
	
	// This function does not work well on Ubuntu18.04
	// due to capture.open(video_path) fails
	// Use DetectLivenessFromVideoFfmpeg() function on Ubuntu
	private static void DetectLivenessFromVideo(String video_path) throws IOException
	{
		String maskPath = modelDir + "/" + "face_mask.png";
		File ff = new File(maskPath);
		if (!ff.exists()) {
			System.out.println("Invalid mask image path, " + maskPath);
			logger.debug("Invalid mask image path, " + maskPath);
			return;
		}
		
		VideoCapture capture = new VideoCapture();
		if (!capture.open(video_path)) {
			System.out.println("Can not open the video...");
			System.out.println("	Invailid video file or unsupported vide codec !");
			capture.close();
			return;
		}

		CanvasFrame mainframe = new CanvasFrame("Face Detection", CanvasFrame.getDefaultGamma() / 2.2);
		mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
		mainframe.setLocationRelativeTo(null);
		mainframe.setVisible(true);
		Mat sample_frame = new Mat();
		capture.read(sample_frame);
		int previewWidth = sample_frame.cols(); 
		int previewHeight = sample_frame.rows();
//		Size img_size = new Size(sample_frame.cols(), sample_frame.rows());
		if (previewHeight > previewWidth){
			mPreviewSize = previewWidth;
		}else{
			mPreviewSize = previewHeight;
		}
		mainframe.setCanvasSize(mPreviewSize, mPreviewSize);

		Mat face_mask = imread(maskPath);
		resize(face_mask, face_mask, new Size(mPreviewSize, mPreviewSize));
		mMaskImg = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		cvtColor(face_mask, mMaskImg, COLOR_BGR2GRAY);
		mBluredFrame = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mBluredGray = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		mCroppedMat = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mEllipseCenterX = (int) (mPreviewSize * 0.5);
		mEllipseCenterY = (int) (mEllipseCenterX * 0.9);
		mEllipseSizeX = (int) (mPreviewSize * 0.225);
		mEllipseSizeY = (int) (mEllipseSizeX * 1.285);
		
		while (capture.read(mRgbFrame) && mainframe.isVisible()) {
			// crop the frame as square one
			if (previewHeight > previewWidth){
				int delta = previewHeight - previewWidth;
				Rect rt0 = new Rect(0, delta/2, previewWidth, previewWidth);
				mCroppedRgb = new Mat(mRgbFrame, rt0);
			}else{
				int delta = previewWidth - previewHeight;
				Rect rt0 = new Rect(delta/2, 0, previewHeight, previewHeight);
				mCroppedRgb = new Mat(mRgbFrame, rt0);
			}
			// copy mCroppedRgb to mCroppedMat for liveness detection
			mCroppedRgb.copyTo(mCroppedMat);
			cvtColor(mCroppedRgb, mBluredGray, COLOR_BGR2GRAY);
			cvtColor(mBluredGray, mBluredFrame, COLOR_GRAY2BGR );
//			Imgproc.GaussianBlur(mRgbFrame, mBluredFrame, new Size(15, 15), 11.0);
			mBluredFrame.copyTo(mCroppedRgb, mMaskImg);
            putText(mCroppedRgb, "Please fit face in ellipse area... ",
            		new Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.7,
    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
					new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
					new Scalar(255, 0, 255, 0) , 2, LINE_8, 0);
			//faceDetector.getFaceBoxes(mCroppedRgb, 0.9f, true);
			faceBox facebox = faceDetector.extract_facebox(mCroppedRgb, 0.8f, true);
			if (facebox ==null) continue;
			boolean isFitInEllipse = checkfit(facebox);
			String resultString = "";
			if (isFitInEllipse) {
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(255, 0, 0, 0) , 2, LINE_8, 0);
				//int width = facebox.x_right - facebox.x_left;
				//int height = facebox.y_bottom - facebox.y_top;
				//Rect rectCrop =new Rect(facebox.x_left, facebox.y_top, width, height);
				//Mat face_img = new Mat(mCroppedMat, rectCrop);
				boolean liveness = DetectLivenessFromFrame(mCroppedMat);
				if (liveness) {
					resultString = "Real";
				}else {
					resultString = "Fake";
				}
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(0, 255, 0, 0) , 2, LINE_8, 0);
			}else {
				resultString = "Fit Face!";
			}
			
			putText(mCroppedRgb, resultString,
	            		new Point(mPreviewSize / 2 - 50, mPreviewSize - 30), CV_FONT_HERSHEY_SIMPLEX, 1,
	    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			mainframe.showImage(converter.convert(mCroppedRgb));
			try {
				Thread.sleep(30);
			} catch (InterruptedException ex) {
				System.out.println(ex.getMessage());
			}
		}//while (capture.read(colorimg) && mainframe.isVisible()) 
		capture.close();
	}
	
	private static void DetectLivenessFromCameraFfmpeg(int deviceNum) throws IOException
	{
		FrameGrabber grabber = FrameGrabber.createDefault(deviceNum);
		Frame tempVideoFrame;
		Mat sample_frame;
		int previewWidth =480, previewHeight =480;
		try {
			grabber.start();
			if (grabber.grab() != null) {
				tempVideoFrame = grabber.grab();
				sample_frame = toJavaCvMat.convert(tempVideoFrame);
				previewWidth = sample_frame.cols(); 
				previewHeight = sample_frame.rows();
			}

			grabber.stop();
			grabber.release();
		} catch (FrameGrabber.Exception e) {
			e.printStackTrace();
			return;
		} 

		CanvasFrame mainframe = new CanvasFrame("Face Detection", CanvasFrame.getDefaultGamma() / 2.2);
		mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
		mainframe.setLocationRelativeTo(null);
		mainframe.setVisible(true);

		if (previewHeight > previewWidth){
			mPreviewSize = previewWidth;
		}else{
			mPreviewSize = previewHeight;
		}
		mainframe.setCanvasSize(mPreviewSize, mPreviewSize);
		String maskPath = modelDir + "/" + "face_mask.png";
		File ff = new File(maskPath);
		if (!ff.exists()) {
			System.out.println("Invalid mask image path, " + maskPath);
			logger.debug("Invalid mask image path, " + maskPath);
			return;
		}
		Mat face_mask = imread(maskPath);
		resize(face_mask, face_mask, new Size(mPreviewSize, mPreviewSize));
		mMaskImg = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		cvtColor(face_mask, mMaskImg, COLOR_BGR2GRAY);
		mBluredFrame = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mBluredGray = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		mCroppedMat = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mEllipseCenterX = (int) (mPreviewSize * 0.5);
		mEllipseCenterY = (int) (mEllipseCenterX * 0.9);
		mEllipseSizeX = (int) (mPreviewSize * 0.225);
		mEllipseSizeY = (int) (mEllipseSizeX * 1.285);
		
		grabber.start();
		while (grabber.grab() != null && mainframe.isVisible()) {
			tempVideoFrame = grabber.grab();
			mRgbFrame = toJavaCvMat.convert(tempVideoFrame);
			if (mRgbFrame==null) continue;
			// crop the frame as square one
			if (previewHeight > previewWidth){
				int delta = previewHeight - previewWidth;
				Rect rt0 = new Rect(0, delta/2, previewWidth, previewWidth);
				mCroppedRgb = new Mat(mRgbFrame, rt0);
			}else{
				int delta = previewWidth - previewHeight;
				Rect rt0 = new Rect(delta/2, 0, previewHeight, previewHeight);
				mCroppedRgb = new Mat(mRgbFrame, rt0);
			}
			// copy mCroppedRgb to mCroppedMat for liveness detection
			mCroppedRgb.copyTo(mCroppedMat);
			cvtColor(mCroppedRgb, mBluredGray, COLOR_BGR2GRAY);
			cvtColor(mBluredGray, mBluredFrame, COLOR_GRAY2BGR );
//			Imgproc.GaussianBlur(mRgbFrame, mBluredFrame, new Size(15, 15), 11.0);
			mBluredFrame.copyTo(mCroppedRgb, mMaskImg);
            putText(mCroppedRgb, "Please fit face in ellipse area... ",
            		new Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.7,
    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
					new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
					new Scalar(255, 0, 255, 0) , 2, LINE_8, 0);
			//faceDetector.getFaceBoxes(mCroppedRgb, 0.9f, true);
			faceBox facebox = faceDetector.extract_facebox(mCroppedRgb, 0.8f, true);
			if (facebox ==null) continue;
			boolean isFitInEllipse = checkfit(facebox);
			String resultString = "";
			if (isFitInEllipse) {
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(255, 0, 0, 0) , 2, LINE_8, 0);
				//int width = facebox.x_right - facebox.x_left;
				//int height = facebox.y_bottom - facebox.y_top;
				//Rect rectCrop = new Rect(facebox.x_left, facebox.y_top, width, height);
				//Mat face_img = new Mat(mCroppedMat, rectCrop);
				boolean liveness = DetectLivenessFromFrame(mCroppedMat);
				if (liveness) {
					resultString = "Real";
				}else {
					resultString = "Fake";
				}
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(0, 255, 0, 0) , 2, LINE_8, 0);
			}else {
				resultString = "Fit Face!";
			}
			
			putText(mCroppedRgb, resultString,
	            		new Point(mPreviewSize / 2 - 50, mPreviewSize - 30), CV_FONT_HERSHEY_SIMPLEX, 1,
	    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			mainframe.showImage(converter.convert(mCroppedRgb));
			try {
				Thread.sleep(30);
			} catch (InterruptedException ex) {
				System.out.println(ex.getMessage());
			}
		}//while (grabber.grab() != null && mainframe.isVisible()) 
		grabber.stop();
		grabber.release();
	}
	
	private static void DetectLivenessFromVideoFfmpeg(String videoPath) throws IOException
	{
		File f = new File(videoPath);
		if (!f.exists()) {
			System.out.println("Invalid video file, check if it exists.. " + videoPath);
			logger.debug("Invalid video file, check if it exists.. " + videoPath);
			return;
		}
		if(f.isDirectory()) { 
			System.out.println("Please specify full path of video file, not directory, " + videoPath);
			logger.debug("Please specify full path of video file, not directory, " + videoPath);
			return;
		}
		
		String maskPath = modelDir + "/" + "face_mask.png";
		File ff = new File(maskPath);
		if (!ff.exists()) {
			System.out.println("Invalid mask image path, " + maskPath);
			logger.debug("Invalid mask image path, " + maskPath);
			return;
		}
		
		FFmpegFrameGrabber VIDEO_GRABBER = new FFmpegFrameGrabber(videoPath);
		Mat sample_frame;
		int previewWidth =480, previewHeight =480;
		try {
			VIDEO_GRABBER.start();
			tempVideoFrame[0] = VIDEO_GRABBER.grabImage();
			if (tempVideoFrame[0] != null) {
				sample_frame = toJavaCvMat.convert(tempVideoFrame[0]);
				previewWidth = sample_frame.cols(); 
				previewHeight = sample_frame.rows();
			}

			VIDEO_GRABBER.stop();
			VIDEO_GRABBER.release();
		} catch (FrameGrabber.Exception e) {
			e.printStackTrace();
			VIDEO_GRABBER.close();
			return;
		} 

		CanvasFrame mainframe = new CanvasFrame("Face Detection", CanvasFrame.getDefaultGamma() / 2.2);
		mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
		mainframe.setLocationRelativeTo(null);
		mainframe.setVisible(true);

		if (previewHeight > previewWidth){
			mPreviewSize = previewWidth;
		}else{
			mPreviewSize = previewHeight;
		}
		mainframe.setCanvasSize(mPreviewSize, mPreviewSize);

		Mat face_mask = imread(maskPath);
		resize(face_mask, face_mask, new Size(mPreviewSize, mPreviewSize));
		mMaskImg = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		cvtColor(face_mask, mMaskImg, COLOR_BGR2GRAY);
		mBluredFrame = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mBluredGray = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC1);
		mCroppedMat = new Mat(new Size(mPreviewSize, mPreviewSize), CV_8UC3);
		mEllipseCenterX = (int) (mPreviewSize * 0.5);
		mEllipseCenterY = (int) (mEllipseCenterX * 0.9);
		mEllipseSizeX = (int) (mPreviewSize * 0.225);
		mEllipseSizeY = (int) (mEllipseSizeX * 1.285);
		
		VIDEO_GRABBER.start();
		while (!mStop && mainframe.isVisible()) {
			tempVideoFrame[0] = VIDEO_GRABBER.grabImage();
		    if (tempVideoFrame[0] == null) {
		      stop();
		      break;
		    }
		    mRgbFrames[0] = new OpenCVFrameConverter.ToMat().convert(tempVideoFrame[0]);
		    if (mRgbFrames[0] == null) {
		      continue;
		    }
			// crop the frame as square one
			if (previewHeight > previewWidth){
				int delta = previewHeight - previewWidth;
				Rect rt0 = new Rect(0, delta/2, previewWidth, previewWidth);
				mCroppedRgb = new Mat(mRgbFrames[0], rt0);
			}else{
				int delta = previewWidth - previewHeight;
				Rect rt0 = new Rect(delta/2, 0, previewHeight, previewHeight);
				mCroppedRgb = new Mat(mRgbFrames[0], rt0);
			}
			// copy mCroppedRgb to mCroppedMat for liveness detection
			mCroppedRgb.copyTo(mCroppedMat);
			cvtColor(mCroppedRgb, mBluredGray, COLOR_BGR2GRAY);
			cvtColor(mBluredGray, mBluredFrame, COLOR_GRAY2BGR );
//			Imgproc.GaussianBlur(mRgbFrame, mBluredFrame, new Size(15, 15), 11.0);
			mBluredFrame.copyTo(mCroppedRgb, mMaskImg);
            putText(mCroppedRgb, "Please fit face in ellipse area... ",
            		new Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.7,
    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
					new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
					new Scalar(255, 0, 255, 0) , 2, LINE_8, 0);
			//faceDetector.getFaceBoxes(mCroppedRgb, 0.9f, true);
			faceBox facebox = faceDetector.extract_facebox(mCroppedRgb, 0.8f, true);
			if (facebox ==null) continue;
			boolean isFitInEllipse = checkfit(facebox);
			String resultString = "";
			if (isFitInEllipse) {
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(255, 0, 0, 0) , 2, LINE_8, 0);
				//int width = facebox.x_right - facebox.x_left;
				//int height = facebox.y_bottom - facebox.y_top;
				//Rect rectCrop = new Rect(facebox.x_left, facebox.y_top, width, height);
				//Mat face_img = new Mat(mCroppedMat, rectCrop);
				boolean liveness = DetectLivenessFromFrame(mCroppedMat);
				if (liveness) {
					resultString = "Real";
				}else {
					resultString = "Fake";
				}
				ellipse (mCroppedRgb, new Point(mEllipseCenterX, mEllipseCenterY), 
						new Size(mEllipseSizeX, mEllipseSizeY), 0, 0, 360, 
						new Scalar(0, 255, 0, 0) , 2, LINE_8, 0);
			}else {
				resultString = "Fit Face!";
			}
			
			putText(mCroppedRgb, resultString,
	            		new Point(mPreviewSize / 2 - 50, mPreviewSize - 30), CV_FONT_HERSHEY_SIMPLEX, 1,
	    				new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
			mainframe.showImage(converter.convert(mCroppedRgb));
			try {
				Thread.sleep(30);
			} catch (InterruptedException ex) {
				System.out.println(ex.getMessage());
			}
			//mCroppedRgb.release();
		}//while (!mStop && mainframe.isVisible()) 
		VIDEO_GRABBER.stop();
		VIDEO_GRABBER.release();
		VIDEO_GRABBER.close();
		mainframe.setVisible(false);
	}
	
	private static boolean checkfit(faceBox box) {
		float width = box.x_right - box.x_left;
		float height = box.y_bottom - box.y_top;
		float minWidth = mEllipseSizeX * 0.9f;
		float minHeight = mEllipseSizeY * 0.9f;
		float left = mEllipseCenterX - mEllipseSizeX * 0.9f;
		float right = mEllipseCenterX + mEllipseSizeX * 0.9f;
		float top = mEllipseCenterY - mEllipseSizeY * 0.9f;
		float bottom = mEllipseCenterY + mEllipseSizeY * 0.9f;
		if (box.x_left > left && box.x_right < right && box.y_top > top && box.y_bottom < bottom) {
			if (width > minWidth && height > minHeight){
				return true;
			}
		}
		return false;
	}
  
	public static void stop() {
        if (!mStop) {
        	mStop = true;
            //destroyAllWindows();
        }
    }
}
