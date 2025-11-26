package LivenessDetect;

import java.io.File;
import java.util.ArrayList;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FaceDetector {
	private static final Logger logger = LoggerFactory.getLogger(LivenessDetector.class);
	private static final String PROTO_FILE = "deploy.prototxt";
    private static final String CAFFE_MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel";
    private String modelDir = "/home/model/";
    private static Net net = null;
    
    public FaceDetector(String modelDir)
    {
    	this.modelDir = modelDir;
    }
    
	public boolean init()
	{
    	// https://github.com/deepjavalibrary/djl/issues/147
    	// http://docs.djl.ai/docs/load_model.html#implement-your-own-repository
    	try {
    		File dir = new File(modelDir);
    		if (!dir.exists()) {
    			System.out.println("Invalid model dir, " + modelDir);
    			logger.debug("Invalid model dir, " + modelDir);
    			return false;
    		}
    		if(!dir.isDirectory()) { 
    			System.out.println("Please specify full path of model directory, not file, " + modelDir);
    			logger.debug("Please specify full path of model directory, not file, " + modelDir);
    			return false;
    		}
    		String proto_path = modelDir + "/" + PROTO_FILE;
    		String caffe_model_path = modelDir + "/" + CAFFE_MODEL_FILE; 
        	
    		net = readNetFromCaffe(proto_path, caffe_model_path);
        	
    	}catch(Exception e)
    	{
    		e.printStackTrace();
    		logger.debug(e.toString());
    		return false;
    	}
		return true;
    }

    public static void detectAndDraw(Mat image) {//detect faces and draw a blue rectangle arroung each face

        resize(image, image, new Size(300, 300));//resize the image to match the input size of the model

        //create a 4-dimensional blob from image with NCHW (Number of images in the batch -for training only-, Channel, Height, Width) dimensions order,
        //for more detailes read the official docs at https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
        Mat blob = blobFromImage(image, 1.0, new Size(300, 300), 
        		new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        net.setInput(blob);//set the input to network model
        Mat output = net.forward();//feed forward the input to the netwrok to get the output matrix

        Mat ne = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));//extract a 2d matrix for 4d output matrix with form of (number of detections x 7)

        FloatIndexer srcIndexer = ne.createIndexer(); // create indexer to access elements of the matric

        for (int i = 0; i < output.size(3); i++) {//iterate to extract elements
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > .6) {
                float tx = f1 * 300;//top left point's x
                float ty = f2 * 300;//top left point's y
                float bx = f3 * 300;//bottom right point's x
                float by = f4 * 300;//bottom right point's y
                rectangle(image, new Rect(new Point((int) tx, (int) ty), new Point((int) bx, (int) by)), 
                		new Scalar(255, 0, 0, 0));//print blue rectangle 
            }
        }
    }
	
	public ArrayList<faceBox> getFaceBoxes(Mat image, float threshold, boolean bDraw) {
		//   Get the bounding box of faces in image using dnn.
		int cols = image.cols();
		int rows = image.rows();
        //create a 4-dimensional blob from image with NCHW (Number of images in the batch -for training only-, Channel, Height, Width) dimensions order,
        //for more detailes read the official docs at https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
        Mat blob = blobFromImage(image, 1.0, new Size(300, 300), 
        		new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        net.setInput(blob);//set the input to network model
        Mat output = net.forward();//feed forward the input to the netwrok to get the output matrix

        Mat ne = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));//extract a 2d matrix for 4d output matrix with form of (number of detections x 7)

        FloatIndexer srcIndexer = ne.createIndexer(); // create indexer to access elements of the matric
        ArrayList<faceBox> faceBoxes = new ArrayList<faceBox>();
        for (int i = 0; i < output.size(3); i++) {//iterate to extract elements
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > threshold) {
                float tx = f1 * cols;//top left point's x
                float ty = f2 * rows;//top left point's y
                float bx = f3 * cols;//bottom right point's x
                float by = f4 * rows;//bottom right point's y
                faceBox box = new faceBox((int) tx, (int) ty, (int) bx, (int) by);
                if (bDraw)
                	rectangle(image, new Rect(new Point((int) tx, (int) ty), new Point((int) bx, (int) by)), 
                			new Scalar(255, 0, 0, 0));//print blue rectangle
                faceBoxes.add(box);
            }
        }
        return faceBoxes;
    }

    public boolean box_in_image(faceBox box, Mat image) {
        // """Check if the box is in image"""
        int rows = image.rows();
        int cols = image.cols();
        return box.x_left >= 0 && box.y_top >= 0 && box.x_right <= cols && box.y_bottom <= rows;
    }

    public faceBox move_box(faceBox box, int off_x, int off_y)
    {
        // """Move the box to direction specified by vector offset"""
        int left_x = box.x_left + off_x;
        int top_y = box.y_top + off_y;
        int right_x = box.x_right + off_x;
        int bottom_y = box.y_bottom + off_y;
        return new faceBox(left_x, top_y, right_x, bottom_y);
    }

    public faceBox get_square_box(faceBox box)
    {
    	// """Get a square box out of the given box, by expanding it."""
        int left_x = box.x_left;
        int top_y = box.y_top;
        int right_x = box.x_right;
        int bottom_y = box.y_bottom;

        int box_width = right_x - left_x;
        int box_height = bottom_y - top_y;

        // Check if box is already a square. If not, make it a square.
        int diff = box_height - box_width;
        int delta = (int)(Math.abs(diff) / 2);

        if (diff == 0) //               // Already a square.
            return box;
        else if (diff > 0)				// Height > width, a slim box.
        {                  
            left_x -= delta;
            right_x += delta;
            if (diff % 2 == 1)
                right_x += 1;
        }
        else
        {                         		// Width > height, a short box.
            top_y -= delta;
            bottom_y += delta;
            if (diff % 2 == 1)
                bottom_y += 1;
        }
        // Make sure box is always square.
        assert((right_x - left_x) == (bottom_y - top_y));
        return new faceBox(left_x, top_y, right_x, bottom_y);
    }
        

	public faceBox extract_cnn_facebox(Mat image, float threshold, boolean bDraw)
	{
		// Extract face area from image.
		ArrayList<faceBox> raw_boxes = getFaceBoxes(image, threshold, true);
		for (int i=0; i< raw_boxes.size(); i++ ) {
			// Move box down.
			faceBox box = raw_boxes.get(i);
			int height = box.y_bottom - box.y_top;
            int offset_y = (int) Math.abs(height * 0.12f);
            faceBox box_moved = move_box(box, 0, offset_y);
            // Make box square.
            faceBox facebox = get_square_box(box_moved);
            if (box_in_image(facebox, image)) {
                if (bDraw) {
                	Point pt1 = new Point(facebox.x_left, facebox.y_top);
                	Point pt2 = new Point(facebox.x_right, facebox.y_bottom);
                	rectangle(image, new Rect(pt1, pt2), new Scalar(0, 255, 0, 0));
                }

                return facebox;
            }
		}
        return null;
	}
	
	public faceBox extract_facebox(Mat image, float threshold, boolean bDraw)
	{
		// Extract face area from image.
		ArrayList<faceBox> raw_boxes = getFaceBoxes(image, threshold, false);
		for (int i=0; i< raw_boxes.size(); i++ ) {
			// Move box down.
			faceBox box = raw_boxes.get(i);
			int height = box.y_bottom - box.y_top;
            int offset_y = (int) Math.abs(height * 0.1f);
            offset_y = -offset_y;
            faceBox facebox = move_box(box, 0, offset_y);
            if (box_in_image(facebox, image)) {
                if (bDraw)
                {
                	Point pt1 = new Point(facebox.x_left, facebox.y_top);
                	Point pt2 = new Point(facebox.x_right, facebox.y_bottom);
                	rectangle(image, new Rect(pt1, pt2), new Scalar(0, 255, 0, 0));
                }
                return facebox;
            }
		}
        return null;
	}
	
	
        
}
