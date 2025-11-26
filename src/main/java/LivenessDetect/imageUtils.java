package LivenessDetect;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

import org.opencv.core.CvType;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;


public class imageUtils {
	
	static OpenCVFrameConverter.ToMat toJavaCvMat = new OpenCVFrameConverter.ToMat();
	static OpenCVFrameConverter.ToOrgOpenCvCoreMat toOpenCvMat = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();
	
	public static BufferedImage mat2BufferedImage(org.opencv.core.Mat mat) {
		BufferedImage image = new BufferedImage(mat.width(), mat.height(), BufferedImage.TYPE_3BYTE_BGR);
		WritableRaster raster = image.getRaster();
		DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
		byte[] data = dataBuffer.getData();
		mat.get(0, 0, data);
		return image;
	}
	
	public static org.opencv.core.Mat bufferedImage2Mat(BufferedImage bi) {
		org.opencv.core.Mat mat = new org.opencv.core.Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		mat.put(0, 0, data);
		return mat;
	}
	
	public static org.bytedeco.opencv.opencv_core.Mat bufferedImage2JavaCvMat(BufferedImage bi) {
        OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat();
        return cv.convertToMat(new Java2DFrameConverter().convert(bi));
        // return  Java2DFrameUtils.toMat(bi); // https://github.com/bytedeco/javacv/blob/master/src/main/java/org/bytedeco/javacv/Java2DFrameUtils.java
    }
	// not worked well in gradle project due to
	// implementation 'org.opencv:opencv:4.5.2' -error
	public static org.opencv.core.Mat javaCvMat2openCvMat(org.bytedeco.opencv.opencv_core.Mat javaCvMat) {
		org.opencv.core.Mat openCvMat = toOpenCvMat.convert(toJavaCvMat.convert(javaCvMat));
		return openCvMat;
	}
	// not worked well in gradle project due to
	// implementation 'org.opencv:opencv:4.5.2' -error
	public static org.bytedeco.opencv.opencv_core.Mat openCvMat2javaCvMat(org.opencv.core.Mat openCvMat) {
		org.bytedeco.opencv.opencv_core.Mat javaCvMat = toJavaCvMat.convert(toOpenCvMat.convert(openCvMat));
		return javaCvMat;
	}
	

}
