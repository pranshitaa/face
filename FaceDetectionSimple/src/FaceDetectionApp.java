import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

public class FaceDetectionApp {
    public static void main(String[] args) throws Exception {
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
        grabber.start();

        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt.xml");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        CanvasFrame canvas = new CanvasFrame("Face Detection", CanvasFrame.getDefaultGamma() / grabber.getGamma());
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);

        while (canvas.isVisible()) {
            Frame frame = grabber.grab();
            Mat mat = converter.convert(frame);

            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(mat, faces);

            for (int i = 0; i < faces.size(); i++) {
                Rect face = faces.get(i);
                rectangle(mat, face, Scalar.RED);
            }

            canvas.showImage(converter.convert(mat));
        }

        grabber.stop();
        canvas.dispose();
    }
}
