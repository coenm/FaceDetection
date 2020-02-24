using System.Drawing;
using OpenCvSharp;
using Point = OpenCvSharp.Point;

namespace FaceDetection.OpenCv.OpenCvHelpers
{
    internal static class FaceBoxer
    {
        // private static readonly Scalar green = new Scalar(0, 255, 0);

        public static void Draw(Mat frame, Point p1, Point p2, float confidence, Color color)
        {
            var thickness = (int)(5 * confidence);
            if (thickness <= 0)
                thickness = 1;

            Scalar c = new Scalar(color.R, color.G, color.B);
            Cv2.Rectangle(frame, p1, p2, c, thickness, LineTypes.Link4);
            var text = $"{confidence:N}";

            // Cv2.PutText(frame, text, new Point(x1, y1), HersheyFonts.HersheySimplex, 10, Scalar.Green, 5, LineTypes.AntiAlias, false);
            // Cv2.PutText(frame, text, new Point(x2, y2), HersheyFonts.HersheySimplex, 1, Scalar.Red, 5, LineTypes.AntiAlias, false);
            // Cv2.PutText(frame, text, new Point(x2, y1), HersheyFonts.HersheySimplex, 2, Scalar.BlueViolet, 5, LineTypes.AntiAlias, false);

            // var p1s = $"({p1.X} , {p1.Y})";
            // Cv2.PutText(frame, p1s, p1, HersheyFonts.HersheySimplex, 0.5, Scalar.Red, 2, LineTypes.AntiAlias, false);
            //
            // p1 = new Point(x2, y2);
            // p1s = $"({p1.X} , {p1.Y})";
            // Cv2.PutText(frame, p1s, p1, HersheyFonts.HersheySimplex, 0.5, green, 2, LineTypes.AntiAlias, false);

            var pText = new Point(p1.X + 10, p2.Y - 10);
            Cv2.PutText(frame, text, pText, HersheyFonts.HersheySimplex, 0.5, c, 2, LineTypes.AntiAlias, false);

        }
    }
}