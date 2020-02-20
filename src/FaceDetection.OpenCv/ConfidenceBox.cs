using OpenCvSharp;

namespace FaceDetection.OpenCv
{
    public readonly struct ConfidenceBox
    {
        public ConfidenceBox(Point p1, Point p2, float confidence)
        {
            P1 = p1;
            P2 = p2;
            Confidence = confidence;
        }

        public Point P1 { get; }

        public Point P2 { get; }

        public float Confidence { get; }
    }
}