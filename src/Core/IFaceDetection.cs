using System;

namespace Core
{
    public interface IFaceDetection
    {
        string Name { get; }

        int Process(string inputFilename, string outputDirectory);
    }
}
