using System;
using System.Threading.Tasks;

namespace Core
{
    public interface IFaceDetection
    {
        string Name { get; }

        Task<int> ProcessAsync(string inputFilename, string outputDirectory);
    }
}
