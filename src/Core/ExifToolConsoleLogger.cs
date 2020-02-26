using CoenM.ExifToolLib.Logging;

namespace Core
{
    public class ExifToolConsoleLogger : ILogger
    {
        private readonly object syncLock = new object();

        public void Log(LogEntry entry)
        {
            lock (syncLock)
            {
                System.Console.WriteLine($"LOG => {entry.Severity} => {entry.Message}");
            }
        }

        public bool IsEnabled(LogLevel logLevel) => true;
    }
}