using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Windows.Forms;

namespace VoiceWorkbenchLauncher
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string pythonExe = Path.Combine(baseDir, "python", "python.exe");
            string launcherScript = Path.Combine(baseDir, "product_launcher.py");
            string runtimeDir = Path.Combine(baseDir, "runtime");
            string cacheDir = Path.Combine(baseDir, ".tts_cache");
            string url = "http://127.0.0.1:8000";

            if (IsServerUp("127.0.0.1", 8000))
            {
                OpenUrl(url);
                return;
            }

            if (!File.Exists(pythonExe))
            {
                MessageBox.Show(
                    "Bundled Python runtime not found.\nExpected: " + pythonExe,
                    "VoiceWorkbench",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            if (!File.Exists(launcherScript))
            {
                MessageBox.Show(
                    "Launcher script not found.\nExpected: " + launcherScript,
                    "VoiceWorkbench",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            Directory.CreateDirectory(runtimeDir);

            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = "\"" + launcherScript + "\"",
                WorkingDirectory = baseDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };

            psi.EnvironmentVariables["PYTHONHOME"] = Path.Combine(baseDir, "python");
            psi.EnvironmentVariables["PYTHONPATH"] = baseDir;
            psi.EnvironmentVariables["VOICE_WORKBENCH_HOME"] = runtimeDir;
            psi.EnvironmentVariables["VOICE_CLONE_CACHE_DIR"] = cacheDir;
            psi.EnvironmentVariables["VOICE_WORKBENCH_OPEN_BROWSER"] = "1";

            try
            {
                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Unable to start VoiceWorkbench.\n\n" + ex.Message,
                    "VoiceWorkbench",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            for (int attempt = 0; attempt < 40; attempt++)
            {
                if (IsServerUp("127.0.0.1", 8000))
                {
                    break;
                }
                Thread.Sleep(500);
            }

            OpenUrl(url);
        }

        private static bool IsServerUp(string host, int port)
        {
            try
            {
                using (TcpClient client = new TcpClient())
                {
                    IAsyncResult result = client.BeginConnect(host, port, null, null);
                    bool success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromMilliseconds(300));
                    if (!success)
                    {
                        return false;
                    }
                    client.EndConnect(result);
                    return true;
                }
            }
            catch
            {
                return false;
            }
        }

        private static void OpenUrl(string url)
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = url,
                    UseShellExecute = true
                });
            }
            catch
            {
                // Best effort only.
            }
        }
    }
}
