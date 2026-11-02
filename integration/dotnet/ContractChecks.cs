using System.Security.Cryptography;
using System.Text.Json;

namespace BadmintonAI.Integration;

internal static class ContractChecks
{
    public static void Run()
    {
        var root = Path.Combine(Path.GetTempPath(), "badminton-contract-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            File.WriteAllText(Path.Combine(root, "preview.mp4"), "synthetic artifact, not media");
            var hash = new string('a', 64);
            var metrics = new Dictionary<string, object> { ["coaching"] = new { status = "not_available", value = (object?)null } };
            var result = new Dictionary<string, object>
            {
                ["schema_version"] = "1.0.0", ["session_id"] = "test", ["status"] = "needs_review",
                ["source"] = new { sha256 = hash }, ["preview"] = new { frames = 10, decoded_preview_frames = 10, fps = 30 },
                ["artifacts"] = new { preview = "preview.mp4" }, ["metrics"] = metrics
            };
            void Save(string jobStatus = "needs_review")
            {
                var bytes = JsonSerializer.SerializeToUtf8Bytes(result);
                File.WriteAllBytes(Path.Combine(root, "result.json"), bytes);
                File.WriteAllText(Path.Combine(root, "job.json"), JsonSerializer.Serialize(new
                {
                    schema_version = "1.0.0", session_id = "test", status = jobStatus, binding = new { source = new { sha256 = hash } },
                    stages = new { export = new { status = "completed", artifacts = new Dictionary<string, string> {
                        ["result.json"] = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant() } } }
                }));
            }
            Save();
            var imported = SessionReader.Read(Path.Combine(root, "result.json"));
            if (!imported.NeedsReview || imported.MeasurementStatuses["coaching"] != "not_available") throw new InvalidOperationException("Partial results lost review status");
            void Reject(Action action)
            {
                try { action(); }
                catch (InvalidDataException) { return; }
                throw new InvalidOperationException("Invalid fixture was accepted");
            }
            result["schema_version"] = "9.0.0"; Save(); Reject(() => SessionReader.Read(Path.Combine(root, "result.json")));
            result["schema_version"] = "1.0.0"; result["status"] = "stale"; Save(); Reject(() => SessionReader.Read(Path.Combine(root, "result.json")));
            result["status"] = "needs_review"; Save("running"); Reject(() => SessionReader.Read(Path.Combine(root, "result.json")));
            Save(); File.AppendAllText(Path.Combine(root, "result.json"), " "); Reject(() => SessionReader.Read(Path.Combine(root, "result.json")));
            metrics["coaching"] = new { status = "not_available", value = 0 }; Save(); Reject(() => SessionReader.Read(Path.Combine(root, "result.json")));
            foreach (var path in new[] { "../outside.mp4", "C:/outside.mp4", "https://example.com/video.mp4", "/outside.mp4", "folder\\..\\outside.mp4" })
                Reject(() => SessionReader.ContainedArtifact(root, path));
        }
        finally { Directory.Delete(root, recursive: true); }
    }
}
