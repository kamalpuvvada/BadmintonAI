using System.Security.Cryptography;
using System.Text.Json;

namespace BadmintonAI.Integration;

public sealed record SessionImportSummary(
    string SessionId, string ProcessingStatus, bool NeedsReview,
    int PreviewFrames, string SourceVideoSha256, string PreviewPath,
    IReadOnlyDictionary<string, string> MeasurementStatuses);

/// <summary>Read-only file boundary for a future RACKETEK worker adapter.</summary>
public static class SessionReader
{
    public static SessionImportSummary Read(string resultPath)
    {
        resultPath = Path.GetFullPath(resultPath);
        var root = Path.GetDirectoryName(resultPath)!;
        var bytes = File.ReadAllBytes(resultPath);
        using var document = JsonDocument.Parse(bytes);
        using var manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "job.json")));
        var result = document.RootElement;
        var job = manifest.RootElement;
        Require(Text(result, "schema_version") == "1.0.0" && Text(job, "schema_version") == "1.0.0",
            "Unsupported session/job schema version");
        var status = Text(result, "status");
        Require(status is "needs_review" or "completed", "Result is not a current processing snapshot");
        Require(Text(job, "status") == status, "Job and result status differ; refresh before importing");
        Require(Text(job.GetProperty("stages").GetProperty("export"), "status") == "completed",
            "The job has not published its result");
        var expectedHash = Text(job.GetProperty("stages").GetProperty("export").GetProperty("artifacts"), "result.json");
        Require(Convert.ToHexString(SHA256.HashData(bytes)).Equals(expectedHash, StringComparison.OrdinalIgnoreCase),
            "Result bytes do not match the completed job export");
        var sessionId = Text(result, "session_id");
        Require(sessionId == Text(job, "session_id"), "Session identity differs from job");
        var videoHash = Text(result.GetProperty("source"), "sha256");
        Require(videoHash.Length == 64 && videoHash.All(Uri.IsHexDigit), "Invalid source video hash");
        Require(videoHash == Text(job.GetProperty("binding").GetProperty("source"), "sha256"), "Source binding differs from job");
        var preview = result.GetProperty("preview");
        var frames = preview.GetProperty("frames").GetInt32();
        Require(frames > 0 && preview.GetProperty("decoded_preview_frames").GetInt32() == frames,
            "Preview decode was not verified for every frame");
        var fps = preview.GetProperty("fps").GetDouble();
        Require(double.IsFinite(fps) && fps > 0, "Invalid preview frame rate");
        foreach (var artifact in result.GetProperty("artifacts").EnumerateObject())
        {
            if (artifact.Value.ValueKind == JsonValueKind.Null) continue;
            var path = ContainedArtifact(root, artifact.Value.GetString()!);
            Require(File.Exists(path), $"Missing artifact: {artifact.Name}");
        }
        var measurements = new Dictionary<string, string>();
        foreach (var metric in result.GetProperty("metrics").EnumerateObject())
        {
            var metricStatus = Text(metric.Value, "status");
            if (metricStatus == "not_available")
                Require(metric.Value.GetProperty("value").ValueKind == JsonValueKind.Null,
                    "Unavailable measurements must be null, never zero");
            measurements.Add(metric.Name, metricStatus);
        }
        var previewPath = ContainedArtifact(root, Text(result.GetProperty("artifacts"), "preview"));
        // Completed means processing finished. This reader does not grant review
        // approval or derive coaching/physical metrics from temporary track IDs.
        return new(sessionId, status, status == "needs_review", frames, videoHash, previewPath, measurements);
    }

    internal static string ContainedArtifact(string root, string relative)
    {
        Require(!string.IsNullOrWhiteSpace(relative) && !relative.Contains(':') && !relative.StartsWith('/') && !relative.StartsWith('\\'),
            "Artifact must be a relative local path");
        var normalized = relative.Replace('\\', '/');
        Require(!normalized.Split('/').Any(part => part is ".." or "." or ""), "Artifact path contains traversal");
        var fullRoot = Path.GetFullPath(root);
        var full = Path.GetFullPath(Path.Combine(fullRoot, normalized.Replace('/', Path.DirectorySeparatorChar)));
        Require(full.StartsWith(fullRoot + Path.DirectorySeparatorChar, OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal), "Artifact escapes the job folder");
        // Reject links/reparse points in this import boundary, including ancestors.
        for (var current = full; current != fullRoot; current = Path.GetDirectoryName(current)!)
            if (File.Exists(current) || Directory.Exists(current))
                Require((File.GetAttributes(current) & FileAttributes.ReparsePoint) == 0, "Linked artifacts are not supported");
        return full;
    }

    private static string Text(JsonElement node, string name) =>
        node.GetProperty(name).GetString() ?? throw new InvalidDataException($"Missing text field: {name}");

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }
}
