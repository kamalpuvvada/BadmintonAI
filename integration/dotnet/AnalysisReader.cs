using System.Security.Cryptography;
using System.Text.Json;

namespace BadmintonAI.Integration;

public sealed record AnalysisImportSummary(string SchemaVersion, string SessionId, int Revision,
    string? ReviewerKind, int AcceptedContacts, int AcceptedShotClasses,
    int AnnotatedRallyPointsA, int AnnotatedRallyPointsB, string CoachingStatus);

/// <summary>Imports a separate version 2 analysis snapshot, never upgrades a v1 tracking result.</summary>
public static class AnalysisReader
{
    public static AnalysisImportSummary Read(string jobFolder, string analysisFolder)
    {
        jobFolder = Path.GetFullPath(jobFolder);
        analysisFolder = Path.GetFullPath(analysisFolder);
        var session = SessionReader.Read(Path.Combine(jobFolder, "result.json"));
        using var job = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(jobFolder, "job.json")));
        using var manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(analysisFolder, "analysis_manifest.json")));
        var manifestRoot = manifest.RootElement;
        Check(manifestRoot.GetProperty("schema_version").GetString() == "2.0.0", "Unsupported analysis manifest schema");
        var artifacts = manifestRoot.GetProperty("artifacts");
        foreach (var required in new[] { "analysis.json", "report.json", "preview.mp4", "index.html" })
            Check(artifacts.TryGetProperty(required, out _), "Incomplete analysis manifest");
        foreach (var artifact in artifacts.EnumerateObject())
        {
            var path = SessionReader.ContainedArtifact(analysisFolder, artifact.Name);
            var actual = HashFile(path);
            Check(actual.Equals(artifact.Value.GetString(), StringComparison.OrdinalIgnoreCase), "Analysis artifact changed: " + artifact.Name);
        }
        using var packet = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(analysisFolder, "analysis.json")));
        using var report = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(analysisFolder, "report.json")));
        var ledger = packet.RootElement;
        var summary = report.RootElement;
        Check(ledger.GetProperty("schema_version").GetString() == "2.0.0" && summary.GetProperty("schema_version").GetString() == "2.0.0",
            "Unsupported analysis packet/report schema");
        var source = ledger.GetProperty("source_binding");
        foreach (var key in new[] { "source_video_sha256", "session_id", "summary_sha256", "tracks_sha256" })
            Check(source.GetProperty(key).GetString() == manifestRoot.GetProperty("source_binding").GetProperty(key).GetString(), "Analysis manifest source differs from packet");
        Check(source.GetProperty("source_video_sha256").GetString() == session.SourceVideoSha256 && source.GetProperty("session_id").GetString() == session.SessionId,
            "Analysis belongs to a different source/session");
        var players = job.RootElement.GetProperty("stages").GetProperty("players");
        var playerFolder = SessionReader.ContainedArtifact(jobFolder, players.GetProperty("directory").GetString()!);
        foreach (var item in new[] { ("summary_sha256", "tracking_summary.json"), ("tracks_sha256", "player_tracks.jsonl") })
        {
            var file = SessionReader.ContainedArtifact(playerFolder, item.Item2);
            var hash = HashFile(file);
            Check(hash.Equals(source.GetProperty(item.Item1).GetString(), StringComparison.OrdinalIgnoreCase), "Analysis player observations are stale");
            Check(hash.Equals(players.GetProperty("artifacts").GetProperty(item.Item2).GetString(), StringComparison.OrdinalIgnoreCase), "Job player artifacts changed");
        }
        var previewHash = HashFile(Path.Combine(analysisFolder, "preview.mp4"));
        Check(previewHash.Equals(ledger.GetProperty("source").GetProperty("preview_sha256").GetString(), StringComparison.OrdinalIgnoreCase), "Analysis preview binding differs");
        Check(previewHash.Equals(job.RootElement.GetProperty("stages").GetProperty("render").GetProperty("artifacts").GetProperty("preview.mp4").GetString(), StringComparison.OrdinalIgnoreCase),
            "Analysis preview differs from the source job");
        Check(ledger.GetProperty("source").GetProperty("frames").GetInt32() == session.PreviewFrames, "Analysis timebase differs from job");
        var kind = ledger.GetProperty("reviewer_kind").GetString();
        Check(kind is null or "human" or "automated", "Unknown reviewer provenance");
        Check(summary.GetProperty("reviewer_kind").GetString() == kind, "Report reviewer provenance differs");
        var coverage = summary.GetProperty("coverage");
        var contacts = coverage.GetProperty("accepted_contacts").GetInt32();
        var classes = coverage.GetProperty("accepted_shot_classes").GetInt32();
        Check(contacts >= 0 && classes >= 0, "Negative annotation counts");
        var points = summary.GetProperty("accepted_rally_points");
        var a = points.TryGetProperty("A", out var pa) ? pa.GetInt32() : 0;
        var b = points.TryGetProperty("B", out var pb) ? pb.GetInt32() : 0;
        Check(a >= 0 && b >= 0, "Negative point annotations");
        Check(kind is not null || (contacts == 0 && classes == 0 && a == 0 && b == 0), "Accepted annotations have no reviewer provenance");
        var coaching = summary.GetProperty("coaching_metrics").GetProperty("status").GetString()
            ?? throw new InvalidDataException("Missing coaching availability status");
        Check(coaching == "not_available", "This contract does not authorize coaching metrics");
        return new("2.0.0", session.SessionId, ledger.GetProperty("revision").GetInt32(), kind, contacts, classes, a, b, coaching);
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }

    private static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }
}
