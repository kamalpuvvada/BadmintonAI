using System.Text.Json;
using BadmintonAI.Integration;

if (args.Length != 1 && !(args.Length == 3 && args[0] == "--analysis"))
{
    Console.Error.WriteLine("Usage: BadmintonAI.SessionReader <result.json> | --self-test | --analysis <job folder> <analysis folder>");
    return 2;
}
try
{
    if (args[0] == "--self-test")
    {
        ContractChecks.Run();
        Console.WriteLine("Session contract checks passed.");
    }
    else if (args[0] == "--analysis")
    {
        Console.WriteLine(JsonSerializer.Serialize(AnalysisReader.Read(args[1], args[2]), new JsonSerializerOptions { WriteIndented = true }));
    }
    else
    {
        Console.WriteLine(JsonSerializer.Serialize(SessionReader.Read(args[0]), new JsonSerializerOptions { WriteIndented = true }));
    }
    return 0;
}
catch (Exception error) when (error is IOException or InvalidDataException or JsonException or InvalidOperationException or KeyNotFoundException or FormatException)
{
    Console.Error.WriteLine("Session import rejected: " + error.Message);
    return 1;
}
