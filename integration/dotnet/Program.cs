using System.Text.Json;
using BadmintonAI.Integration;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: BadmintonAI.SessionReader <result.json> | --self-test");
    return 2;
}
try
{
    if (args[0] == "--self-test")
    {
        ContractChecks.Run();
        Console.WriteLine("Session contract checks passed.");
    }
    else
    {
        Console.WriteLine(JsonSerializer.Serialize(SessionReader.Read(args[0]), new JsonSerializerOptions { WriteIndented = true }));
    }
    return 0;
}
catch (Exception error) when (error is IOException or JsonException or InvalidOperationException or KeyNotFoundException or FormatException)
{
    Console.Error.WriteLine("Session import rejected: " + error.Message);
    return 1;
}
