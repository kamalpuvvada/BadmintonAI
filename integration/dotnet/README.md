# RACKETEK session result boundary

This small .NET 8 console adapter reads the Python worker's version 1.0.0 file
contract. It is an integration starting point, not a deployed RACKETEK service.
It does not create users, sessions or permissions in the actual backend.

```powershell
Push-Location integration/dotnet
dotnet run -- --self-test
dotnet run -- ../../out/academy-session-002/result.json
Pop-Location
```

The reader requires a matching, finished `job.json` export, verifies the exact
result SHA-256, rejects stale/running/failed snapshots and unsupported schemas,
preserves `needs_review`, and checks that referenced local artifacts exist inside
the job folder. It rejects artifact traversal and links. It does not verify every
media byte against the stage manifest or continuously monitor review revocation;
resume the Python job after review changes and import the newly published result.
Unavailable metrics remain null. Processing completion never grants permission to
publish coaching measurements.

Keep long inference in a separate worker process. The real RACKETEK host should
launch or queue the worker using its own academy/session authorization, observe
job status, and import a completed snapshot. Its hosting/queue/database contracts
must be inspected before implementing that host-specific adapter.

The directory pins the .NET 8 SDK family and clears package feeds because this
adapter uses only the installed framework. Run from this directory so its SDK
selection applies. No external NuGet packages are needed.
