Using Powershell in Windows, how can I add a new bin to my PATH environment variable?

You can add a new bin to your PATH environment variable in PowerShell by following these steps:

1. Open PowerShell with administrative privileges.
2. Use the `$env:Path` command to view your current PATH.
3. To add a new path, use the following command:

```powershell
$env:Path += ";C:\your\new\path"
```

Replace `C:\your\new\path` with the path you want to add. 4. To make this change permanent, you need to set it in the system environment variables, which you can do with the following command:

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)
```

Please note that you need to restart your PowerShell session (or any other command line sessions) to see the changes.

Remember, modifying the PATH incorrectly can cause programs to stop working, so be careful when making changes. Always make sure to backup your current PATH before making any modifications.
